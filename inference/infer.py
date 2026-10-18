#!/usr/bin/env python3
"""Run CORAL checkpoint 19999: fixed fibril docking or blind PoseBusters with MSA."""
import argparse
import hashlib
import json
import logging
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def write_receipt(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True, help="One-case input JSON")
    p.add_argument("--mode", choices=["fibril", "posebusters"], default="fibril")
    p.add_argument("--receptor", type=Path, help="Fibril mode: receptor PDB with adjacent .refmap.json")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "checkpoints/mix3_19999_inference.pt")
    p.add_argument("--data", type=Path, default=ROOT / "data")
    p.add_argument("--out", type=Path, required=True, help="New output directory; existing cases are refused")
    p.add_argument("--check-only", action="store_true", help="Validate inputs/environment without loading the model")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--cycles", type=int, default=10)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--kernels", choices=["optimized", "torch"], default="optimized",
                   help="optimized: fused LayerNorm + triattention + cuequivariance; torch: PyTorch CUDA kernels")
    a = p.parse_args()
    if min(a.samples, a.steps, a.cycles) < 1 or a.workers < 0:
        p.error("samples, steps and cycles must be positive; workers must be nonnegative")
    a.out = a.out.resolve()
    a.data = a.data.resolve()
    a.checkpoint = a.checkpoint.resolve()
    for file in (a.checkpoint, a.data / "components.cif", a.data / "components.cif.rdkit_mol.pkl"):
        if not file.is_file():
            p.error(f"Missing required file: {file}")
    from inputs import load_input, check_environment, EXPERIMENT_ENV
    jf = a.input.resolve()
    rec = a.receptor.resolve() if a.receptor else None
    refmap = rec.with_suffix(".refmap.json") if rec else None
    try:
        check_environment(os.environ)
        data, msa_files = load_input(jf, a.mode, rec)
        if rec and not rec.is_file():
            raise ValueError(f"Missing receptor: {rec}")
    except (OSError, ValueError, TypeError, KeyError) as e:
        p.error(str(e))
    input_case = data[0]
    name = input_case["name"]
    if a.out.exists() and any(a.out.iterdir()):
        p.error("Output directory must be empty or new")
    is_fibril = a.mode == "fibril"
    if a.check_only:
        print(f"Input valid: {name}; mode={a.mode}; MSA files={len(msa_files)}")
        return
    manifest = json.loads((ROOT / "assets.json").read_text())
    asset_paths = {"checkpoints/mix3_19999_inference.pt": a.checkpoint,
                   "data/components.cif": a.data / "components.cif",
                   "data/components.cif.rdkit_mol.pkl": a.data / "components.cif.rdkit_mol.pkl"}
    asset_hashes = {key: digest(path) for key,path in asset_paths.items()}
    for key,value in asset_hashes.items():
        if value != manifest[key]["sha256"]:
            p.error(f"Asset hash mismatch: {asset_paths[key]}; use the matching checkpoint/CCD bundle")

    # Set before importing vendor code; ignore unrelated caller environment settings.
    os.environ["PROTENIX_DATA_ROOT_DIR"] = str(a.data)
    os.environ["SUBSTRUCT_OFF"] = "1"
    os.environ.pop("PF_ID_SPLIT", None)  # each JSON explicitly declares PF and layer
    os.environ["LAYERNORM_TYPE"] = "fast_layernorm" if a.kernels == "optimized" else "torch"
    sys.path.insert(0, str(ROOT / "vendor"))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Inference requires an NVIDIA CUDA GPU.")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("This bf16 bundle requires an Ampere or newer NVIDIA GPU (compute capability >= 8.0).")
    from configs.configs_base import configs as base
    from configs.configs_data import data_configs
    from configs.configs_inference import inference_configs
    from configs.configs_model_type import model_configs
    from protenix.config import parse_configs
    from runner.inference import InferenceRunner, infer_predict
    from validation import validate_case, validate_outputs

    from runner.dumper import DataDumper
    from ligand_io import LigandTemplates
    import shutil

    class InputChemistryDumper(DataDumper):
        def _save_ligand_sdf(self, pred_coordinates, prediction_save_dir, sample_name,
                             atom_array, entity_poly_type, sorted_indices):
            templates = LigandTemplates(a.data)
            ranks = sorted_indices if sorted_indices is not None else range(len(pred_coordinates))
            for idx, rank in enumerate(ranks):
                atoms = atom_array.copy()
                atoms.coord = pred_coordinates[idx].cpu().numpy()
                dest = Path(prediction_save_dir) / f"{sample_name}_ligand_sample_{rank}.sdf"
                templates.write(dest, atoms, input_case)
                if int(rank) == 0:
                    shutil.copy2(dest, Path(prediction_save_dir) / f"{sample_name}_ligand.sdf")

    class ModelOnlyRunner(InferenceRunner):
        def predict(self, data, fix_protein_atoms=False):
            if fix_protein_atoms:
                feat = data["input_feature_dict"]
                if "global_ref_pos" not in feat or "is_structured_protein" not in feat or not feat["is_structured_protein"].any():
                    raise ValueError("Fixed receptor coordinates/mask missing; refusing fallback")
            return super().predict(data, fix_protein_atoms=fix_protein_atoms)

        def init_dumper(self, **kwargs):
            self.dumper = InputChemistryDumper(base_dir=self.dump_dir, **kwargs)

        def load_checkpoint(self):
            ckpt = torch.load(a.checkpoint, map_location="cpu", weights_only=True, mmap=True)
            if int(ckpt.get("step", -1)) != 19999:
                raise ValueError(f"Expected step 19999, got {ckpt.get('step')}")
            weights = {k.removeprefix("module."): v for k, v in ckpt["model"].items()}
            for k in ("relative_position_encoding.linear_no_bias.weight",
                      "diffusion_module.diffusion_conditioning.relpe.linear_no_bias.weight"):
                if tuple(weights[k].shape) != (128, 151):
                    raise ValueError(f"Wrong position encoding: {k}: {weights[k].shape}")
            self.model.load_state_dict(weights, strict=True)
            self.model.eval()
            logging.info("Loaded exact step 19999, strict=True, position weights both (128,151)")

    cfg = parse_configs({**base, "data": data_configs, **inference_configs},
                        fill_required_with_null=True)
    cfg.model_name = "protenix_base_constraint_v0.5.0"
    cfg.update(model_configs[cfg.model_name])
    cfg.load_checkpoint_path = str(a.checkpoint)
    cfg.dump_dir = str(a.out)
    cfg.seeds = [a.seed]
    cfg.num_workers = a.workers
    cfg.use_msa = not is_fibril
    cfg.data.msa.enable = not is_fibril
    cfg.fix_protein_atoms = is_fibril
    cfg.cif_file_path = str(rec) if rec else None
    if not is_fibril:
        cfg.data.AmyloidAtlas.constraint.enable = True
    cfg.inference_center_method = "protein"
    cfg.dump_structure_format = "cif" if is_fibril else "pdb"
    # Export the complete complex as CIF; optional SDF conversion is not needed.
    cfg.dump_ligand_sdf = not is_fibril
    cfg.model.N_cycle = a.cycles
    cfg.sample_diffusion.N_sample = a.samples
    cfg.sample_diffusion.N_step = a.steps
    cfg.model.relative_position_encoding.use_pf_level = True
    cfg.model.relative_position_encoding.layer_max = 5
    cfg.model.relative_position_encoding.s_max = 2
    cfg.model.relative_position_encoding.use_pf_flag = False
    cfg.model.constraint_embedder.contact_embedder.enable = True
    cfg.model.constraint_embedder.pocket_embedder.enable = True
    cfg.model.constraint_embedder.constraint_weight = 1.0
    cfg.triangle_attention = "triattention" if a.kernels == "optimized" else "torch"
    cfg.triangle_multiplicative = "cuequivariance" if a.kernels == "optimized" else "torch"
    a.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(), logging.FileHandler(a.out / "run.log")], force=True)
    receipt = {"status": "running", "checkpoint_sha256": asset_hashes["checkpoints/mix3_19999_inference.pt"], "assets_sha256": asset_hashes,
               "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
               "settings": {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
               "cases": [], "started_unix": time.time()}
    write_receipt(a.out / "run.json", receipt)
    try:
        receipt["experimental_overrides_absent"] = list(EXPERIMENT_ENV)
        receipt["msa_sha256"] = {str(p): digest(p) for p in msa_files}
        resolved_input = a.out / "resolved_input.json"
        resolved_input.write_text(json.dumps(data, indent=2) + "\n")
        receipt["resolved_input_sha256"] = digest(resolved_input)
        receipt["source_sha256"] = {str(p.relative_to(ROOT)): digest(p) for p in sorted((ROOT/"vendor").rglob("*.py"))}
        receipt["source_sha256"].update({p.name: digest(p) for p in ROOT.glob("*.py")})
        runner = ModelOnlyRunner(cfg)
        cfg.input_json_path = str(resolved_input)
        start = time.time()
        infer_predict(runner, cfg, fix_protein_atoms=is_fibril)
        case_report = validate_outputs(a.out, input_case, a.seed, a.samples, a.mode)
        if is_fibril:
            fixed = validate_case(a.out, name, rec, a.seed, a.samples)
            case_report["pinning"] = fixed
            case_report["runtime_ok"] &= fixed["runtime_ok"]
        case_report.update({"seconds": time.time() - start, "input_sha256": digest(jf)})
        if rec:
            case_report.update(receptor_sha256=digest(rec), refmap_sha256=digest(refmap))
        receipt["cases"].append(case_report)
        if not case_report["runtime_ok"]:
            raise RuntimeError(f"Missing/invalid outputs or pinning failure: {name}; inspect run.json and ERR/")
        receipt["output_sha256"] = {str(p.relative_to(a.out)): digest(p) for p in sorted((a.out/name).rglob("*")) if p.is_file()}
    except Exception as e:
        receipt.update(status="failed", error=str(e), finished_unix=time.time())
        write_receipt(a.out / "run.json", receipt)
        raise
    receipt.update(status="complete", finished_unix=time.time())
    write_receipt(a.out / "run.json", receipt)
    print(f"Completed {name}; see {a.out / 'run.json'}.")


if __name__ == "__main__":
    main()
