# CORAL inference

One checkpoint, two modes: ligand docking around a supplied amyloid fibril,
and blind protein–ligand prediction with precomputed MSA.
This release uses the **mix3 step-19999 checkpoint with PF/layer position encoding**.

[View sample predictions](examples/README.md): four fibril stacks and two protein–ligand complexes, with PyMOL comparisons, input files and downloadable outputs.

## Install and download

Linux, Python 3.11, NVIDIA Ampere or newer GPU, CUDA 12.x toolkit (`nvcc`)
and a C++ compiler are required. Tested on H100/H200; required memory depends
on complex size. First use compiles the LayerNorm kernel.

Let the first run finish compilation before starting parallel runs from the
same checkout. If a run is terminated during compilation, stop all runs using
that checkout before removing the stale `vendor/protenix/model/layer_norm/lock`
file, then restart one run to complete the build.

```bash
git clone https://github.com/sunyasheng/CORAL.git
cd CORAL/inference
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements-lock.txt
python download_assets.py
```

The downloader places the checkpoint in `checkpoints/` and CCD files in `data/`,
verifies SHA-256, and skips existing matching files. Total download: **2.10 GB**.
Files can also be downloaded from the [release](https://github.com/sunyasheng/CORAL/releases/tag/inference-v0.1.0);
`assets.json` records their exact versions, destinations and hashes.

## Fibril–ligand docking

```bash
python infer.py --mode fibril \
  --input examples/fibril/7WMM_g1/input.json \
  --receptor examples/fibril/7WMM_g1/receptor.pdb \
  --out outputs/fibril
```

This example predicts three ligand copies. Inputs contain sequence, ligand
CCD/SMILES, copy counts and PF/layer declarations. The receptor requires an
adjacent `receptor.refmap.json` mapping input chains/residues to the PDB.
Mapped resolved protein atoms stay fixed. This example includes receptor
contacts and binding-pocket hints. Such constraints are optional for your own
inputs; this mode does not use MSA.
Output complexes use a receptor-centered coordinate frame. Align the output
protein to the input receptor before overlaying them.

## Blind protein–ligand prediction (PoseBusters input)

```bash
python infer.py --mode posebusters \
  --input examples/posebusters/5SD5_HWI/input.json \
  --out outputs/posebusters
```

This mode predicts both protein and ligand. Supply sequence, one ligand,
and precomputed MSA; the example includes its A3M files. MSA paths resolve
relative to the input JSON. Every protein needs `msa.precomputed_msa_dir` and
`msa.pairing_db` (`uniref100` or `uniprot`), with `non_pairing.a3m`; heteromers
also need `pairing.a3m`. At least one protein must have real homologues.
This mode takes no reference receptor, ligand pose, pockets or PF/layer labels.
MSA search and PoseBusters scoring are separate from this prediction command.

## Your inputs and outputs

Use the examples as input templates. JSON contains a one-element list with
`name` and `sequences`; each entity is a `proteinChain` or `ligand` with
positive `count` (default 1). Ligands use `CCD_<id>` or SMILES.
Fibril proteins require `protofilament` (0/1/2 per copy) and nonnegative `layer`.
The refmap uses `version: 1`, matching `sequence`, and `copies` entries with
`json_chain`, PDB `chain`, and `positions` (1-based sequence position → PDB residue).
Blind mode supports one ligand copy and at most 26 total chains.
Add `--check-only` to validate inputs before loading the model.

Default: seed 101, five candidates, 200 diffusion steps, ten cycles.
Use `--seed`, `--samples`, `--steps`, `--cycles`, `--workers`, and
`--kernels torch` to configure a run. The output directory must be new or empty.

Structures are under `<out>/<name>/seed_101/predictions/`: complex CIF for
fibrils; complex PDB plus ligand SDF for blind prediction. `sample_0` is the
highest-confidence candidate, matched to its confidence JSON. `run.json`
records settings and output checks. Summary `plddt` uses a 0–100 scale;
`chain_plddt` and `chain_pair_plddt` use 0–1. A successful run confirms readable,
finite outputs and correct atom correspondence, not binding accuracy or
physical validity. Coordinates can vary across GPU/kernel/seed.

Model: `vendor/protenix/model/protenix.py`; position encoding:
`vendor/protenix/model/modules/embedders.py`. Code and checkpoint: Apache-2.0;
see [LICENSE](LICENSE) and [NOTICE](NOTICE) for upstream and data attribution.
