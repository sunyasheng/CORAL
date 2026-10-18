"""Structural sanity checks, deliberately separate from pose accuracy."""
import json
from pathlib import Path

import numpy as np
from biotite.structure.io import pdb, pdbx
from scipy.spatial import cKDTree


def read_structure(path):
    if Path(path).suffix == ".cif":
        return pdbx.get_structure(pdbx.CIFFile.read(path), model=1)
    return pdb.PDBFile.read(path).get_structure(model=1)


def align_to_reference(atoms, receptor, refmap):
    """Use explicit chain/residue map and atom names, never guessed chain matching."""
    lookup = {(str(x.chain_id), int(x.res_id), str(x.atom_name)): x.coord for x in receptor}
    mapping = {e["json_chain"]: e for e in refmap["copies"]}
    moving, target, indices = [], [], []
    for i, atom in enumerate(atoms):
        e = mapping.get(str(atom.chain_id))
        if e is None:
            continue
        rid = e["positions"].get(str(int(atom.res_id)))
        key = (e["chain"], rid, str(atom.atom_name))
        if key in lookup:
            moving.append(atom.coord)
            target.append(lookup[key])
            indices.append(i)
    if len(moving) < 3:
        raise ValueError("Fewer than 3 explicit reference atoms matched")
    moving, target = np.asarray(moving, dtype=float), np.asarray(target, dtype=float)
    mc, tc = moving.mean(0), target.mean(0)
    u, _, vt = np.linalg.svd((moving - mc).T @ (target - tc))
    d = np.eye(3)
    d[2, 2] = np.linalg.det(u @ vt)
    rotation = u @ d @ vt
    aligned = (atoms.coord - mc) @ rotation + tc
    delta = aligned[indices] - target
    return aligned, {"matched_atoms": len(indices),
                     "rmsd_A": float(np.sqrt((delta ** 2).sum(1).mean())),
                     "max_error_A": float(np.linalg.norm(delta, axis=1).max())}


def validate_case(output, name, receptor_path, seed=101, expected=5):
    output, receptor_path = Path(output), Path(receptor_path)
    pred = output / name / f"seed_{seed}" / "predictions"
    files = sorted(pred.glob(f"{name}_sample_*.cif"))
    receptor = read_structure(receptor_path)
    refmap = json.loads(receptor_path.with_suffix(".refmap.json").read_text())
    protein_chains = {e["json_chain"] for e in refmap["copies"]}
    results = []
    for file in files:
        item = {"file": str(file.relative_to(output))}
        try:
            aa = read_structure(file)
            finite = bool(np.isfinite(aa.coord).all())
            if not finite:
                raise ValueError("Nonfinite coordinates")
            aligned, pin = align_to_reference(aa, receptor, refmap)
            protmask = np.isin(aa.chain_id, list(protein_chains))
            heavy = ~np.isin(aa.element, ["H", "D"])
            ligand_mask = ~protmask & heavy
            tree = cKDTree(aligned[protmask & heavy])
            ligands = []
            for chain in sorted(set(aa.chain_id[ligand_mask])):
                mask = ligand_mask & (aa.chain_id == chain)
                coords = aligned[mask]
                distances = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
                np.fill_diagonal(distances, np.inf)
                nn = float(distances.min()) if len(coords) > 1 else None
                contacts = tree.query(coords)[0]
                ligands.append({"chain": str(chain), "heavy_atoms": int(mask.sum()),
                                "min_internal_distance_A": nn,
                                "internal_pairs_below_0_9_A": int((distances < 0.9).sum() // 2),
                                "min_protein_distance_A": float(contacts.min()),
                                "atoms_with_protein_below_1_5_A": int((contacts < 1.5).sum())})
            runtime_ok = bool(ligands) and pin["rmsd_A"] < 0.01 and pin["max_error_A"] < 0.05
            item.update(finite=finite, atom_count=len(aa), pinning=pin, ligands=ligands,
                        runtime_ok=runtime_ok,
                        geometry_flag=any(x["internal_pairs_below_0_9_A"] or x["atoms_with_protein_below_1_5_A"] for x in ligands))
        except Exception as e:
            item.update(runtime_ok=False, error=str(e))
        results.append(item)
    err = output / "ERR" / f"{name}.txt"
    return {"case": name, "expected_samples": expected, "cif_samples": len(files),
            "runtime_ok": len(files) == expected and all(x["runtime_ok"] for x in results),
            "geometry_flagged_samples": sum(x.get("geometry_flag", False) for x in results),
            "error_file_present": err.is_file(), "samples": results}


def validate_outputs(output, case, seed, expected, mode):
    """Check exact sample IDs, confidence, atom identities and matching PDB/SDF coordinates."""
    import math
    import re
    from inputs import entities_by_chain
    from rdkit import Chem

    def finite_json(value):
        if isinstance(value, float):
            return math.isfinite(value)
        if isinstance(value, dict):
            return all(finite_json(x) for x in value.values())
        if isinstance(value, list):
            return all(finite_json(x) for x in value)
        return True

    output = Path(output)
    name = case['name']
    pred = output / name / f'seed_{seed}' / 'predictions'
    ext = 'cif' if mode == 'fibril' else 'pdb'
    expected_files = {f'{name}_sample_{i}.{ext}' for i in range(expected)}
    errors = []
    if {p.name for p in pred.glob(f'{name}_sample_*.{ext}')} != expected_files:
        errors.append('Missing or unexpected sample IDs')
    chains = entities_by_chain(case)
    ligand_chains = [c for c,(k,_) in chains.items() if k == 'ligand']
    rows = []; ranks = []; identity = None
    for i in range(expected):
        try:
            file = pred / f'{name}_sample_{i}.{ext}'
            aa = read_structure(file)
            if not len(aa) or not np.isfinite(aa.coord).all():
                raise ValueError('Empty or nonfinite structure')
            if set(aa.chain_id) != set(chains):
                raise ValueError('Output chains do not match requested protein/ligand copies')
            sig = list(zip(aa.chain_id.tolist(), aa.res_id.tolist(), aa.res_name.tolist(), aa.atom_name.tolist(), aa.element.tolist()))
            if len(set(sig)) != len(sig):
                raise ValueError('Duplicate atom identities')
            if identity is not None and identity != sig:
                raise ValueError('Atom identities differ across samples')
            identity = sig
            cf = json.loads((pred / f'{name}_summary_confidence_sample_{i}.json').read_text())
            if not finite_json(cf) or not isinstance(cf.get('ranking_score'), (int, float)):
                raise ValueError('Invalid confidence values')
            ranks.append(cf['ranking_score'])
            row = {'sample': i, 'atoms': len(aa), 'ranking_score': cf['ranking_score']}
            if mode == 'posebusters':
                sdf = pred / f'{name}_ligand_sample_{i}.sdf'
                mols = list(Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False))
                if len(mols) != len(ligand_chains) or any(m is None for m in mols):
                    raise ValueError('Missing/unreadable SDF ligand records')
                maxdelta = 0.
                for c,mol in zip(ligand_chains, mols):
                    part = aa[aa.chain_id == c]
                    xyz = mol.GetConformer().GetPositions()
                    elements = [a.GetSymbol().upper() for a in mol.GetAtoms()]
                    if elements != list(part.element) or not np.isfinite(xyz).all():
                        raise ValueError('SDF atom elements or coordinates do not match PDB')
                    delta = float(np.linalg.norm(xyz-part.coord, axis=1).max())
                    if delta > .002:
                        raise ValueError('SDF coordinates mismatch same-ranked PDB')
                    maxdelta = max(maxdelta, delta)
                row['pdb_sdf_max_atom_delta_A'] = maxdelta
            rows.append(row)
        except Exception as e:
            errors.append(f'sample_{i}: {e}')
    if ranks != sorted(ranks, reverse=True):
        errors.append('Confidence files are not ordered by ranking score')
    error_files = [str(p.relative_to(output)) for p in (output/'ERR').rglob('*') if p.is_file() and p.stat().st_size]
    if error_files:
        errors.append('Pipeline error files: ' + ', '.join(error_files))
    log = (output/'run.log').read_text()
    depths = [int(x) for x in re.findall(r'N_msa (\d+)', log)]
    if mode == 'posebusters':
        if not depths or min(depths) <= 1:
            errors.append('True MSA was not consumed by the model')
        if 'injected global_ref_pos' in log or 'Pinned ' in log:
            errors.append('Unexpected receptor pinning in blind mode')
    elif 'injected global_ref_pos from ref_pos' in log:
        errors.append('Fixed receptor fell back to generated reference coordinates')
    return {'case': name, 'runtime_ok': not errors, 'mode': mode, 'errors': errors,
            'msa_depths': depths, 'expected_samples': expected, 'samples': rows}
