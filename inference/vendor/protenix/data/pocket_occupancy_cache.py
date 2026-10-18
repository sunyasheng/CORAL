# Modified for the CORAL inference distribution (2026); see inference/NOTICE.
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""
Pocket residues from docked MOL2 occupancy, using the **same** atom_array as training
(Biotite chain_id / res_id as in bioassembly PKL). Results are disk-cached under a
single directory keyed by (sample_pdb_id, pocket_id) so repeated samples skip geometry.

Logic matches metrics/analyze_pocket_residue_occupancy.py (distance cutoff, occupancy
fraction), but protein coordinates come from atom_array, not a separate PDB file.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from biotite.structure import AtomArray

try:
    from scipy.spatial import cKDTree

    HAS_SCIPY = True
except Exception:
    HAS_SCIPY = False


def _sanitize_cache_key(sample_pdb_id: str) -> str:
    s = str(sample_pdb_id).strip()
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:200] if len(s) > 200 else s


def _iter_mol2_ligands_coords(mol2_path: Path):
    """Yield (N,3) float32 arrays per @<TRIPOS>MOLECULE ligand (same as analyze script)."""
    coords: List[List[float]] = []
    in_atom_block = False
    with mol2_path.open("r") as f:
        for line in f:
            line_stripped = line.strip()
            if line_stripped.startswith("@<TRIPOS>MOLECULE"):
                if coords:
                    yield np.asarray(coords, dtype=np.float32)
                    coords = []
                in_atom_block = False
                continue
            if line_stripped.startswith("@<TRIPOS>ATOM"):
                in_atom_block = True
                continue
            if line_stripped.startswith("@<TRIPOS>"):
                in_atom_block = False
                continue
            if not in_atom_block:
                continue
            parts = line_stripped.split()
            if len(parts) < 5:
                continue
            try:
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                continue
            coords.append([x, y, z])
    if coords:
        yield np.asarray(coords, dtype=np.float32)


def _get_residues_within_cutoff(
    prot_coords: np.ndarray,
    prot_res_keys: List[Tuple[str, int, str]],
    lig_coords: np.ndarray,
    cutoff: float,
) -> List[Tuple[str, int, str]]:
    if prot_coords.size == 0 or lig_coords.size == 0:
        return []
    r = float(cutoff)
    if HAS_SCIPY:
        tree = cKDTree(prot_coords)
        hits = tree.query_ball_point(lig_coords, r=r)
        hit_indices = {idx for sub in hits for idx in sub}
    else:
        diff = prot_coords[None, :, :] - lig_coords[:, None, :]
        dist2 = np.sum(diff * diff, axis=-1)
        hit_indices = set(np.where(dist2 <= r * r)[1])
    residues = {prot_res_keys[i] for i in hit_indices}
    return sorted(residues)


def protein_coords_and_reskeys_from_atom_array(
    atom_array: AtomArray,
) -> Tuple[np.ndarray, List[Tuple[str, int, str]]]:
    """ATOM-like protein atoms only (same numbering as training featurizer)."""
    if hasattr(atom_array, "mol_type"):
        mask = np.asarray(atom_array.mol_type == "protein", dtype=bool)
    else:
        mask = ~np.asarray(atom_array.hetero, dtype=bool)
    idx = np.where(mask)[0]
    coords = atom_array.coord[idx].astype(np.float32)
    keys: List[Tuple[str, int, str]] = []
    for i in idx:
        keys.append(
            (
                str(atom_array.chain_id[i]),
                int(atom_array.res_id[i]),
                str(atom_array.res_name[i]),
            )
        )
    return coords, keys


def _aggregate_occupancy(
    mol2_path: Path,
    prot_coords: np.ndarray,
    prot_res_keys: List[Tuple[str, int, str]],
    cutoff: float,
) -> Tuple[int, Dict[Tuple[str, int, str], int]]:
    residue_hits: Dict[Tuple[str, int, str], int] = defaultdict(int)
    total = 0
    for lig_coords in _iter_mol2_ligands_coords(mol2_path):
        if lig_coords.size == 0:
            continue
        total += 1
        for rk in _get_residues_within_cutoff(
            prot_coords, prot_res_keys, lig_coords, cutoff
        ):
            residue_hits[rk] += 1
    return total, dict(residue_hits)


def _filter_occupancy(
    residue_hits: Dict[Tuple[str, int, str], int],
    total_ligands: int,
    occupancy_min: float,
) -> List[Dict[str, Any]]:
    if total_ligands <= 0:
        return []
    out: List[Dict[str, Any]] = []
    for (chain_id, res_id, res_name), hit_count in sorted(
        residue_hits.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        frac = hit_count / float(total_ligands)
        if frac >= float(occupancy_min):
            out.append(
                {
                    "chain_id": str(chain_id),
                    "res_id": int(res_id),
                    "res_name": str(res_name),
                }
            )
    return out


def _count_residues_matching(
    atom_array: AtomArray, residues: List[Dict[str, Any]]
) -> int:
    n_ok = 0
    for r in residues:
        if not isinstance(r, dict):
            continue
        try:
            cid = str(r.get("chain_id", ""))
            rid = int(r["res_id"])
        except (KeyError, TypeError, ValueError):
            continue
        mask = (atom_array.chain_id == cid) & (atom_array.res_id == rid)
        if np.any(mask):
            n_ok += 1
    return n_ok


def pick_mol2_for_pdb_key(
    ligand_dir: Path,
    pdb_key: str,
    generator: torch.Generator,
) -> Tuple[Optional[str], Optional[Path]]:
    """
    Pick one pocket id and a representative MOL2 path for this PDB prefix.
    Filenames: {pdb_key}_{pocket_id}_*_yes_ligands*.mol2
    """
    groups: Dict[str, List[Path]] = defaultdict(list)
    for p in ligand_dir.glob(f"{pdb_key}_*_yes_ligands*.mol2"):
        if not p.is_file():
            continue
        parts = p.stem.split("_")
        if len(parts) < 2:
            continue
        if parts[0].upper() != pdb_key.upper():
            continue
        pocket_id = parts[1]
        groups[pocket_id].append(p)
    if not groups:
        return None, None
    pocket_ids = sorted(groups.keys())
    pick = int(torch.randint(len(pocket_ids), (1,), generator=generator).item())
    pid = pocket_ids[pick]
    mol2_path = sorted(groups[pid])[0]
    return pid, mol2_path


def get_or_compute_pocket_residues(
    atom_array: AtomArray,
    sample_pdb_id: str,
    pocket_id: str,
    mol2_path: Path,
    cache_dir: Path,
    cutoff: float,
    occupancy_min: float,
    force_recompute: bool = False,
) -> List[Dict[str, Any]]:
    """
    Load cached pocket residue dicts from disk, or compute occupancy from mol2 + atom_array.
    Cache file: {cache_dir}/{sanitize(sample_pdb_id)}__{pocket_id}.json
    """
    cache_dir = Path(cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe = _sanitize_cache_key(sample_pdb_id)
    cache_path = cache_dir / f"{safe}__{pocket_id}.json"

    if cache_path.is_file() and not force_recompute:
        try:
            with open(cache_path, encoding="utf-8") as f:
                payload = json.load(f)
            residues = payload.get("residues", [])
            if isinstance(residues, list) and residues:
                matched = _count_residues_matching(atom_array, residues)
                if matched >= max(1, len(residues) // 2):
                    return residues
        except Exception:
            pass

    prot_coords, prot_keys = protein_coords_and_reskeys_from_atom_array(atom_array)
    total_ligands, hits = _aggregate_occupancy(
        mol2_path, prot_coords, prot_keys, float(cutoff)
    )
    residues = _filter_occupancy(hits, total_ligands, float(occupancy_min))

    payload = {
        "version": 1,
        "sample_pdb_id": str(sample_pdb_id),
        "pocket_id": str(pocket_id),
        "mol2": str(mol2_path),
        "cutoff": float(cutoff),
        "occupancy_min": float(occupancy_min),
        "total_ligands": int(total_ligands),
        "residues": residues,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return residues
