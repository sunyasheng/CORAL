# Modified for the CORAL inference distribution (2026); see inference/NOTICE.
# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions to load protein coordinates from CIF file and update ref_pos.

This module provides two main functions:
1. load_protein_coords_from_cif: Load protein structure from CIF file (filters out ligands)
2. update_ref_pos_from_cif: Update ref_pos coordinates using CIF file coordinates

The matching process works in three stages:
1. Chain matching: Match chains between JSON/CCD and CIF by chain ID or sequence
2. Residue matching: Match residues within each chain by res_id or sequence position
3. Atom matching: Match atoms within each residue by atom name
"""

import logging
import os
import numpy as np
from biotite.structure import AtomArray
from biotite.structure.io.pdbx import CIFFile, get_structure
from biotite.structure import get_chain_starts
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


# ============================================================================
# Constants
# ============================================================================

PROTEIN_1to3 = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS",
    "Q": "GLN", "E": "GLU", "G": "GLY", "H": "HIS", "I": "ILE",
    "L": "LEU", "K": "LYS", "M": "MET", "F": "PHE", "P": "PRO",
    "S": "SER", "T": "THR", "W": "TRP", "Y": "TYR", "V": "VAL",
    "X": "UNK",
}


# ============================================================================
# CIF Loading Functions
# ============================================================================

def load_protein_coords_from_cif(cif_path: str, filter_unresolved: bool = True) -> Optional[AtomArray]:
    """
    Load protein structure from CIF file.
    
    Filters out ligands (HETATM) and only keeps protein atoms (ATOM).
    Optionally filters out unresolved residues (residues without CA atoms).
    Adds required annotations (mol_type, label_entity_id) if missing.
    
    Args:
        cif_path: Path to CIF file
        filter_unresolved: If True, filter out residues without CA atoms (default: True).
                          This ensures CIF matches JSON which only contains resolved residues.
        
    Returns:
        AtomArray containing protein structure only (ligands removed, optionally unresolved residues removed), 
        or None if loading fails
    """
    try:
        if os.path.splitext(cif_path)[1].lower() in (".pdb", ".ent"):
            # PDB carries a single chain-ID column, so the label_asym_id/auth_asym_id
            # ambiguity described below cannot arise -- which is why generated fibril
            # receptors are fed in as PDB rather than converted to CIF. Bonds are never
            # read from this array downstream (only chain_id/res_id/atom_name/coord/
            # hetero/mol_type), so omitting include_bonds costs nothing.
            from biotite.structure.io.pdb import PDBFile

            atom_array = PDBFile.read(cif_path).get_structure(model=1)
        else:
            cif_file = CIFFile.read(cif_path)
            # use_author_fields=False → chain_id reflects label_asym_id instead of
            # auth_asym_id. The JSON generator (cif_sdf_to_json_multi_ligand.py)
            # assigns protein copy_ids in label_asym_id first-encounter order, so
            # loading with label_asym_id here keeps the JSON <-> CIF chain labels in
            # sync. Structures like 7YNN/7YNM use auth_asym_id {A..D, G..J} (skipping
            # E, F) while label_asym_id is {A..H}; using auth_asym_id here caused the
            # JSON chains E/F to fall through to sequence fallback and collide with
            # JSON chains G/H, yielding duplicated (overlapping) fibril layers.
            atom_array = get_structure(
                cif_file, model=1, include_bonds=True, use_author_fields=False
            )

        # Filter out ligands: keep only ATOM records (protein), remove HETATM (ligands)
        # In biotite, hetero flag indicates if it's a heteroatom (ligand)
        # ATOM records have hetero=False, HETATM have hetero=True
        is_protein = ~atom_array.hetero
        protein_atom_array = atom_array[is_protein]
        
        if len(protein_atom_array) == 0:
            logger.warning(f"No protein atoms found in CIF file {cif_path}")
            return None
        
        initial_count = len(protein_atom_array)
        unresolved_count = 0
        
        # Filter out unresolved residues (residues without CA atoms)
        # This matches the behavior of cif_sdf_to_json_multi_ligand.py which crops
        # the sequence to only include resolved residues
        if filter_unresolved:
            # Group atoms by (chain_id, res_id) and check if each residue has a CA atom
            resolved_mask = np.zeros(len(protein_atom_array), dtype=bool)
            for chain_id in np.unique(protein_atom_array.chain_id):
                chain_mask = protein_atom_array.chain_id == chain_id
                
                for res_id in np.unique(protein_atom_array.res_id[chain_mask]):
                    # Get mask for this specific (chain_id, res_id) combination
                    res_mask = chain_mask & (protein_atom_array.res_id == res_id)
                    res_atoms = protein_atom_array[res_mask]
                    
                    # Check if this residue has a CA atom
                    has_ca = np.any(res_atoms.atom_name == "CA")
                    
                    if has_ca:
                        # Mark all atoms in this residue as resolved
                        resolved_mask[res_mask] = True
            
            protein_atom_array = protein_atom_array[resolved_mask]
            
            if len(protein_atom_array) == 0:
                logger.warning(f"No resolved protein residues (with CA atoms) found in CIF file {cif_path}")
                return None
            
            unresolved_count = initial_count - len(protein_atom_array)
            if unresolved_count > 0:
                logger.info(f"Filtered out {unresolved_count} atoms from unresolved residues "
                           f"(residues without CA atoms)")
        
        logger.info(f"Loaded {len(protein_atom_array)} protein atoms from CIF "
                   f"(removed {len(atom_array) - initial_count} ligand atoms"
                   f"{f', {unresolved_count} unresolved residue atoms' if filter_unresolved and unresolved_count > 0 else ''})")
        
        # Add required annotations
        protein_atom_array = _add_required_annotations(protein_atom_array)
        if protein_atom_array is None:
            return None
        
        return protein_atom_array
    except Exception as e:
        logger.warning(f"Failed to load CIF file {cif_path}: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return None


def _add_required_annotations(protein_atom_array: AtomArray) -> Optional[AtomArray]:
    """
    Add required annotations (mol_type, label_entity_id) to protein atom array.
    
    Args:
        protein_atom_array: AtomArray containing only protein atoms
        
    Returns:
        AtomArray with annotations added, or None if failed
    """
    try:
        # Add mol_type annotation first (all are protein since we filtered ligands)
        if "mol_type" not in protein_atom_array.get_annotation_categories():
            mol_type = np.array(["protein"] * len(protein_atom_array), dtype="<U10")
            protein_atom_array.set_annotation("mol_type", mol_type)
            logger.info("Added mol_type annotation: all atoms are protein")
        
        # Try to use AddAtomArrayAnnot for more sophisticated annotation if available
        try:
            from protenix.data.parser import AddAtomArrayAnnot
            entity_poly_type = {}
            for chain_id in np.unique(protein_atom_array.chain_id):
                entity_poly_type[chain_id] = "polypeptide(L)"
            protein_atom_array = AddAtomArrayAnnot.add_token_mol_type(
                protein_atom_array, entity_poly_type
            )
            logger.info("Refined mol_type annotation using AddAtomArrayAnnot")
        except Exception as add_annot_error:
            logger.debug(f"AddAtomArrayAnnot.add_token_mol_type failed (non-critical): {add_annot_error}")
        
        # Add label_entity_id if not present
        if "label_entity_id" not in protein_atom_array.get_annotation_categories():
            label_entity_id = np.empty(len(protein_atom_array), dtype="<U4")
            chain_starts = get_chain_starts(protein_atom_array, add_exclusive_stop=True)
            unique_chains = np.unique(protein_atom_array.chain_id)
            chain_to_entity = {chain_id: str(i + 1) for i, chain_id in enumerate(unique_chains)}
            
            for start, stop in zip(chain_starts[:-1], chain_starts[1:]):
                chain_id = protein_atom_array.chain_id[start]
                entity_id = chain_to_entity.get(chain_id, "1")
                label_entity_id[start:stop] = entity_id
            
            protein_atom_array.set_annotation("label_entity_id", label_entity_id)
            logger.info(f"Added label_entity_id annotation: {len(unique_chains)} chains")
        
        return protein_atom_array
    except Exception as annot_error:
        logger.warning(f"Failed to add annotations to CIF atom_array: {annot_error}")
        import traceback
        logger.debug(traceback.format_exc())
        
        # Fallback: add basic annotations manually
        try:
            if "mol_type" not in protein_atom_array.get_annotation_categories():
                mol_type = np.array(["protein"] * len(protein_atom_array), dtype="<U10")
                protein_atom_array.set_annotation("mol_type", mol_type)
                logger.info("Added basic mol_type annotation (fallback)")
            if "label_entity_id" not in protein_atom_array.get_annotation_categories():
                label_entity_id = np.array(["1"] * len(protein_atom_array), dtype="<U4")
                protein_atom_array.set_annotation("label_entity_id", label_entity_id)
                logger.info("Added basic label_entity_id annotation (fallback)")
            return protein_atom_array
        except Exception as fallback_error:
            logger.error(f"Failed to add annotations even with fallback: {fallback_error}")
            return None


# ============================================================================
# Chain Matching Functions
# ============================================================================

def _extract_chain_sequence(chain_atoms: AtomArray) -> str:
    """
    Extract protein sequence string from chain atoms.
    
    Args:
        chain_atoms: AtomArray for a single chain
        
    Returns:
        Sequence string (1-letter amino acid codes)
    """
    # Extract sequence in order: sort by res_id, then get unique residues in order
    res_id_name_pairs = list(zip(chain_atoms.res_id, chain_atoms.res_name))
    res_id_name_pairs.sort(key=lambda x: x[0])
    
    # Extract unique res_names in order (preserve order of first occurrence)
    seen_res_ids = set()
    ordered_res_names = []
    for res_id, res_name in res_id_name_pairs:
        if res_id not in seen_res_ids:
            seen_res_ids.add(res_id)
            ordered_res_names.append(res_name)
    
    # Convert to sequence string
    chain_seq = "".join([PROTEIN_1to3.get(res, "X") for res in ordered_res_names])
    return chain_seq


def _build_chain_dict(protein_atoms: AtomArray, is_protein: np.ndarray) -> Dict[str, Dict]:
    """
    Build dictionary of chains from protein atoms.
    
    Args:
        protein_atoms: AtomArray containing protein atoms
        is_protein: Boolean mask for protein atoms in original atom_array
        
    Returns:
        Dictionary mapping chain_id to chain info (atoms, mask, seq, start_idx)
    """
    chains = {}
    for chain_id in np.unique(protein_atoms.chain_id):
        chain_mask = protein_atoms.chain_id == chain_id
        chain_atoms = protein_atoms[chain_mask]
        chain_seq = _extract_chain_sequence(chain_atoms)
        
        chains[chain_id] = {
            "atoms": chain_atoms,
            "mask": chain_mask,
            "seq": chain_seq,
            "start_idx": np.where(is_protein)[0][chain_mask][0] if chain_mask.any() else None
        }
        logger.debug(f"Chain {chain_id}: {len(chain_seq)} residues, seq_len={len(chain_seq)}")
    
    return chains


def _match_chains(protein_chains: Dict, cif_chains: Dict) -> List[Tuple[str, str, Dict, Dict]]:
    """
    Match chains between protein (JSON/CCD) and CIF.
    
    Matching strategy (in order):
    1. Match by chain ID (if chain IDs match directly)
    2. Match by sequence (if sequences are identical or similar)
    3. Match by order (fallback: assume chains are in same order)
    
    Args:
        protein_chains: Dictionary of protein chains from JSON/CCD
        cif_chains: Dictionary of CIF chains
        
    Returns:
        List of matched pairs: (p_chain_id, c_chain_id, p_info, c_info)
    """
    matched_pairs = []
    logger.info(f"Matching chains: {len(protein_chains)} protein chains vs {len(cif_chains)} CIF chains")

    # Two-pass matching. The previous single-pass version could double-assign a
    # CIF chain: if JSON had {A..H} but CIF used auth_asym_id {A..D,G..J} (skipping
    # E,F), the old loop would fall back to sequence-match for JSON E/F (grabbing
    # CIF G/H), then re-match JSON G/H directly to CIF G/H, so CIF G/H ended up
    # assigned to two different JSON chains and the later chains silently reused
    # the earlier chains' coordinates (causing duplicate/overlapping layers in
    # predicted amyloid fibrils).
    taken_cif = set()

    # Pass 1: exact chain-ID match for every JSON chain that has a same-named CIF
    # chain. Reserves those CIF chains before sequence fallback can steal them.
    unmatched_protein = []
    for p_chain_id, p_info in protein_chains.items():
        if p_chain_id in cif_chains and p_chain_id not in taken_cif:
            matched_pairs.append((p_chain_id, p_chain_id, p_info, cif_chains[p_chain_id]))
            taken_cif.add(p_chain_id)
            logger.info(f"Matched chain {p_chain_id} by chain ID "
                       f"(len: {len(p_info['seq'])} vs {len(cif_chains[p_chain_id]['seq'])})")
        else:
            unmatched_protein.append((p_chain_id, p_info))

    # Pass 2: sequence-based fallback for JSON chains still unmatched, skipping
    # CIF chains already reserved in Pass 1.
    for p_chain_id, p_info in unmatched_protein:
        best_match = None
        best_score = 0
        for c_chain_id, c_info in cif_chains.items():
            if c_chain_id in taken_cif:
                continue
            # Exact sequence match
            if p_info["seq"] == c_info["seq"]:
                best_match = c_chain_id
                best_score = 1.0
                break
            # Partial match score
            score = len(set(p_info["seq"]) & set(c_info["seq"])) / max(
                len(p_info["seq"]), len(c_info["seq"])
            )
            if score > best_score:
                best_score = score
                best_match = c_chain_id

        if best_match and best_score > 0.5:
            matched_pairs.append((p_chain_id, best_match, p_info, cif_chains[best_match]))
            taken_cif.add(best_match)
            logger.info(f"Matched chain {p_chain_id} with CIF chain {best_match} by sequence "
                       f"(score: {best_score:.2f}, len: {len(p_info['seq'])} vs "
                       f"{len(cif_chains[best_match]['seq'])})")
        else:
            logger.warning(f"Failed to match protein chain {p_chain_id} "
                         f"(best_score: {best_score:.2f}, seq_len: {len(p_info['seq'])})")
            if len(cif_chains) > 0:
                logger.warning(f"  Available CIF chains: {list(cif_chains.keys())}, "
                             f"seq_lens: {[len(c['seq']) for c in cif_chains.values()]}")
    
    # Strategy 3: Fallback - match by order (if no matches found)
    if len(matched_pairs) == 0:
        logger.warning("No chains were matched between protein and CIF file!")
        logger.warning(f"Protein chain IDs: {list(protein_chains.keys())}")
        logger.warning(f"CIF chain IDs: {list(cif_chains.keys())}")
        logger.warning("Trying to match by chain ID order (assuming same order)...")
        
        protein_chain_ids = sorted(protein_chains.keys())
        cif_chain_ids = sorted(cif_chains.keys())
        if len(protein_chain_ids) == len(cif_chain_ids):
            for p_chain_id, c_chain_id in zip(protein_chain_ids, cif_chain_ids):
                matched_pairs.append((
                    p_chain_id, c_chain_id,
                    protein_chains[p_chain_id], cif_chains[c_chain_id]
                ))
                logger.info(f"Matched chain {p_chain_id} with CIF chain {c_chain_id} by order (fallback)")
    
    return matched_pairs


# ============================================================================
# Residue Matching Functions
# ============================================================================

def _build_residue_dict(atoms: AtomArray) -> Dict:
    """
    Build dictionary of residues from atoms.
    
    Args:
        atoms: AtomArray for a chain
        
    Returns:
        Dictionary mapping res_id to residue info (name, indices)
    """
    residues = {}
    for i, (res_id, res_name) in enumerate(zip(atoms.res_id, atoms.res_name)):
        if res_id not in residues:
            residues[res_id] = {"name": res_name, "indices": []}
        residues[res_id]["indices"].append(i)
    return residues


def _match_residues_by_res_id(
    p_residues: Dict, c_residues: Dict, p_atoms: AtomArray, c_atoms: AtomArray,
    p_global_indices_func, ref_pos: np.ndarray, is_structured_protein: np.ndarray
) -> Tuple[int, int, List, List]:
    """
    Match residues by res_id and update coordinates.
    
    Args:
        p_residues: Protein residues dict (from JSON/CCD)
        c_residues: CIF residues dict
        p_atoms: Protein atoms
        c_atoms: CIF atoms
        p_global_indices_func: Function to get global indices for protein atoms
        ref_pos: Reference positions array to update
        is_structured_protein: Mask array to update
        
    Returns:
        Tuple: (matched_residues, chain_updated, unmatched_residues, unmatched_atoms)
    """
    matched_residues = 0
    chain_updated = 0
    unmatched_residues = []
    unmatched_atoms = []
    
    for res_id in p_residues:
        if res_id in c_residues and p_residues[res_id]["name"] == c_residues[res_id]["name"]:
            matched_residues += 1
            p_indices = p_residues[res_id]["indices"]
            c_indices = c_residues[res_id]["indices"]
            
            # Get global indices in atom_array
            p_global_indices = p_global_indices_func(p_indices)
            
            # Match atoms by name
            p_atom_names = p_atoms.atom_name[p_indices]
            c_atom_names = c_atoms.atom_name[c_indices]
            
            for p_idx, p_atom_name in zip(p_global_indices, p_atom_names):
                if p_atom_name in c_atom_names:
                    c_local_idx = c_indices[np.where(c_atom_names == p_atom_name)[0][0]]
                    ref_pos[p_idx] = c_atoms.coord[c_local_idx]
                    is_structured_protein[p_idx] = True
                    chain_updated += 1
                else:
                    res_name = p_residues[res_id]["name"]
                    unmatched_atoms.append((res_id, res_name, p_atom_name, "missing_in_cif"))
        else:
            # Track unmatched residues
            if res_id not in c_residues:
                unmatched_residues.append((res_id, p_residues[res_id]["name"], "missing_in_cif"))
            elif p_residues[res_id]["name"] != c_residues[res_id]["name"]:
                unmatched_residues.append((
                    res_id, p_residues[res_id]["name"],
                    f"name_mismatch_{c_residues[res_id]['name']}"
                ))
    
    return matched_residues, chain_updated, unmatched_residues, unmatched_atoms


def _match_residues_by_position(
    p_residues: Dict, c_residues: Dict, p_atoms: AtomArray, c_atoms: AtomArray,
    p_global_indices_func, ref_pos: np.ndarray, is_structured_protein: np.ndarray
) -> Tuple[int, int, List, List]:
    """
    Match residues by sequence alignment (fallback when res_id matching fails).
    
    This function uses sequence string matching to find the correct alignment between
    JSON/CCD residues (which may be cropped and renumbered) and CIF residues (which
    may have original res_id numbering).
    
    Args:
        p_residues: Protein residues dict (from JSON/CCD)
        c_residues: CIF residues dict
        p_atoms: Protein atoms
        c_atoms: CIF atoms
        p_global_indices_func: Function to get global indices for protein atoms
        ref_pos: Reference positions array to update
        is_structured_protein: Mask array to update
        
    Returns:
        Tuple: (matched_residues, chain_updated, unmatched_residues, unmatched_atoms)
    """
    matched_residues = 0
    chain_updated = 0
    unmatched_residues = []
    unmatched_atoms = []
    
    # Sort residues by their order in the atom array (by first atom index)
    p_residue_list = sorted(p_residues.items(), key=lambda x: p_residues[x[0]]["indices"][0])
    c_residue_list = sorted(c_residues.items(), key=lambda x: c_residues[x[0]]["indices"][0])
    
    # Build sequence strings for alignment
    p_seq = "".join([PROTEIN_1to3.get(p_residues[rid]["name"], "X") for rid, _ in p_residue_list])
    c_seq = "".join([PROTEIN_1to3.get(c_residues[rid]["name"], "X") for rid, _ in c_residue_list])
    
    # Find the best alignment position (when JSON sequence is a substring of CIF sequence)
    # This handles the case where JSON sequence is cropped from the middle of CIF sequence
    best_offset = 0
    best_match_count = 0
    
    # Try to find JSON sequence as a substring in CIF sequence
    if len(p_seq) <= len(c_seq):
        for offset in range(len(c_seq) - len(p_seq) + 1):
            match_count = sum(1 for i, p_char in enumerate(p_seq) 
                            if i + offset < len(c_seq) and p_char == c_seq[i + offset])
            if match_count > best_match_count:
                best_match_count = match_count
                best_offset = offset
    else:
        # If JSON sequence is longer, try reverse: find CIF sequence in JSON sequence
        for offset in range(len(p_seq) - len(c_seq) + 1):
            match_count = sum(1 for i, c_char in enumerate(c_seq) 
                            if i + offset < len(p_seq) and c_char == p_seq[i + offset])
            if match_count > best_match_count:
                best_match_count = match_count
                best_offset = -offset  # Negative to indicate CIF is substring of JSON
    
    # Use the best alignment offset
    if best_offset >= 0:
        # JSON sequence starts at position best_offset in CIF sequence
        logger.info(f"Sequence alignment: JSON sequence starts at position {best_offset} in CIF sequence "
                   f"(match: {best_match_count}/{len(p_seq)} residues)")
        
        # Match residues using the alignment offset
        for i, (p_res_id, p_res_info) in enumerate(p_residue_list):
            c_idx = i + best_offset
            if 0 <= c_idx < len(c_residue_list):
                c_res_id, c_res_info = c_residue_list[c_idx]
                if p_res_info["name"] == c_res_info["name"]:
                    matched_residues += 1
                    p_indices = p_res_info["indices"]
                    c_indices = c_res_info["indices"]
                    
                    # Get global indices in atom_array
                    p_global_indices = p_global_indices_func(p_indices)
                    
                    # Match atoms by name
                    p_atom_names = p_atoms.atom_name[p_indices]
                    c_atom_names = c_atoms.atom_name[c_indices]
                    
                    for p_idx, p_atom_name in zip(p_global_indices, p_atom_names):
                        if p_atom_name in c_atom_names:
                            c_local_idx = c_indices[np.where(c_atom_names == p_atom_name)[0][0]]
                            ref_pos[p_idx] = c_atoms.coord[c_local_idx]
                            is_structured_protein[p_idx] = True
                            chain_updated += 1
                        else:
                            res_name = p_res_info["name"]
                            unmatched_atoms.append((p_res_id, res_name, p_atom_name, "missing_in_cif"))
                else:
                    unmatched_residues.append((
                        p_res_id, p_res_info["name"],
                        f"name_mismatch_{c_res_info['name']} at aligned position {c_idx}"
                    ))
            else:
                unmatched_residues.append((
                    p_res_id, p_res_info["name"],
                    f"out_of_range (aligned position {c_idx} not in CIF range [0, {len(c_residue_list)})"
                ))
    else:
        # CIF sequence is substring of JSON sequence (shouldn't happen in normal cases)
        logger.warning(f"Sequence alignment: CIF sequence is shorter than JSON sequence, "
                      f"using simple position matching")
        # Fallback to simple position matching
        for (p_res_id, p_res_info), (c_res_id, c_res_info) in zip(p_residue_list, c_residue_list):
            if p_res_info["name"] == c_res_info["name"]:
                matched_residues += 1
                p_indices = p_res_info["indices"]
                c_indices = c_res_info["indices"]
                
                # Get global indices in atom_array
                p_global_indices = p_global_indices_func(p_indices)
                
                # Match atoms by name
                p_atom_names = p_atoms.atom_name[p_indices]
                c_atom_names = c_atoms.atom_name[c_indices]
                
                for p_idx, p_atom_name in zip(p_global_indices, p_atom_names):
                    if p_atom_name in c_atom_names:
                        c_local_idx = c_indices[np.where(c_atom_names == p_atom_name)[0][0]]
                        ref_pos[p_idx] = c_atoms.coord[c_local_idx]
                        is_structured_protein[p_idx] = True
                        chain_updated += 1
                    else:
                        res_name = p_res_info["name"]
                        unmatched_atoms.append((p_res_id, res_name, p_atom_name, "missing_in_cif"))
            else:
                unmatched_residues.append((
                    p_res_id, p_res_info["name"],
                    f"name_mismatch_{c_res_info['name']}"
                ))
    
    return matched_residues, chain_updated, unmatched_residues, unmatched_atoms


# ============================================================================
# Main Update Function
# ============================================================================

def load_ref_map(reference_path: str) -> Optional[Dict]:
    """Read the explicit position-to-res_id map written next to a reference structure.

    Returns None when absent, so callers that do not pin coordinates are unaffected. A caller that
    DOES pin must treat None as fatal: without the map the correspondence has to be guessed, and a
    guess that goes wrong pins residues at other residues' coordinates while marking them fixed,
    which is worse than refusing to run. scripts/make_ref_map.py records why the map cannot be
    recovered from the JSON: `count: N` builds N byte-identical copies, so per-chain numbering is
    absent by construction.
    """
    import json as _json

    path = os.path.splitext(reference_path)[0] + ".refmap.json"
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        ref_map = _json.load(fh)
    if ref_map.get("version") != 1:
        raise ValueError(f"{path}: unsupported refmap version {ref_map.get('version')}")
    logger.info("Loaded explicit reference map: %s (%d copies)",
                path, len(ref_map.get("copies", [])))
    return ref_map


def update_ref_pos_from_ref_map(
    atom_array: AtomArray,
    cif_atom_array: AtomArray,
    ref_map: Dict,
) -> tuple[AtomArray, np.ndarray]:
    """Pin protein coordinates by READING the correspondence -- no matching, no alignment.

    Every (json_chain, position) -> (receptor chain, res_id) pair comes from the map. The residue name
    at the target is still re-checked against the built input before any coordinate is written: the
    map is produced by the same tooling that wrote the JSON, so it can be confidently wrong, and an
    explicit mapping is worth having because it is checkable, not because it is trustworthy. Any
    inconsistency raises instead of degrading.

    Atoms with no entry -- a residue that chain never resolved -- keep their CCD ref_pos and stay out
    of the returned mask, so they start from noise and diffuse. That is correct: their positions are
    genuinely unknown. An atom absent only from its own residue (a terminal OXT) is skipped likewise.
    """
    is_protein = atom_array.mol_type == "protein"
    is_structured_protein = np.zeros(len(atom_array), dtype=bool)
    if not is_protein.any():
        raise ValueError("refmap given but atom_array holds no protein atoms")

    if "mol_type" in cif_atom_array.get_annotation_categories():
        cif_atoms = cif_atom_array[cif_atom_array.mol_type == "protein"]
    else:
        cif_atoms = cif_atom_array

    cif_row, cif_res_name = {}, {}
    for i in range(len(cif_atoms)):
        key = (str(cif_atoms.chain_id[i]), int(cif_atoms.res_id[i]))
        cif_row[key + (str(cif_atoms.atom_name[i]),)] = i
        cif_res_name[key] = str(cif_atoms.res_name[i])

    seq = ref_map["sequence"]
    json_chains = set(str(c) for c in np.unique(atom_array.chain_id[is_protein]))
    ref_pos = atom_array.ref_pos.copy()
    written = missing_res = missing_atom = 0

    for entry in ref_map["copies"]:
        jchain, rchain = str(entry["json_chain"]), str(entry["chain"])
        if jchain not in json_chains:
            raise ValueError(
                f"refmap names json_chain {jchain}, not a protein chain in the built "
                f"input (present: {sorted(json_chains)})")
        positions = {int(k): int(v) for k, v in entry["positions"].items()}
        sel = np.where(is_protein & (atom_array.chain_id == jchain))[0]
        if sel.size == 0:
            raise ValueError(f"refmap json_chain {jchain} selected no atoms")
        for i in sel:
            p = int(atom_array.res_id[i])
            res_id = positions.get(p)
            if res_id is None:
                missing_res += 1
                continue
            if not 1 <= p <= len(seq):
                raise ValueError(
                    f"refmap position {p} outside sequence of length {len(seq)}")
            got = cif_res_name.get((rchain, res_id))
            if got is None:
                raise ValueError(
                    f"refmap points json_chain {jchain} position {p} at {rchain}:{res_id}, "
                    f"absent from the reference structure")
            if got != str(atom_array.res_name[i]):
                raise ValueError(
                    f"refmap inconsistent: json_chain {jchain} position {p} is "
                    f"{atom_array.res_name[i]} but {rchain}:{res_id} is {got}")
            j = cif_row.get((rchain, res_id, str(atom_array.atom_name[i])))
            if j is None:
                missing_atom += 1
                continue
            ref_pos[i] = cif_atoms.coord[j]
            is_structured_protein[i] = True
            written += 1

    atom_array.ref_pos = ref_pos
    logger.info(
        "Pinned %d/%d protein atoms from the explicit reference map (%d atoms in residues "
        "that chain does not resolve, %d atoms absent from their residue)",
        written, int(is_protein.sum()), missing_res, missing_atom)
    if written == 0:
        raise ValueError("refmap matched no atoms at all")
    return atom_array, is_structured_protein


def update_ref_pos_from_cif(
    atom_array: AtomArray,
    cif_atom_array: AtomArray,
) -> tuple[AtomArray, np.ndarray]:
    """
    Update ref_pos in atom_array using coordinates from CIF file.
    
    Only updates protein atoms, ligand atoms keep their CCD ref_pos.
    
    The matching process works in three stages:
    1. Chain matching: Match chains by chain ID or sequence
    2. Residue matching: Match residues by res_id (or by position if res_id fails)
    3. Atom matching: Match atoms by atom name
    
    Args:
        atom_array: Target AtomArray (from JSON, may include ligands)
        cif_atom_array: Source AtomArray (from CIF, protein only)
        
    Returns:
        tuple: (Updated AtomArray with ref_pos from CIF for protein atoms, 
                is_structured_protein mask [N_atom] indicating which atoms have coordinates from CIF)
    """
    # Get protein mask
    is_protein = atom_array.mol_type == "protein"
    
    if not is_protein.any():
        logger.warning("No protein atoms found in atom_array, skipping ref_pos update")
        is_structured_protein = np.zeros(len(atom_array), dtype=bool)
        return atom_array, is_structured_protein
    
    # Extract protein chains from both arrays
    protein_atoms = atom_array[is_protein]
    if "mol_type" in cif_atom_array.get_annotation_categories():
        cif_protein_atoms = cif_atom_array[cif_atom_array.mol_type == "protein"]
    else:
        cif_protein_atoms = cif_atom_array
        logger.warning("CIF atom_array missing mol_type annotation, assuming all atoms are protein")
    
    if len(cif_protein_atoms) == 0:
        logger.warning("No protein atoms found in CIF file, skipping ref_pos update")
        is_structured_protein = np.zeros(len(atom_array), dtype=bool)
        return atom_array, is_structured_protein
    
    # Get ref_pos (should already exist from add_reference_features)
    ref_pos = atom_array.ref_pos.copy()
    # Snapshot before CIF overwrite (CCD reference; for optional debug lines)
    ref_pos_before_cif = np.asarray(ref_pos, dtype=np.float64).copy()
    # Experimental coords from PKL/crop — unchanged by this function; compare to CIF in debug
    coord_pkl = np.asarray(atom_array.coord, dtype=np.float64).copy()
    is_structured_protein = np.zeros(len(atom_array), dtype=bool)
    
    # Build chain dictionaries
    protein_chains = _build_chain_dict(protein_atoms, is_protein)
    cif_chains = _build_chain_dict(cif_protein_atoms, np.ones(len(cif_protein_atoms), dtype=bool))
    
    # Match chains
    matched_pairs = _match_chains(protein_chains, cif_chains)
    
    # Update ref_pos for matched chains
    updated_count = 0
    for p_chain_id, c_chain_id, p_info, c_info in matched_pairs:
        p_atoms = p_info["atoms"]
        c_atoms = c_info["atoms"]
        
        logger.info(f"Updating chain {p_chain_id}: {len(p_atoms)} protein atoms vs {len(c_atoms)} CIF atoms")
        
        # Build residue dictionaries
        p_residues = _build_residue_dict(p_atoms)
        c_residues = _build_residue_dict(c_atoms)
        
        logger.info(f"  Chain {p_chain_id}: {len(p_residues)} protein residues (from JSON/CCD) vs "
                   f"{len(c_residues)} CIF residues (from CIF file)")
        
        # Debug: Show res_id ranges
        if len(p_residues) > 0 and len(c_residues) > 0:
            p_res_ids = sorted(p_residues.keys())
            c_res_ids = sorted(c_residues.keys())
            logger.info(f"  Chain {p_chain_id}: JSON/CCD res_id range: {p_res_ids[0]} to {p_res_ids[-1]}, "
                       f"CIF res_id range: {c_res_ids[0]} to {c_res_ids[-1]}")
            logger.info(f"  Chain {p_chain_id}: First 5 JSON/CCD res_ids: {p_res_ids[:5]}, "
                       f"First 5 CIF res_ids: {c_res_ids[:5]}")
            p_res_names_first5 = [p_residues[rid]["name"] for rid in p_res_ids[:5]]
            c_res_names_first5 = [c_residues[rid]["name"] for rid in c_res_ids[:5]]
            logger.info(f"  Chain {p_chain_id}: First 5 JSON/CCD residue names: {p_res_names_first5}, "
                       f"First 5 CIF residue names: {c_res_names_first5}")
        
        # Helper function to get global indices
        def get_global_indices(p_indices):
            return np.where(is_protein)[0][p_info["mask"]][p_indices]
        
        # Try matching by res_id first
        matched_residues, chain_updated, unmatched_residues, unmatched_atoms = \
            _match_residues_by_res_id(
                p_residues, c_residues, p_atoms, c_atoms,
                get_global_indices, ref_pos, is_structured_protein
            )
        
        # If res_id matching failed or matched very few residues, try sequence-based matching
        # This handles the case where JSON sequence is cropped and renumbered (res_id from 1),
        # but CIF has original res_id numbering
        match_ratio = matched_residues / len(p_residues) if len(p_residues) > 0 else 0
        if match_ratio < 0.5:  # If less than 50% matched, try sequence alignment
            logger.warning(f"  Chain {p_chain_id}: res_id matching matched only {matched_residues}/{len(p_residues)} residues "
                          f"({match_ratio*100:.1f}%), trying sequence-based alignment...")
            matched_residues_seq, chain_updated_seq, unmatched_residues_seq, unmatched_atoms_seq = \
                _match_residues_by_position(
                    p_residues, c_residues, p_atoms, c_atoms,
                    get_global_indices, ref_pos, is_structured_protein
                )
            # Use sequence-based matching if it matched more residues
            if matched_residues_seq > matched_residues:
                matched_residues = matched_residues_seq
                chain_updated = chain_updated_seq
                unmatched_residues = unmatched_residues_seq
                unmatched_atoms = unmatched_atoms_seq
                logger.info(f"  Chain {p_chain_id}: Sequence-based alignment succeeded: "
                           f"Matched {matched_residues}/{len(p_residues)} residues, "
                           f"Updated {chain_updated} atoms")
            elif matched_residues_seq > 0:
                logger.info(f"  Chain {p_chain_id}: Sequence-based alignment matched {matched_residues_seq}/{len(p_residues)} residues "
                           f"(not better than res_id matching: {matched_residues}), keeping res_id matching results")
        
        # Log results
        logger.info(f"  Chain {p_chain_id}: Matched {matched_residues}/{len(p_residues)} residues, "
                   f"Updated {chain_updated} atoms")
        if unmatched_atoms:
            logger.warning(f"  Chain {p_chain_id}: {len(unmatched_atoms)} unmatched atoms "
                         f"(not found in CIF): {unmatched_atoms[:10]}")
        if unmatched_residues:
            logger.warning(f"  Chain {p_chain_id}: {len(unmatched_residues)} unmatched residues "
                         f"(first 10): {unmatched_residues[:10]}")
        if len(c_residues) > len(p_residues):
            logger.warning(f"  Chain {p_chain_id}: CIF has {len(c_residues) - len(p_residues)} "
                         f"more residues than JSON/CCD")
        elif len(p_residues) > len(c_residues):
            logger.warning(f"  Chain {p_chain_id}: JSON/CCD has {len(p_residues) - len(c_residues)} "
                         f"more residues than CIF")
        if len(p_atoms) != len(c_atoms):
            if matched_residues > 0:
                logger.info(f"  Chain {p_chain_id}: Atom count difference: {len(p_atoms)} (JSON/CCD) vs "
                           f"{len(c_atoms)} (CIF) = {len(p_atoms) - len(c_atoms)} extra atom(s) in JSON/CCD")
            else:
                logger.debug(f"  Chain {p_chain_id}: Atom count difference (no matches): "
                           f"{len(p_atoms)} (JSON/CCD) vs {len(c_atoms)} (CIF)")
        
        updated_count += chain_updated
    
    # Optional debug: matching sanity = PKL experimental coord vs CIF coord (staged in ref_pos).
    # CCD ref_pos vs CIF is expected large; atom_array.coord vs CIF should be ~0 if same structure.
    # export DEBUG_CIF_REF_POS_TRACE=1  (pdb: **c** continue, not **q**)
    _dbg_trace = os.environ.get("DEBUG_CIF_REF_POS_TRACE", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if _dbg_trace:
        m = is_structured_protein
        n_struct = int(m.sum())
        logger.warning(
            "[DEBUG_CIF_REF_POS_TRACE] update_ref_pos_from_cif: updated_count=%s "
            "is_structured_protein=%s",
            updated_count,
            n_struct,
        )
        if n_struct > 0:
            pkl_xyz = coord_pkl[m]
            cif_xyz = np.asarray(ref_pos[m], dtype=np.float64)
            d_match = cif_xyz - pkl_xyz
            rms_match = float(np.sqrt((d_match**2).mean()))
            rmax_match = float(np.abs(d_match).max())
            logger.warning(
                "[DEBUG_CIF_REF_POS_TRACE] Match check: |CIF_coord − atom_array.coord| on "
                "structured-protein atoms → max=%.6f Å  RMS=%.6f Å  "
                "(~0 = pairing OK & same mmCIF frame as PKL)",
                rmax_match,
                rms_match,
            )
            old_ccd = ref_pos_before_cif[m]
            d_ccd_cif = cif_xyz - old_ccd
            logger.warning(
                "[DEBUG_CIF_REF_POS_TRACE] (FYI) |CIF_coord − old ref_pos (CCD)| → "
                "max=%.6f Å  RMS=%.6f Å  (large is normal; CCD ≠ crystal)",
                float(np.abs(d_ccd_cif).max()),
                float(np.sqrt((d_ccd_cif**2).mean())),
            )
        else:
            logger.warning(
                "[DEBUG_CIF_REF_POS_TRACE] No structured-protein atoms flagged "
                "(check matching)."
            )

    # Final summary
    if updated_count > 0:
        logger.info(f"Updated {updated_count} protein atom coordinates from CIF file")
        atom_array.set_annotation("ref_pos", ref_pos)
        # Debug: Print updated ref_pos stats
        print(f"[DEBUG] update_ref_pos_from_cif: Updated {updated_count} atoms")
        print(f"  Updated ref_pos range: X=[{ref_pos[:, 0].min():.2f}, {ref_pos[:, 0].max():.2f}], "
              f"Y=[{ref_pos[:, 1].min():.2f}, {ref_pos[:, 1].max():.2f}], "
              f"Z=[{ref_pos[:, 2].min():.2f}, {ref_pos[:, 2].max():.2f}]")
        print(f"  Updated ref_pos center: ({ref_pos[:, 0].mean():.2f}, {ref_pos[:, 1].mean():.2f}, "
              f"{ref_pos[:, 2].mean():.2f})")
        print(f"  is_structured_protein: {is_structured_protein.sum()} atoms have coordinates from CIF")
    else:
        logger.warning("No atoms were updated from CIF file - coordinates may not match")
        logger.warning(f"Total protein atoms: {is_protein.sum()}, Total CIF atoms: {len(cif_protein_atoms)}")
    
    return atom_array, is_structured_protein


# ============================================================================
# Test Functions
# ============================================================================

def test_load_cif(cif_path: str):
    """
    Test loading CIF file.
    
    Args:
        cif_path: Path to CIF file
    """
    print(f"\n{'='*80}")
    print(f"Testing: load_protein_coords_from_cif")
    print(f"CIF file: {cif_path}")
    print(f"{'='*80}\n")
    
    result = load_protein_coords_from_cif(cif_path)
    
    if result is None:
        print("❌ FAILED: Could not load CIF file")
        return False
    
    print(f"✅ SUCCESS: Loaded {len(result)} protein atoms")
    print(f"   Chains: {np.unique(result.chain_id)}")
    print(f"   Residues: {len(np.unique(result.res_id))}")
    print(f"   Annotations: {result.get_annotation_categories()}")
    
    return True


def test_update_ref_pos(cif_path: str, create_dummy_atom_array: bool = True):
    """
    Test updating ref_pos from CIF file.
    
    Args:
        cif_path: Path to CIF file
        create_dummy_atom_array: If True, create a dummy atom_array from CIF for testing
    """
    print(f"\n{'='*80}")
    print(f"Testing: update_ref_pos_from_cif")
    print(f"CIF file: {cif_path}")
    print(f"{'='*80}\n")
    
    # Load CIF file
    cif_atom_array = load_protein_coords_from_cif(cif_path)
    if cif_atom_array is None:
        print("❌ FAILED: Could not load CIF file")
        return False
    
    # Create a dummy atom_array for testing (copy from CIF and add ref_pos)
    if create_dummy_atom_array:
        print("Creating dummy atom_array from CIF for testing...")
        atom_array = cif_atom_array.copy()
        
        # Add ref_pos annotation (initialize with zeros)
        ref_pos = np.zeros((len(atom_array), 3), dtype=np.float32)
        atom_array.set_annotation("ref_pos", ref_pos)
        
        # Ensure mol_type is set
        if "mol_type" not in atom_array.get_annotation_categories():
            mol_type = np.array(["protein"] * len(atom_array), dtype="<U10")
            atom_array.set_annotation("mol_type", mol_type)
        
        print(f"   Created atom_array with {len(atom_array)} atoms")
    else:
        print("❌ ERROR: create_dummy_atom_array=False not supported in test mode")
        return False
    
    # Test update_ref_pos_from_cif
    print("\nUpdating ref_pos from CIF...")
    updated_atom_array, is_structured_protein = update_ref_pos_from_cif(
        atom_array, cif_atom_array
    )
    
    # Check results
    num_updated = is_structured_protein.sum()
    total_atoms = len(atom_array)
    
    print(f"\n{'='*80}")
    print(f"Results:")
    print(f"  Total atoms: {total_atoms}")
    print(f"  Updated atoms: {num_updated} ({100*num_updated/total_atoms:.1f}%)")
    print(f"  Not updated: {total_atoms - num_updated}")
    
    if num_updated > 0:
        print(f"✅ SUCCESS: Updated {num_updated} atoms from CIF")
        
        # Check if coordinates changed
        old_ref_pos = atom_array.ref_pos
        new_ref_pos = updated_atom_array.ref_pos
        coord_diff = np.linalg.norm(new_ref_pos - old_ref_pos, axis=1)
        max_diff = coord_diff.max()
        mean_diff = coord_diff[is_structured_protein].mean() if is_structured_protein.any() else 0
        
        print(f"  Max coordinate change: {max_diff:.3f} Å")
        print(f"  Mean coordinate change (updated atoms): {mean_diff:.3f} Å")
        
        return True
    else:
        print(f"⚠️  WARNING: No atoms were updated")
        return False


def main():
    """
    Main test function. Can be run directly: python cif_ref_pos_loader.py <cif_file>
    """
    import sys
    import os
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Test CIF reference position loader functions'
    )
    parser.add_argument(
        'cif_file',
        type=str,
        nargs='?',
        help='Path to CIF file to test'
    )
    parser.add_argument(
        '--test-load',
        action='store_true',
        help='Test load_protein_coords_from_cif function'
    )
    parser.add_argument(
        '--test-update',
        action='store_true',
        help='Test update_ref_pos_from_cif function'
    )
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    LOG_FORMAT = "%(asctime)s,%(msecs)-3d %(levelname)-8s [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(
        format=LOG_FORMAT,
        level=getattr(logging, args.log_level),
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # If no CIF file provided, try to find example files
    if args.cif_file is None:
        # Try to find example CIF files in common locations
        example_paths = [
            "debug/fibris_protein_structured/8zlo_no_unobs.cif",
            "debug/fibris_protein_structured/8x7b_no_unobs.cif",
            "debug/fibris_protein_structured/8kf1_no_unobs.cif",
        ]
        
        for path in example_paths:
            if os.path.exists(path):
                args.cif_file = path
                print(f"Using example file: {path}")
                break
        
        if args.cif_file is None:
            print("❌ ERROR: No CIF file provided and no example files found")
            print("\nUsage:")
            print("  python cif_ref_pos_loader.py <cif_file> [--test-load] [--test-update]")
            print("\nExample:")
            print("  python cif_ref_pos_loader.py debug/fibris_protein_structured/8zlo_no_unobs.cif --test-load --test-update")
            sys.exit(1)
    
    # Check if file exists
    if not os.path.exists(args.cif_file):
        print(f"❌ ERROR: CIF file not found: {args.cif_file}")
        sys.exit(1)
    
    # Run tests
    success = True
    
    # Test 1: Load CIF file
    if args.test_load or (not args.test_update):
        success &= test_load_cif(args.cif_file)
    
    # Test 2: Update ref_pos
    if args.test_update or (not args.test_load):
        success &= test_update_ref_pos(args.cif_file)
    
    # Summary
    print(f"\n{'='*80}")
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed!")
    print(f"{'='*80}\n")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
