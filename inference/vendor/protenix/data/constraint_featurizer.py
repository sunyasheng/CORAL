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

import copy
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from biotite.structure import AtomArray, get_chain_starts
from scipy.spatial.distance import cdist

from protenix.data.constants import ELEMS, STD_RESIDUES
from protenix.data.tokenizer import Token, TokenArray
from protenix.data.utils import get_atom_mask_by_name
from protenix.utils.logger import get_logger

logger = get_logger(__name__)

# Debug flag to avoid printing the ribbon merge stats too many times (DDP will print per-process).
_RIBBON_DEBUG_PRINTED = False

def _normalize_pdb_key_for_pocket(pdb_id: str) -> str:
    """e.g. 6HRE_337cc107 -> 6HRE (prefix for MOL2 glob / cache keys)."""
    s = str(pdb_id).strip()
    if not s:
        return ""
    return s.split("_")[0].upper()


# How many pdb/pause stops to allow per process (each dataset sample used to stop = unusable).
_POCKET_DEBUG_STOPS_USED = 0


def _pocket_debug_take_stop_slot() -> bool:
    """Return True if we may stop once more (honors DEBUG_POCKET_DEFINITIONS_TRACE_MAX_STOPS)."""
    global _POCKET_DEBUG_STOPS_USED
    try:
        max_stops = int(os.environ.get("DEBUG_POCKET_DEFINITIONS_TRACE_MAX_STOPS", "1"))
    except ValueError:
        max_stops = 1
    if max_stops <= 0:
        return False
    if _POCKET_DEBUG_STOPS_USED >= max_stops:
        return False
    _POCKET_DEBUG_STOPS_USED += 1
    return True


def _debug_pocket_definitions_maybe_trace(user_frame=None) -> None:
    """Enter pdb when DEBUG_POCKET_DEFINITIONS_TRACE=1 (consume one stop slot).

    In pdb: type **c** (continue) to resume training. **q** raises BdbQuit and aborts the run.

    ``user_frame``: pass ``sys._getframe(0)`` from ``_generate_pocket_features`` so locals work.
    """
    if os.environ.get("DEBUG_POCKET_DEFINITIONS_TRACE", "0") != "1":
        return
    if not _pocket_debug_take_stop_slot():
        return
    force = os.environ.get("DEBUG_POCKET_DEFINITIONS_FORCE_PDB", "0") == "1"
    if not force and not sys.stdin.isatty():
        logger.warning(
            "[DEBUG_POCKET_DEFINITIONS_TRACE] skip pdb: stdin not a TTY. "
            "Use DEBUG_POCKET_DEFINITIONS_PAUSE=1 for Enter-to-continue, or FORCE_PDB=1."
        )
        return
    logger.warning(
        "[DEBUG_POCKET_DEFINITIONS_TRACE] pdb: type **c** to continue (do not use **q**)."
    )
    import pdb

    frame = user_frame if user_frame is not None else sys._getframe().f_back
    pdb.Pdb().set_trace(frame)


def _build_pocket_specifics_from_residue_list(
    residue_list: list,
    atom_array: AtomArray,
    atom_to_token_idx: np.ndarray,
    binder_token_list: np.ndarray,
    max_d: float,
    ignore_res_name: bool,
    pdb_key: str,
    chosen_pid: str,
) -> list:
    """Map pocket residue dicts to (binder_tokens, pocket_tokens, max_d) specs for PocketFeaturizer."""
    pocket_specifics: list = []
    for res in residue_list:
        if not isinstance(res, dict):
            continue
        cid = str(res.get("chain_id", ""))
        try:
            rid = int(res["res_id"])
        except (KeyError, TypeError, ValueError):
            continue
        mask = (atom_array.chain_id == cid) & (atom_array.res_id == rid)
        rn = res.get("res_name")
        if (
            not ignore_res_name
            and rn is not None
            and str(rn).strip()
        ):
            mask = mask & (atom_array.res_name == str(rn).strip())
        if not np.any(mask):
            logger.debug(
                "[pocket] residue not in structure: chain=%s res_id=%s pdb_key=%s pocket=%s",
                cid,
                rid,
                pdb_key,
                chosen_pid,
            )
            continue
        pocket_token_list = atom_to_token_idx[mask]
        pocket_specifics.append(
            (
                binder_token_list,
                pocket_token_list,
                max_d,
            )
        )
    return pocket_specifics


def _debug_pocket_occupancy_log(
    atom_array: AtomArray,
    sample_pdb_id: str,
    pdb_key: str,
    pocket_id: str,
    mol2_path: Path,
    residue_list: list,
    pocket_specifics: list,
    occ_binder_mode: str,
    binder_chain_id: Any,
    binder_token_list: np.ndarray,
    pocket_dist_mat_shape: Optional[tuple] = None,
) -> None:
    """Verbose pocket-occupancy trace when DEBUG_POCKET_OCCUPANCY=1 (set in shell)."""
    if os.environ.get("DEBUG_POCKET_OCCUPANCY", "0") != "1":
        return

    if hasattr(atom_array, "is_ligand"):
        lig_mask = np.asarray(atom_array.is_ligand, dtype=bool)
    else:
        lig_mask = np.asarray(atom_array.mol_type == "ligand", dtype=bool)

    n_ligand_mols = 0
    lig_rows: list[str] = []
    if np.any(lig_mask):
        asym_ids = np.unique(atom_array.asym_id_int[lig_mask])
        n_ligand_mols = int(asym_ids.size)
        for aid in asym_ids:
            sub = lig_mask & (atom_array.asym_id_int == aid)
            ch = str(atom_array.chain_id[sub][0])
            n_at = int(np.sum(sub))
            ent = ""
            if hasattr(atom_array, "label_entity_id"):
                ent = str(atom_array.label_entity_id[sub][0])
            resn = ""
            if hasattr(atom_array, "res_name"):
                resn = str(atom_array.res_name[sub][0])
            mol_id = ""
            if hasattr(atom_array, "molecule_id"):
                try:
                    mol_id = str(atom_array.molecule_id[sub][0])
                except Exception:
                    mol_id = ""
            extra = f" entity={ent}" if ent else ""
            rn = f" res_name={resn}" if resn else ""
            mid = f" mol_id={mol_id}" if mol_id else ""
            lig_rows.append(
                f"    asym_id_int={int(aid)} chain_id={ch} n_atoms={n_at}{rn}{extra}{mid}"
            )

    pocket_prev: list[str] = []
    for i, res in enumerate(residue_list[:40]):
        if not isinstance(res, dict):
            continue
        pocket_prev.append(
            f"({res.get('chain_id')},{res.get('res_id')},{res.get('res_name')})"
        )
    if len(residue_list) > 40:
        pocket_prev.append(f"... (+{len(residue_list) - 40} more)")

    ub = np.unique(binder_token_list) if binder_token_list.size else np.array([], dtype=np.int64)

    msg_lines = [
        "[DEBUG_POCKET_OCCUPANCY] ========== pocket occupancy ==========",
        f"  sample_pdb_id={sample_pdb_id!r} pdb_key={pdb_key!r}",
        f"  pocket_id={pocket_id!r} mol2={mol2_path}",
        f"  binder_mode={occ_binder_mode!r} sample_binder_chain_id={binder_chain_id!r}",
        f"  ligand molecules in atom_array (is_ligand): count={n_ligand_mols}",
    ]
    if lig_rows:
        msg_lines.extend(lig_rows)
    else:
        msg_lines.append("    (no ligand atoms)")
    msg_lines.extend(
        [
            f"  occupancy residue_list: n={len(residue_list)} preview={pocket_prev}",
            f"  mapped pocket_specifics (token-matrix rows / residue specs): n={len(pocket_specifics)} "
            f"(raw list {len(residue_list)} → mapped {len(pocket_specifics)}; "
            f"unmapped = not in crop or chain/res mismatch)",
            f"  binder side: n_binder_atom_tokens={binder_token_list.size} "
            f"unique_binder_token_idx={ub.size} "
            f"uniq_preview={ub[:32].tolist()}{'...' if ub.size > 32 else ''}",
        ]
    )
    if pocket_dist_mat_shape is not None:
        msg_lines.append(
            f"  pocket_dist_mat (before encode): shape={pocket_dist_mat_shape} "
            f"— nonzeros = constraint edges binder↔pocket token pairs"
        )
    msg_lines.append(
        "[DEBUG_POCKET_OCCUPANCY] (matrix values filled in PocketFeaturizer.generate_spec_constraint)"
    )
    msg_lines.append("[DEBUG_POCKET_OCCUPANCY] =====================================")
    logger.warning("\n".join(msg_lines))


class ConstraintFeatureGenerator:
    """Generates constraint features by coordinating different featurizers"""

    def __init__(
        self, constraint_config: dict[str, Any], ab_top2_clusters: set[str] = None
    ):
        """
        Args:
            constraint_config: Configuration dict for constraints
            ab_top2_clusters: Optional antibody cluster info
        """
        self.constraint = constraint_config
        self.ab_top2_clusters = ab_top2_clusters
        self._last_selected_pocket_id: Optional[str] = None

    # function set for inference
    @staticmethod
    def _canonicalize_contact_format(
        sequences: list[dict[str, Any]], pair: dict[str, Any]
    ) -> dict[str, Any]:
        pair = copy.deepcopy(pair)
        _pair = {}

        for id_num in ["1", "2"]:
            res_info = []
            for key in ["entity", "copy", "position", "atom"]:
                identifier_value = pair.get(f"{key}{id_num}", None)
                if identifier_value is None:
                    assert (
                        key == "atom"
                    ), "contact should have at least 3 identifiers('entity', 'copy', 'position')"
                if key == "atom" and (identifier_value is not None):
                    if isinstance(identifier_value, int):
                        entity_dict = list(
                            sequences[
                                int(pair.get(f"entity{id_num}", None) - 1)
                            ].values()
                        )[0]
                        assert "atom_map_to_atom_name" in entity_dict
                        identifier_value = entity_dict["atom_map_to_atom_name"][
                            identifier_value
                        ]
                res_info.append(identifier_value)
            _pair[f"id{id_num}"] = res_info

        if hash(tuple(_pair["id1"][:2])) == hash(tuple(_pair["id2"][:2])):
            raise ValueError("A contact pair can not be specified on the same chain")

        _pair["min_distance"] = float(pair.get("min_distance", 0))
        _pair["max_distance"] = float(pair["max_distance"])
        if _pair["max_distance"] < _pair["min_distance"]:
            raise ValueError("max_distance must be greater than min_distance")
        if "atom1" in pair and "atom2" in pair:
            _pair["contact_type"] = "atom_contact"
        else:
            _pair["contact_type"] = "token_contact"
        return _pair

    @staticmethod
    def _canonicalize_pocket_res_format(binder: dict, pocket_pos: dict) -> dict:
        assert len(pocket_pos) == 3
        if hash(tuple([binder["entity"], binder["copy"]])) == hash(
            tuple([pocket_pos["entity"], pocket_pos["copy"]])
        ):
            raise ValueError("Pockets can not be the same chain with the binder")
        return pocket_pos

    @staticmethod
    def _log_constraint_feature(
        atom_array: AtomArray, token_array: TokenArray, constraint_feature: dict
    ):

        atom_feature = constraint_feature["contact_atom"]
        if atom_feature.sum() > 0:
            # logging contact feature
            token_idx_1, token_idx_2 = torch.nonzero(
                torch.triu(atom_feature[..., 1]), as_tuple=True
            )

            atom1_index = token_array[token_idx_1].get_annotation("centre_atom_index")
            atom2_index = token_array[token_idx_2].get_annotation("centre_atom_index")

            res_name_1 = atom_array.res_name[atom1_index]
            res_name_2 = atom_array.res_name[atom2_index]

            atom_name_1 = atom_array.atom_name[atom1_index]
            atom_name_2 = atom_array.atom_name[atom2_index]

            chain_id_1 = atom_array.chain_id[atom1_index]
            chain_id_2 = atom_array.chain_id[atom2_index]

            max_distance = atom_feature[token_idx_1, token_idx_2, 1]
            min_distance = atom_feature[token_idx_1, token_idx_2, 0]

            contact_info = {
                "chain_id": np.stack([chain_id_1, chain_id_2]).T.tolist(),
                "res_name": np.stack([res_name_1, res_name_2]).T.tolist(),
                "atom_name": np.stack([atom_name_1, atom_name_2]).T.tolist(),
                "max_distance": max_distance.tolist(),
                "min_distance": min_distance.tolist(),
            }
            logger.info(f"loaded atom contact info:{contact_info}")

        token_feature = constraint_feature["contact"]
        if token_feature.sum() > 0:
            # logging contact feature
            token_idx_1, token_idx_2 = torch.nonzero(
                torch.triu(token_feature[..., 1]), as_tuple=True
            )

            atom1_index = token_array[token_idx_1].get_annotation("centre_atom_index")
            atom2_index = token_array[token_idx_2].get_annotation("centre_atom_index")

            res_name_1 = atom_array.res_name[atom1_index]
            res_name_2 = atom_array.res_name[atom2_index]

            atom_name_1 = atom_array.atom_name[atom1_index]
            atom_name_2 = atom_array.atom_name[atom2_index]

            chain_id_1 = atom_array.chain_id[atom1_index]
            chain_id_2 = atom_array.chain_id[atom2_index]

            max_distance = token_feature[token_idx_1, token_idx_2, 1]
            min_distance = token_feature[token_idx_1, token_idx_2, 0]

            contact_info = {
                "chain_id": np.stack([chain_id_1, chain_id_2]).T.tolist(),
                "res_name": np.stack([res_name_1, res_name_2]).T.tolist(),
                "atom_name": np.stack([atom_name_1, atom_name_2]).T.tolist(),
                "max_distance": max_distance.tolist(),
                "min_distance": min_distance.tolist(),
            }
            logger.info(f"loaded contact info:{contact_info}")

        pocket_feature = constraint_feature["pocket"]
        if pocket_feature.sum() > 0:
            # logging contact feature
            binder_idx, pocket_idx = torch.nonzero(
                pocket_feature[..., 0], as_tuple=True
            )
            atom1_index = token_array[binder_idx].get_annotation("centre_atom_index")
            atom2_index = token_array[pocket_idx].get_annotation("centre_atom_index")

            res_name_1 = atom_array.res_name[atom1_index]
            res_name_2 = atom_array.res_name[atom2_index]

            atom_name_1 = atom_array.atom_name[atom1_index]
            atom_name_2 = atom_array.atom_name[atom2_index]

            chain_id_1 = atom_array.chain_id[atom1_index]
            chain_id_2 = atom_array.chain_id[atom2_index]

            distance = pocket_feature[binder_idx, pocket_idx, 0]

            pocket_info = {
                "binder_chain_id": np.unique(chain_id_1).tolist(),
                "binder_res_name": np.unique(res_name_1).tolist(),
                "binder_atom_name": np.unique(atom_name_1).tolist(),
                "pocket_chain_id": np.unique(chain_id_2).tolist(),
                "pocket_res_name": np.unique(res_name_2).tolist(),
                "pocket_atom_name": np.unique(atom_name_2).tolist(),
                "distance": distance.unique().item(),
            }
            logger.info(f"loaded pocket info:{pocket_info}")

    @staticmethod
    def generate_from_json(
        token_array: TokenArray,
        atom_array: AtomArray,
        sequences: list[dict[str, Any]],
        constraint_param: dict,
    ) -> tuple[dict[str, Any], TokenArray, AtomArray]:
        # DEBUG: Check input constraint parameters
        logger.debug(f"constraint_param keys: {constraint_param.keys()}")
        logger.debug(f"Number of contact constraints: {len(constraint_param.get('contact', []))}")
        if len(constraint_param.get("contact", [])) > 0:
            logger.debug(f"First contact constraint: {constraint_param.get('contact', [])[0]}")
        
        feature_dict = {}

        # build atom-level contact features
        contact_inputs = [
            ConstraintFeatureGenerator._canonicalize_contact_format(sequences, pair)
            for pair in constraint_param.get("contact", [])
        ]
        # DEBUG: Check canonicalized contact inputs
        logger.debug(f"Number of contact_inputs: {len(contact_inputs)}")
        logger.debug(f"Number of token_contact_inputs: {len([d for d in contact_inputs if d['contact_type'] == 'token_contact'])}")
        if len(contact_inputs) > 0:
            logger.debug(f"First canonicalized contact: {contact_inputs[0]}")
        atom_contact_inputs = list(
            filter(lambda d: d["contact_type"] == "atom_contact", contact_inputs)
        )
        token_contact_inputs = list(
            filter(lambda d: d["contact_type"] == "token_contact", contact_inputs)
        )

        atom_to_token_idx_dict = {}
        for idx, token in enumerate(token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_to_token_idx_dict[atom_idx] = idx

        atom_contact_specifics = []
        for i, pair in enumerate(atom_contact_inputs):
            atom_mask1 = get_atom_mask_by_name(
                atom_array=atom_array,
                entity_id=pair["id1"][0],
                copy_id=pair["id1"][1],
                position=pair["id1"][2],
                atom_name=pair["id1"][3],
            )
            atom_mask2 = get_atom_mask_by_name(
                atom_array=atom_array,
                entity_id=pair["id2"][0],
                copy_id=pair["id2"][1],
                position=pair["id2"][2],
                atom_name=pair["id2"][3],
            )
            atom_list_1 = np.nonzero(atom_mask1)[0]
            atom_list_2 = np.nonzero(atom_mask2)[0]
            if np.size(atom_list_1) != 1 or np.size(atom_list_2) != 1:
                raise ValueError(f"Atom contact {i} not found for the input")
            atom_contact_specifics.append(
                (
                    atom_list_1.item(),
                    atom_list_2.item(),
                    pair["max_distance"],
                    pair["min_distance"],
                )
            )

        token_array, atom_array, atom_contact_specifics, _, _ = (
            ConstraintFeatureGenerator.expand_tokens(
                token_array,
                atom_array,
                atom_contact_specifics,
                atom_to_token_idx_dict,
                None,
            )
        )
        atom_contact_featurizer = ContactAtomFeaturizer(
            token_array=token_array, atom_array=atom_array
        )
        feature_dict["contact_atom"], _ = (
            atom_contact_featurizer.generate_spec_constraint(
                atom_contact_specifics,
                feature_type="continuous",
                shape=(len(token_array), len(token_array), 2),
            )
        )

        # build token-level contact
        atom_to_token_idx_dict = {}
        for idx, token in enumerate(token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_to_token_idx_dict[atom_idx] = idx
        atom_to_token_idx = np.array(
            [atom_to_token_idx_dict[atom_idx] for atom_idx in range(len(atom_array))]
        )
        token_contact_specifics = []
        for i, pair in enumerate(token_contact_inputs):
            atom_mask1 = get_atom_mask_by_name(
                atom_array=atom_array,
                entity_id=pair["id1"][0],
                copy_id=pair["id1"][1],
                position=pair["id1"][2],
                atom_name=pair["id1"][3],
            )
            atom_mask2 = get_atom_mask_by_name(
                atom_array=atom_array,
                entity_id=pair["id2"][0],
                copy_id=pair["id2"][1],
                position=pair["id2"][2],
                atom_name=pair["id2"][3],
            )
            token_list_1 = atom_to_token_idx[atom_mask1]
            token_list_2 = atom_to_token_idx[atom_mask2]
            if np.size(token_list_1) == 0 or np.size(token_list_2) == 0:
                raise ValueError(f"Contact {i} not found for the input")
            token_contact_specifics.append(
                (token_list_1, token_list_2, pair["max_distance"], pair.get("min_distance", 0))
            )
        
        # DEBUG: Check final token_contact_specifics
        logger.debug(f"Number of token_contact_specifics: {len(token_contact_specifics)}")
        if len(token_contact_specifics) > 0:
            logger.debug(f"First token_contact_specific: {token_contact_specifics[0]}")

        contact_featurizer = ContactFeaturizer(
            token_array=token_array, atom_array=atom_array
        )
        contact_feature, constrained_tokens = contact_featurizer.generate_spec_constraint(
            token_contact_specifics, feature_type="continuous"
        )
        feature_dict["contact"] = contact_feature
        
        # DEBUG: Check generated contact feature
        logger.debug(f"contact_feature shape: {contact_feature.shape}")
        logger.debug(f"contact_feature non-zero count: {(contact_feature != 0).sum().item()}")
        logger.debug(f"constrained_tokens: {constrained_tokens}")

        # build pocket features (support single "pocket" or multiple "pockets")
        pocket_specifics = []
        pocket_list = constraint_param.get("pockets")
        if pocket_list is None and constraint_param.get("pocket"):
            pocket_list = [constraint_param["pocket"]]
        if pocket_list:
            for pocket_idx, pocket in enumerate(pocket_list):
                pocket = pocket.get("pocket", pocket)  # unwrap nested {"pocket": {...}} format
                distance = pocket["max_distance"]
                binder = pocket["binder_chain"]
                assert len(binder) == 2

                atom_mask_binder = get_atom_mask_by_name(
                    atom_array=atom_array,
                    entity_id=binder["entity"],
                    copy_id=binder["copy"],
                )

                binder_asym_id = torch.tensor(
                    atom_array.asym_id_int[atom_mask_binder], dtype=torch.long
                )
                num_binder = binder_asym_id.unique().numel()
                if num_binder == 0:
                    raise ValueError(
                        f"Binder does not exist. pocket_idx={pocket_idx}, num_binder={num_binder}"
                    )
                elif num_binder > 1:
                    raise ValueError(
                        f"#Binders is more than 1. pocket_idx={pocket_idx}, num_binder={num_binder}"
                    )
                else:
                    binder_token_list = atom_to_token_idx[atom_mask_binder]

                    for j, pocket_res in enumerate(pocket["contact_residues"]):
                        pocket_res = (
                            ConstraintFeatureGenerator._canonicalize_pocket_res_format(
                                binder, pocket_res
                            )
                        )

                        atom_mask_pocket = get_atom_mask_by_name(
                            atom_array=atom_array,
                            entity_id=pocket_res["entity"],
                            copy_id=pocket_res["copy"],
                            position=pocket_res["position"],
                        )
                        pocket_token_list = atom_to_token_idx[atom_mask_pocket]

                        if np.size(pocket_token_list) == 0:
                            raise ValueError(
                                f"Pocket not found: pocket_idx={pocket_idx}, res_idx={j}"
                            )

                        pocket_specifics.append(
                            (binder_token_list, pocket_token_list, distance)
                        )
            logger.info(f"#pocket: {len(pocket_specifics)} pairs from {len(pocket_list)} pocket definition(s)")

        pocket_featurizer = PocketFeaturizer(
            token_array=token_array, atom_array=atom_array
        )
        feature_dict["pocket"], _ = pocket_featurizer.generate_spec_constraint(
            pocket_specifics,
            feature_type="continuous",
        )

        # build substructure features
        substructure_specifics = {
            "token_indices": [],
            "token_coords": [],
        }
        if substructure := constraint_param.get("structure", {}):
            # Upstream leaves this as a `pass`, so anything under the JSON "structure"
            # key is silently ignored. Wire it up: the caller names residues/atoms the
            # same way pocket/contact do, and supplies their coordinates; we resolve
            # each to a token index and hand (token_indices, token_coords) to
            # SubStructureFeaturizer, which bins the pairwise distances.
            #
            #   "structure": {"tokens": [{"entity": 1, "copy": 1, "position": 12,
            #                             "coord": [x, y, z]}, ...]}
            #
            # Optional "atom" selects a specific atom of that residue. Entries that
            # do not resolve are logged and skipped (never silently dropped).
            n_bad = 0
            for k, tok in enumerate(substructure.get("tokens", [])):
                if "coord" not in tok:
                    logger.warning(f"[substructure] entry {k} has no 'coord'; skipped")
                    n_bad += 1
                    continue
                atom_mask = get_atom_mask_by_name(
                    atom_array=atom_array,
                    entity_id=tok["entity"],
                    copy_id=tok["copy"],
                    position=tok["position"],
                    atom_name=tok.get("atom"),
                )
                tok_list = atom_to_token_idx[atom_mask]
                if np.size(tok_list) == 0:
                    logger.warning(
                        f"[substructure] entry {k} not found: entity={tok['entity']} "
                        f"copy={tok['copy']} position={tok['position']} "
                        f"atom={tok.get('atom')}"
                    )
                    n_bad += 1
                    continue
                substructure_specifics["token_indices"].append(int(tok_list[0]))
                substructure_specifics["token_coords"].append(
                    [float(x) for x in tok["coord"]]
                )
            logger.info(
                f"#substructure tokens:{len(substructure_specifics['token_indices'])}"
                f" (skipped {n_bad})"
            )
        substructure_featurizer = SubStructureFeaturizer(
            token_array=token_array, atom_array=atom_array
        )
        feature_dict["substructure"] = substructure_featurizer.generate_spec_constraint(
            substructure_specifics, feature_type="one_hot"
        )

        logger.info(
            f"Loaded constraint feature: #atom contact:{len(atom_contact_specifics)} #contact:{len(token_contact_specifics)} #pocket:{len(pocket_specifics)}"
        )
        ConstraintFeatureGenerator._log_constraint_feature(
            atom_array, token_array, feature_dict
        )

        return feature_dict, token_array, atom_array

    @staticmethod
    def generate_ribbon_constraints_dynamic(
        atom_array: AtomArray,
        token_array: TokenArray,
        min_distance: float = 4.0,
        max_distance: float = 6.0,
        tolerance: float = 1.0,
        min_common_residues: int = 5,
        distance_ratio_threshold: float = 1.5,
    ) -> dict[str, Any]:
        """
        动态生成 ribbon constraints（训练时调用，基于 cropped 数据）。
        
        Ribbon constraints 是用于 fibril 结构的约束：
        - 只应用于蛋白质链（排除配体、RNA、DNA）
        - 找到同一 protofilament 的相邻链（通过 CA-CA 距离中位数判断）
        - 对于相邻链，在相同 res_id 位置生成距离约束
        
        Args:
            atom_array: Cropped atom array
            token_array: Cropped token array
            min_distance: 最小 CA-CA 距离（Å），用于判断相邻链（默认 4.0，相邻链通常 4-6 Å）
            max_distance: 最大 CA-CA 距离（Å），用于判断相邻链（默认 6.0，相邻链通常 4-6 Å）
            tolerance: 距离容差（Å），用于生成约束范围（默认 1.0）
            min_common_residues: 最小共有残基数，用于计算中位数距离（默认 5）
            distance_ratio_threshold: 距离比例阈值，用于判断第二近邻是否也是邻居（默认 1.5）
        
        Returns:
            Constraint feature dictionary (只包含 "contact" feature)
        """
        feature_dict = {}
        
        # 1. 只提取蛋白质链的 CA 原子（排除配体、RNA、DNA）
        protein_mask = atom_array.mol_type == "protein"
        ca_mask = (atom_array.atom_name == "CA") & protein_mask
        ca_atoms = atom_array[ca_mask]
        
        if len(ca_atoms) == 0:
            logger.debug("[Ribbon] No protein CA atoms found, skipping ribbon constraint generation")
            # Return empty contact feature
            # ContactFeaturizer is defined later in this file (line ~1854)
            # Use forward reference or lazy import to avoid issues
            contact_featurizer = ContactFeaturizer(
                token_array=token_array, atom_array=atom_array
            )
            empty_contact_feature, _ = contact_featurizer.generate_spec_constraint(
                [], feature_type="continuous"
            )
            feature_dict["contact"] = empty_contact_feature
            return feature_dict
        
        # 2. 按 chain_id 组织残基（res_id -> CA 原子）
        residues_by_chain = {}
        chain_ids = np.unique(ca_atoms.chain_id)
        
        for chain_id in chain_ids:
            chain_mask = ca_atoms.chain_id == chain_id
            chain_ca = ca_atoms[chain_mask]
            
            # 按 res_id 组织
            residues_by_chain[chain_id] = {}
            # Get indices of CA atoms in the full atom_array
            ca_indices_in_full = np.where(ca_mask)[0]
            chain_ca_indices_in_full = ca_indices_in_full[np.where(chain_mask)[0]]
            
            for res_id in np.unique(chain_ca.res_id):
                res_mask = chain_ca.res_id == res_id
                res_ca = chain_ca[res_mask]
                if len(res_ca) > 0:
                    # 取第一个 CA 原子（通常每个残基只有一个 CA）
                    # Find the index of this CA atom in the full atom_array
                    res_ca_indices_in_chain = np.where(res_mask)[0]
                    if len(res_ca_indices_in_chain) > 0:
                        ca_index_in_full = chain_ca_indices_in_full[res_ca_indices_in_chain[0]]
                        residues_by_chain[chain_id][res_id] = {
                            'ca_coord': res_ca.coord[0],
                            'ca_index': int(ca_index_in_full),
                        }
        
        if len(residues_by_chain) < 2:
            logger.debug(f"[Ribbon] Only {len(residues_by_chain)} protein chain(s) found, need at least 2 for ribbon constraints")
            # ContactFeaturizer is defined later in this file (line ~1854)
            contact_featurizer = ContactFeaturizer(
                token_array=token_array, atom_array=atom_array
            )
            empty_contact_feature, _ = contact_featurizer.generate_spec_constraint(
                [], feature_type="continuous"
            )
            feature_dict["contact"] = empty_contact_feature
            return feature_dict
        
        # 3. 计算链对之间的 CA-CA 距离中位数，识别相邻链
        chain_ids_list = sorted(residues_by_chain.keys())
        # logger.info(f"[Ribbon Constraint] Found {len(chain_ids_list)} protein chains: {chain_ids_list}")
        
        neighbors: dict[str, list[tuple[str, float]]] = {cid: [] for cid in chain_ids_list}
        pair_medians: dict[tuple[str, str], float] = {}
        
        for i, chain_i in enumerate(chain_ids_list):
            pos_i = residues_by_chain[chain_i]
            for chain_j in chain_ids_list[i + 1:]:
                pos_j = residues_by_chain[chain_j]
                common_pos = set(pos_i.keys()) & set(pos_j.keys())
                
                if len(common_pos) < min_common_residues:
                    continue
                
                # 计算共有残基位置上的 CA-CA 距离
                dists = []
                for res_id in common_pos:
                    ca_i = pos_i[res_id]['ca_coord']
                    ca_j = pos_j[res_id]['ca_coord']
                    dist = float(np.linalg.norm(ca_i - ca_j))
                    dists.append(dist)
                
                if len(dists) == 0:
                    continue
                
                median_dist = float(np.median(dists))
                logger.debug(f"[Ribbon Constraint] Chain {chain_i} <-> {chain_j}: "
                           f"median_CA_dist={median_dist:.2f} Å, "
                           f"common_residues={len(common_pos)}, "
                           f"distance_range=[{min(dists):.2f}, {max(dists):.2f}] Å")
                
                # 只保留在阈值范围内的（同一个 protofilament 且相互靠近）
                if min_distance <= median_dist <= max_distance:
                    pair_medians[(chain_i, chain_j)] = median_dist
                    # logger.info(f"[Ribbon Constraint] ✓ Neighbor pair: {chain_i} <-> {chain_j} "
                    #               f"(median_CA_dist={median_dist:.2f} Å, within [{min_distance}, {max_distance}] Å)")
                else:
                    logger.debug(f"[Ribbon Constraint] ✗ Not neighbors: {chain_i} <-> {chain_j} "
                               f"(median_CA_dist={median_dist:.2f} Å, outside [{min_distance}, {max_distance}] Å)")
        
        # 建邻接表
        # logger.info(f"[Ribbon Constraint] Found {len(pair_medians)} neighbor pairs (before filtering)")
        for (chain_i, chain_j), median_dist in pair_medians.items():
            neighbors[chain_i].append((chain_j, median_dist))
            neighbors[chain_j].append((chain_i, median_dist))
        
        # 对每条链，只保留 1-2 个最近邻（左右邻居）
        for cid in chain_ids_list:
            nbrs = sorted(neighbors[cid], key=lambda x: x[1])
            if not nbrs:
                neighbors[cid] = []
                # logger.info(f"[Ribbon Constraint] Chain {cid}: no neighbors (isolated or different protofilament)")
                continue
            
            # 至少保留最近的一个邻居
            final_nbrs: list[tuple[str, float]] = [nbrs[0]]
            
            # 再看第二近邻是否"够近"
            if len(nbrs) > 1:
                d1 = nbrs[0][1]
                d2 = nbrs[1][1]
                if d2 <= d1 * distance_ratio_threshold:
                    # 距离量级接近，认为是另一侧的邻居
                    final_nbrs.append(nbrs[1])
                    logger.debug(f"[Ribbon Constraint] Chain {cid}: kept 2 neighbors "
                               f"(d1={d1:.2f} Å, d2={d2:.2f} Å, ratio={d2/d1:.2f} <= {distance_ratio_threshold})")
                else:
                    logger.debug(f"[Ribbon Constraint] Chain {cid}: kept only 1 neighbor "
                               f"(d1={d1:.2f} Å, d2={d2:.2f} Å, ratio={d2/d1:.2f} > {distance_ratio_threshold})")
            
            neighbors[cid] = final_nbrs
            # 打印最终的邻居关系
            # msg_parts = [f"{nbr} (median_CA_dist={dist:.2f} Å)" for nbr, dist in final_nbrs]
            # logger.info(f"[Ribbon Constraint] Chain {cid}: neighbors -> " + ", ".join(msg_parts))
        
        # 4. 生成邻接链对集合（无向）
        neighbor_pairs = {
            frozenset((cid, nbr))
            for cid, nbrs in neighbors.items()
            for (nbr, _) in nbrs
        }
        # logger.info(f"[Ribbon Constraint] Final neighbor pairs (undirected): {len(neighbor_pairs)} pairs")
        if neighbor_pairs:
            pair_list = sorted([sorted(list(pair)) for pair in neighbor_pairs])
            # logger.info(f"[Ribbon Constraint] Neighbor pairs: {pair_list}")
        
        if len(neighbor_pairs) == 0:
            logger.debug("[Ribbon] No adjacent protein chain pairs found, skipping ribbon constraint generation")
            # ContactFeaturizer is defined later in this file (line ~1854)
            contact_featurizer = ContactFeaturizer(
                token_array=token_array, atom_array=atom_array
            )
            empty_contact_feature, _ = contact_featurizer.generate_spec_constraint(
                [], feature_type="continuous"
            )
            feature_dict["contact"] = empty_contact_feature
            return feature_dict
        
        # 5. 找到所有相邻链对的共有残基位置
        common_positions = set()
        for chain_pair in neighbor_pairs:
            chain_list = list(chain_pair)
            if len(chain_list) != 2:
                continue
            chain1, chain2 = chain_list[0], chain_list[1]
            common_pos = set(residues_by_chain[chain1].keys()) & set(residues_by_chain[chain2].keys())
            common_positions.update(common_pos)
        
        if len(common_positions) == 0:
            logger.debug("[Ribbon] No common residue positions found between adjacent chains")
            # ContactFeaturizer is defined later in this file (line ~1854)
            contact_featurizer = ContactFeaturizer(
                token_array=token_array, atom_array=atom_array
            )
            empty_contact_feature, _ = contact_featurizer.generate_spec_constraint(
                [], feature_type="continuous"
            )
            feature_dict["contact"] = empty_contact_feature
            return feature_dict
        
        # 6. 构建 atom_to_token_idx 映射
        atom_to_token_idx_dict = {}
        for idx, token in enumerate(token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_to_token_idx_dict[atom_idx] = idx
        atom_to_token_idx = np.array(
            [atom_to_token_idx_dict.get(atom_idx, -1) for atom_idx in range(len(atom_array))]
        )
        
        # 7. 为相邻链对的相同 res_id 位置生成 token-level contact constraints
        token_contact_specifics = []
        for position in sorted(common_positions):
            # 找到所有在这个位置有残基的链
            chains_with_position = [
                chain_id for chain_id in chain_ids_list
                if position in residues_by_chain[chain_id]
            ]
            
            if len(chains_with_position) < 2:
                continue
            
            # 为相邻链对生成约束
            for i, chain1 in enumerate(chains_with_position):
                for chain2 in chains_with_position[i + 1:]:
                    # 只对被判定为邻居的链对生成约束
                    if frozenset((chain1, chain2)) not in neighbor_pairs:
                        continue
                    
                    # 获取 CA 原子索引
                    ca_idx1 = residues_by_chain[chain1][position]['ca_index']
                    ca_idx2 = residues_by_chain[chain2][position]['ca_index']
                    
                    # 计算实际距离
                    ca_coord1 = residues_by_chain[chain1][position]['ca_coord']
                    ca_coord2 = residues_by_chain[chain2][position]['ca_coord']
                    distance = float(np.linalg.norm(ca_coord1 - ca_coord2))
                    
                    # 过滤距离范围
                    if distance < min_distance or distance > max_distance:
                        continue
                    
                    # 转换为 token indices
                    # 需要找到包含这些 CA 原子的 token
                    # 由于 CA 是蛋白质残基的中心原子，应该能找到对应的 token
                    token_idx1 = atom_to_token_idx[ca_idx1] if ca_idx1 < len(atom_to_token_idx) else -1
                    token_idx2 = atom_to_token_idx[ca_idx2] if ca_idx2 < len(atom_to_token_idx) else -1
                    
                    # 如果找不到对应的 token，跳过
                    if token_idx1 < 0 or token_idx2 < 0:
                        continue
                    
                    # token_idx 应该是标量（单个 token index）
                    # 但为了兼容 ContactFeaturizer 的接口，需要转换为数组
                    token_list_1 = np.array([token_idx1])
                    token_list_2 = np.array([token_idx2])
                    
                    # 生成约束（使用实际距离 ± tolerance）
                    max_dist = distance + tolerance
                    min_dist = max(0.0, distance - tolerance)
                    
                    token_contact_specifics.append(
                        (token_list_1, token_list_2, max_dist, min_dist)
                    )
        
        # 8. 生成 contact feature
        # ContactFeaturizer is defined later in this file (line ~1854)
        contact_featurizer = ContactFeaturizer(
            token_array=token_array, atom_array=atom_array
        )
        # logger.info(f"[Ribbon Constraint] Generating constraints for {len(token_contact_specifics)} token pairs")
        contact_feature, constrained_tokens = contact_featurizer.generate_spec_constraint(
            token_contact_specifics, feature_type="continuous"
        )
        feature_dict["contact"] = contact_feature
        
        # logger.info(
        #     f"[Ribbon Constraint] ✓ Successfully generated ribbon constraints: "
        #     f"{len(neighbor_pairs)} neighbor pairs, {len(token_contact_specifics)} constraint pairs"
        # )
        
        return feature_dict

    # function set for training
    def generate(
        self,
        atom_array: AtomArray,
        token_array: TokenArray,
        sample_indice: pd.core.series.Series,
        pdb_indice: pd.core.series.Series,
        msa_features: dict[str, np.ndarray],
        max_entity_mol_id: int,
        full_atom_array: AtomArray,
    ) -> tuple[
        TokenArray,
        AtomArray,
        dict[str, np.ndarray],
        dict[str, torch.Tensor],
        dict[str, Any],
        torch.Tensor,
        AtomArray,
    ]:
        """Generate all constraint features

        Args:
            idx: Data index
            atom_array: Atom array data
            token_array: Token array data
            sample_indice: Sample index information
            pdb_indice: PDB index information
            msa_features: MSA features
            max_entity_mol_id: Maximum entity mol id
            full_atom_array: Full atom array data
        Returns:
            Dictionary of constraint features
        """
        # Normal mode (including "ribbon-only" via config): always run the standard pipeline.
        # To use only ribbon constraints, set other constraint probs to 0.0 in config/CLI.
        # Setup constraint generator
        constraint_generator = self._setup_constraint_generator(sample_indice)

        # Get base features for constraint featurizer
        features_dict = self._get_base_features(token_array, atom_array)

        # Generate contact atom features
        (_, contact_atom_featurizer, contact_pairs, tokens_w_atom_contact) = (
            self._generate_contact_atom_features(
                atom_array,
                token_array,
                constraint_generator,
                features_dict,
                pdb_indice,
            )
        )

        # Expand token according to atom-contact pairs
        if len(contact_pairs) > 0:
            logger.info("Expanding tokens for contact atom constraint feature")
            token_array, atom_array, contact_pairs, token_map, full_atom_array = (
                ConstraintFeatureGenerator.expand_tokens(
                    token_array,
                    atom_array,
                    contact_pairs,
                    features_dict["atom_to_token_dict"],
                    full_atom_array,
                )
            )
            features_dict = self._get_base_features(token_array, atom_array)
        else:
            token_map = {}
        # make atom contact feature
        contact_atom_constraint_feature, tokens_w_atom_contact = (
            contact_atom_featurizer.generate_spec_constraint(
                contact_pairs,
                feature_type=self.constraint.get("contact_atom", {}).get(
                    "feature_type", "continuous"
                ),
                shape=(len(token_array), len(token_array), 2),
            )
        )

        # Expand MSA features
        if len(msa_features) > 0 and len(contact_pairs) > 0:
            msa_features = self.expand_msa_features(msa_features, token_map)

        # Generate pocket features
        pocket_constraint_feature, _, tokens_w_pocket = self._generate_pocket_features(
            atom_array,
            token_array,
            constraint_generator,
            sample_indice,
        )

        # Generate contact features (internally merges ribbon into contact if enabled)
        contact_constraint_feature, _, _, tokens_w_contact = (
            self._generate_contact_features(
                atom_array,
                token_array,
                constraint_generator,
                features_dict,
                pdb_indice,
            )
        )

        # Generate substructure features
        (
            substructure_constraint_feature,
            substructure_featurizer,
            tokens_w_substructure,
        ) = self._generate_substructure_features(
            atom_array,
            token_array,
            constraint_generator,
            sample_indice,
        )

        # Combine features
        constraint_feature = {
            "contact": contact_constraint_feature,
            "pocket": pocket_constraint_feature,
            "contact_atom": contact_atom_constraint_feature,
            "substructure": substructure_constraint_feature,
        }

        # change entity_mol_id in case of permutation of constraint pairs
        featured_tokens = (
            tokens_w_contact
            | tokens_w_atom_contact
            | tokens_w_pocket
            | tokens_w_substructure
        )

        if max_entity_mol_id is not None:
            atom_array, full_atom_array = self.change_entity_mol_id(
                token_array,
                atom_array,
                max_entity_mol_id,
                full_atom_array,
                featured_tokens,
            )

        # Log feature statistics
        feature_info = self._get_feature_statistics(
            constraint_feature,
            atom_array,
            token_array,
            features_dict,
            contact_atom_featurizer,
            substructure_featurizer,
        )
        log_constraint = self._log_feature_statistics(feature_info)
        if self._last_selected_pocket_id:
            log_constraint["pocket_selected_id"] = self._last_selected_pocket_id
        return (
            token_array,
            atom_array,
            msa_features,
            constraint_feature,
            feature_info,
            log_constraint,
            full_atom_array,
        )

    @staticmethod
    def expand_tokens(
        token_array: TokenArray,
        atom_array: AtomArray,
        contact_pairs: list[tuple[int, int, float, float]],
        atom_to_token: dict[int, int],
        full_atom_array: AtomArray,
    ) -> tuple[TokenArray, AtomArray, list[tuple[int, int, float, float]]]:
        """
        Expand selected tokens into atom-level tokens and update related arrays.

        Args:
            token_array: Original token array
            atom_array: Original atom array
            contact_pairs: Original contact pairs
            atom_to_token: Atom to token mapping
            full_atom_array: Full atom array
        Returns:
            Updated token array, atom array and transformed constraint pairs
        """
        # Update token array
        tokens_to_expand = set()
        for atom_i, atom_j, _, _ in contact_pairs:
            tokens_to_expand.add(atom_to_token[atom_i])
            tokens_to_expand.add(atom_to_token[atom_j])

        if len(tokens_to_expand) == 0:
            return token_array, atom_array, contact_pairs, {}, full_atom_array

        new_tokens = []
        # Maps old token idx to list of new token indices
        token_map = {}
        curr_token_idx = 0

        for old_token_idx, token in enumerate(token_array):
            if old_token_idx in tokens_to_expand:
                # Check if token represents standard residue
                centre_atom = atom_array[token.centre_atom_index]
                if (
                    centre_atom.res_name in STD_RESIDUES
                    and centre_atom.mol_type != "ligand"
                ):
                    # Expand token into atom-level tokens
                    atom_tokens = []
                    for atom_idx, atom_name in zip(
                        token.atom_indices, token.atom_names
                    ):
                        atom = atom_array[atom_idx]
                        atom_token = Token(ELEMS[atom.element])
                        atom_token.atom_indices = [atom_idx]
                        atom_token.atom_names = [atom_name]
                        atom_token.centre_atom_index = atom_idx
                        atom_tokens.append(atom_token)
                    new_tokens.extend(atom_tokens)
                    token_map[old_token_idx] = list(
                        range(curr_token_idx, curr_token_idx + len(atom_tokens))
                    )
                    curr_token_idx += len(atom_tokens)
                else:
                    new_tokens.append(token)
                    token_map[old_token_idx] = [curr_token_idx]
                    curr_token_idx += 1
            else:
                new_tokens.append(token)
                token_map[old_token_idx] = [curr_token_idx]
                curr_token_idx += 1

        updated_token_array = TokenArray(new_tokens)

        # Create atom_idx to new_token_idx mapping for expanded tokens
        atom_to_new_token = {}
        for new_token_idx, token in enumerate(updated_token_array):
            for atom_idx in token.atom_indices:
                atom_to_new_token[atom_idx] = new_token_idx

        # Update atom array annotations
        atom_array.centre_atom_mask = np.zeros(len(atom_array), dtype=bool)
        for token in updated_token_array:
            atom_array.centre_atom_mask[token.centre_atom_index] = True

        # Update tokatom_idx and distogram_rep_atom_mask
        expanded_atoms = set()

        for tokens in token_map.values():
            if len(tokens) > 1:  # Expanded tokens
                for token_idx in tokens:
                    token = updated_token_array[token_idx]
                    expanded_atoms.update(token.atom_indices)

        atom_array.tokatom_idx = np.array(
            [
                0 if i in expanded_atoms else idx
                for i, idx in enumerate(atom_array.tokatom_idx)
            ]
        )

        atom_array.distogram_rep_atom_mask = np.array(
            [
                1 if i in expanded_atoms else mask
                for i, mask in enumerate(atom_array.distogram_rep_atom_mask)
            ]
        )
        if len(expanded_atoms) > 0:
            logger.info(f"Expanded atoms: {expanded_atoms}")

        # Create mapping between atom_array and full_atom_array atoms
        if full_atom_array is not None:
            expanded_atom_keys = set()
            for atom_idx in expanded_atoms:
                atom = atom_array[atom_idx]
                # Create unique key using chain_id, res_id and atom_name
                atom_key = (atom.chain_id, atom.res_id, atom.atom_name)
                expanded_atom_keys.add(atom_key)

            # Update full_atom_array centre_atom_mask using the mapping
            full_atom_array.centre_atom_mask = np.array(
                [
                    (
                        1
                        if (atom.chain_id, atom.res_id, atom.atom_name)
                        in expanded_atom_keys
                        else mask
                    )
                    for atom, mask in zip(
                        full_atom_array, full_atom_array.centre_atom_mask
                    )
                ]
            )
        # Transform constraint pairs using atom_to_new_token mapping
        transformed_pairs = []
        for atom_i, atom_j, min_dist, max_dist in contact_pairs:
            new_token_i = atom_to_new_token[atom_i]
            new_token_j = atom_to_new_token[atom_j]
            transformed_pairs.append((new_token_i, new_token_j, min_dist, max_dist))

        return (
            updated_token_array,
            atom_array,
            transformed_pairs,
            token_map,
            full_atom_array,
        )

    def expand_msa_features(
        self,
        msa_features: dict[str, np.ndarray],
        token_map: dict[
            int, list[int]
        ],  # Maps old token idx to list of new token indices
    ) -> dict[str, np.ndarray]:
        """
        Expand MSA features for expanded tokens.

        Args:
            msa_features: Original MSA features
            token_map: Mapping from old token indices to new token indices

        Returns:
            Updated MSA features with expanded tokens
        """
        # Calculate new number of tokens
        num_new_tokens = max(max(new_idxs) for new_idxs in token_map.values()) + 1

        old_indices = []
        new_indices = []
        for old_idx, new_idxs in token_map.items():
            old_indices.extend([old_idx] * len(new_idxs))
            new_indices.extend(new_idxs)
        old_indices = np.array(old_indices, dtype=int)
        new_indices = np.array(new_indices, dtype=int)

        # For sequence-based features (msa, has_deletion, deletion_value)
        for feat_name in ["msa", "has_deletion", "deletion_value"]:
            if feat_name not in msa_features:
                continue

            feat = msa_features[feat_name]
            num_seqs = feat.shape[0]  # Number of sequences in MSA

            # Create new feature array
            new_feat = np.zeros((num_seqs, num_new_tokens), dtype=feat.dtype)

            # Copy features according to token mapping
            new_feat = np.zeros((num_seqs, num_new_tokens), dtype=feat.dtype)
            new_feat[:, new_indices] = feat[:, old_indices]
            msa_features[feat_name] = new_feat

        # Handle deletion_mean (1D array)
        if "deletion_mean" in msa_features:
            feat = msa_features["deletion_mean"]
            new_feat = np.zeros(num_new_tokens, dtype=feat.dtype)
            new_feat[new_indices] = feat[old_indices]
            msa_features["deletion_mean"] = new_feat

        # Handle profile (2D array: tokens x channels)
        if "profile" in msa_features:
            feat = msa_features["profile"]
            num_channels = feat.shape[1]
            new_feat = np.zeros((num_new_tokens, num_channels), dtype=feat.dtype)
            new_feat[new_indices, :] = feat[old_indices, :]
            msa_features["profile"] = new_feat

        return msa_features

    def change_entity_mol_id(
        self,
        token_array: TokenArray,
        atom_array: AtomArray,
        max_entity_mol_id: int,
        full_atom_array: AtomArray,
        featured_tokens: set[int],
    ) -> tuple[AtomArray, AtomArray]:
        """Update entity_mol_id for atoms involved in constraints"""
        if max_entity_mol_id is None or len(featured_tokens) == 0:
            return atom_array, full_atom_array

        # Get atom indices for all constrained tokens
        constrained_atom_indices = set()
        centre_atom_indices = token_array.get_annotation("centre_atom_index")

        for token_idx in featured_tokens:
            constrained_atom_indices.add(centre_atom_indices[token_idx])

        # Get mol_ids for constrained atoms
        constrained_mol_ids = set(atom_array.mol_id[list(constrained_atom_indices)])

        # Create mapping for new entity_mol_ids
        new_id = max_entity_mol_id + 1
        id_mapping = {}
        for old_id in constrained_mol_ids:
            id_mapping[old_id] = new_id
            new_id += 1

        # Update entity_mol_ids in atom_array
        for old_id, new_id in id_mapping.items():
            mask = atom_array.mol_id == old_id
            atom_array.entity_mol_id[mask] = new_id
            mask = full_atom_array.mol_id == old_id
            full_atom_array.entity_mol_id[mask] = new_id

        return atom_array, full_atom_array

    def _get_chain_interface_mask(
        self,
        pdb_indice: pd.core.series.Series,
        atom_array_chain_id: np.array,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        df = pdb_indice.copy()

        def get_atom_mask(row):
            chain_1_id = row["chain_1_id"]
            chain_1_mask = atom_array_chain_id == chain_1_id
            matched_chain_ids = np.unique(atom_array_chain_id[chain_1_mask])
            
            # Warn if a query chain id matches multiple chain ids (should not happen)
            if len(matched_chain_ids) > 1:
                logger.warning(
                    f"[_get_chain_interface_mask] chain_1_id='{chain_1_id}' matched multiple chain_ids: {list(matched_chain_ids)}"
                )
            
            if row["type"] == "chain":
                chain_2_mask = chain_1_mask
            else:
                chain_2_id = row["chain_2_id"]
                chain_2_mask = atom_array_chain_id == chain_2_id
                matched_chain_ids_2 = np.unique(atom_array_chain_id[chain_2_mask])
                if len(matched_chain_ids_2) > 1:
                    logger.warning(
                        f"[_get_chain_interface_mask] chain_2_id='{chain_2_id}' matched multiple chain_ids: {list(matched_chain_ids_2)}"
                    )
            
            chain_1_mask = torch.tensor(chain_1_mask).bool()
            chain_2_mask = torch.tensor(chain_2_mask).bool()
            if chain_1_mask.sum() == 0 or chain_2_mask.sum() == 0:
                return None, None
            return chain_1_mask, chain_2_mask

        df["chain_1_mask"], df["chain_2_mask"] = zip(*df.apply(get_atom_mask, axis=1))
        df = df[df["chain_1_mask"].notna()]  # drop NaN

        chain_1_mask = df["chain_1_mask"].tolist()  # [N_eval, N_atom]
        chain_2_mask = df["chain_2_mask"].tolist()  # [N_eval, N_atom]

        return chain_1_mask, chain_2_mask

    def _setup_constraint_generator(self, sample_indice: pd.core.series.Series) -> Any:
        """Setup random generator with optional fixed seed"""
        if not self.constraint.get("fix_seed", False):
            return None

        constraint_seed = int(
            hashlib.sha256(sample_indice.pdb_id.encode("utf-8")).hexdigest(), 16
        ) % (2**32)
        logger.info(
            f"[constraint_seed seed]: {constraint_seed} for pdb {sample_indice.pdb_id}"
        )
        return torch.Generator().manual_seed(constraint_seed)

    def _get_base_features(
        self, token_array: TokenArray, atom_array: AtomArray
    ) -> dict[str, Any]:
        """Get base features for constraint featurizer"""
        features_dict = {}
        centre_atoms_indices = token_array.get_annotation("centre_atom_index")
        centre_atoms = atom_array[centre_atoms_indices]
        features_dict["asym_id"] = torch.Tensor(centre_atoms.asym_id_int).long()

        atom_to_token_idx_dict = {}
        for idx, token in enumerate(token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_to_token_idx_dict[atom_idx] = idx

        # ensure the order of the atom_to_token_idx is the same as the atom_array
        atom_to_token_idx = [
            atom_to_token_idx_dict[atom_idx] for atom_idx in range(len(atom_array))
        ]
        features_dict["atom_to_token_idx"] = torch.Tensor(atom_to_token_idx).long()

        features_dict["is_dna"] = torch.Tensor(atom_array.is_dna).bool()
        features_dict["is_rna"] = torch.Tensor(atom_array.is_rna).bool()
        features_dict["is_ligand"] = torch.Tensor(atom_array.is_ligand).bool()
        features_dict["atom_to_token_dict"] = atom_to_token_idx_dict
        return features_dict

    def _generate_pocket_features(
        self,
        atom_array: AtomArray,
        token_array: TokenArray,
        generator: torch.Generator,
        sample_indice: pd.core.series.Series,
    ) -> tuple[torch.Tensor, Any, set[int]]:
        """Generate pocket constraint features"""
        self._last_selected_pocket_id = None
        # pocket feature
        pocket_featurizer = PocketFeaturizer(
            token_array=token_array,
            atom_array=atom_array,
            generator=generator,
        )
        pocket_cfg = self.constraint.get("pocket", {})
        use_random_pocket = np.random.rand() < pocket_cfg.get("prob", 0)
        pocket_dbg = os.environ.get("DEBUG_POCKET_DEFINITIONS", "0") == "1"
        if pocket_dbg:
            logger.warning(
                "[DEBUG_POCKET_DEFINITIONS] enter: pdb_id=%r pocket.prob=%s "
                "use_random_pocket=%s pocket_occupancy_enable=%s "
                "pocket_occupancy_binder_mode=%s pocket_occupancy_ligand_dir=%r "
                "pocket_occupancy_cache_dir=%r",
                getattr(sample_indice, "pdb_id", None),
                pocket_cfg.get("prob", 0),
                use_random_pocket,
                pocket_cfg.get("pocket_occupancy_enable", False),
                pocket_cfg.get("pocket_occupancy_binder_mode", "sample"),
                pocket_cfg.get("pocket_occupancy_ligand_dir", ""),
                pocket_cfg.get("pocket_occupancy_cache_dir", ""),
            )
        if use_random_pocket:
            binder_asym_id = None
            binder_chain_id = None
            use_pocket_occupancy = bool(
                pocket_cfg.get("pocket_occupancy_enable", False)
            )
            _lig_occ = str(pocket_cfg.get("pocket_occupancy_ligand_dir", "") or "").strip()
            _cache_occ = str(pocket_cfg.get("pocket_occupancy_cache_dir", "") or "").strip()
            if use_pocket_occupancy and (not _lig_occ or not _cache_occ):
                logger.warning(
                    "[pocket_occupancy] enable=True but pocket_occupancy_ligand_dir or "
                    "pocket_occupancy_cache_dir empty; disabling occupancy path."
                )
                use_pocket_occupancy = False
            need_binder_chain = bool(
                pocket_cfg.get("spec_binder_chain", False)
            ) or use_pocket_occupancy
            if need_binder_chain:
                # find ligand binder
                if sample_indice.mol_1_type == "ligand":
                    binder_chain_id = str(sample_indice.chain_1_id)
                elif sample_indice.mol_2_type == "ligand":
                    binder_chain_id = str(sample_indice.chain_2_id)
                # find antibody binder
                elif (
                    f"{sample_indice.pdb_id.lower()}_{sample_indice.entity_1_id}"
                    in self.ab_top2_clusters
                ):
                    binder_chain_id = str(sample_indice.chain_1_id)
                elif (
                    f"{sample_indice.pdb_id.lower()}_{sample_indice.entity_2_id}"
                    in self.ab_top2_clusters
                ):
                    binder_chain_id = str(sample_indice.chain_2_id)
                else:
                    binder_chain_id = -1
                    logger.info("No binder found!")
                binder_asym_id = atom_array.asym_id_int[
                    atom_array.chain_id == binder_chain_id
                ]
                num_unique = len(np.unique(binder_asym_id))
                assert num_unique <= 1
                logger.info(f"found binder chains: {num_unique}")
                binder_asym_id = binder_asym_id[0] if num_unique == 1 else -1

            if use_pocket_occupancy:
                from protenix.data.pocket_occupancy_cache import (
                    get_or_compute_pocket_residues,
                    pick_mol2_for_pdb_key,
                )

                max_d = float(pocket_cfg.get("pocket_occupancy_max_distance", 8.0))
                feature_type = pocket_cfg.get("feature_type", "continuous")
                pdb_key = _normalize_pdb_key_for_pocket(sample_indice.pdb_id)
                empty_pocket = lambda: pocket_featurizer.generate(
                    {}, "continuous", size=0, distance_type=None
                )
                occ_cut = float(pocket_cfg.get("pocket_occupancy_cutoff", 8.0))
                occ_min = float(pocket_cfg.get("pocket_occupancy_min_fraction", 0.5))
                lig_dir = Path(_lig_occ).expanduser().resolve()
                cache_dir = Path(_cache_occ).expanduser().resolve()

                occ_binder_mode = str(
                    pocket_cfg.get("pocket_occupancy_binder_mode", "sample") or "sample"
                ).strip().lower()
                if occ_binder_mode not in ("sample", "all_ligands"):
                    logger.warning(
                        "[pocket_occupancy] unknown pocket_occupancy_binder_mode=%r; "
                        "using sample",
                        occ_binder_mode,
                    )
                    occ_binder_mode = "sample"

                # "sample" requires a single binder chain from indices (ligand or ab); "all_ligands"
                # uses every ligand atom in atom_array and skips this check.
                if occ_binder_mode == "sample" and (
                    binder_asym_id == -1 or binder_chain_id in (None, -1)
                ):
                    logger.warning(
                        "[pocket_occupancy] no valid binder chain; pdb_id=%s",
                        sample_indice.pdb_id,
                    )
                    pocket_constraint_feature, constrained_tokens = empty_pocket()
                elif not lig_dir.is_dir():
                    logger.warning(
                        "[pocket_occupancy] ligand dir not found: %s", lig_dir
                    )
                    pocket_constraint_feature, constrained_tokens = empty_pocket()
                else:
                    pocket_id, mol2_path = pick_mol2_for_pdb_key(
                        lig_dir, pdb_key, generator
                    )
                    if mol2_path is None:
                        logger.warning(
                            "[pocket_occupancy] no MOL2 for pdb_key=%s in %s",
                            pdb_key,
                            lig_dir,
                        )
                        pocket_constraint_feature, constrained_tokens = empty_pocket()
                    else:
                        residue_list = get_or_compute_pocket_residues(
                            atom_array,
                            str(sample_indice.pdb_id),
                            pocket_id,
                            mol2_path,
                            cache_dir,
                            occ_cut,
                            occ_min,
                        )
                        chosen_pid = pocket_id
                        self._last_selected_pocket_id = str(chosen_pid)
                        if not isinstance(residue_list, list):
                            residue_list = []

                        atom_to_token_idx_dict: dict[int, int] = {}
                        for idx_tok, token in enumerate(token_array.tokens):
                            for atom_idx in token.atom_indices:
                                atom_to_token_idx_dict[atom_idx] = idx_tok
                        atom_to_token_idx = np.array(
                            [
                                atom_to_token_idx_dict[i]
                                for i in range(len(atom_array))
                            ],
                            dtype=np.int64,
                        )

                        binder_ok = True
                        if occ_binder_mode == "all_ligands":
                            if hasattr(atom_array, "is_ligand"):
                                atom_mask_binder = np.asarray(
                                    atom_array.is_ligand, dtype=bool
                                )
                            else:
                                atom_mask_binder = np.asarray(
                                    atom_array.mol_type == "ligand", dtype=bool
                                )
                            if not np.any(atom_mask_binder):
                                logger.warning(
                                    "[pocket_occupancy] all_ligands mode but no ligand "
                                    "atoms in atom_array; pdb_id=%s",
                                    sample_indice.pdb_id,
                                )
                                pocket_constraint_feature, constrained_tokens = (
                                    empty_pocket()
                                )
                                binder_ok = False
                            else:
                                binder_token_list = atom_to_token_idx[atom_mask_binder]
                                logger.debug(
                                    "[pocket_occupancy] all_ligands: %d ligand atoms "
                                    "-> %d binder tokens",
                                    int(np.sum(atom_mask_binder)),
                                    len(binder_token_list),
                                )
                        else:
                            atom_mask_binder = atom_array.chain_id == binder_chain_id
                            binder_token_list = atom_to_token_idx[atom_mask_binder]

                        ignore_res_name = bool(
                            pocket_cfg.get("pocket_occupancy_ignore_res_name", False)
                        )

                        if binder_ok:
                            pocket_specifics = _build_pocket_specifics_from_residue_list(
                                residue_list,
                                atom_array,
                                atom_to_token_idx,
                                binder_token_list,
                                max_d,
                                ignore_res_name,
                                pdb_key,
                                chosen_pid,
                            )
                            if not pocket_specifics:
                                logger.warning(
                                    "[pocket_occupancy] no residues mapped for "
                                    "pdb_key=%s pocket=%s (from mol2=%s)",
                                    pdb_key,
                                    chosen_pid,
                                    mol2_path,
                                )
                                pocket_constraint_feature, constrained_tokens = (
                                    empty_pocket()
                                )
                            else:
                                _debug_pocket_occupancy_log(
                                    atom_array,
                                    str(sample_indice.pdb_id),
                                    pdb_key,
                                    chosen_pid,
                                    mol2_path,
                                    residue_list,
                                    pocket_specifics,
                                    occ_binder_mode,
                                    binder_chain_id,
                                    binder_token_list,
                                    pocket_dist_mat_shape=None,
                                )
                                pocket_constraint_feature, constrained_tokens = (
                                    pocket_featurizer.generate_spec_constraint(
                                        pocket_specifics,
                                        feature_type=feature_type,
                                    )
                                )
                                if os.environ.get("DEBUG_POCKET_OCCUPANCY", "0") == "1":
                                    nnz_feat = int(
                                        torch.count_nonzero(pocket_constraint_feature).item()
                                    )
                                    logger.warning(
                                        "[DEBUG_POCKET_OCCUPANCY] "
                                        "pocket_constraint_feature.shape=%s "
                                        "nnz(raw continuous)=%s unique_constrained_token_idx=%s "
                                        "(after generate_spec_constraint encode)",
                                        tuple(pocket_constraint_feature.shape),
                                        nnz_feat,
                                        len(constrained_tokens)
                                        if constrained_tokens is not None
                                        else 0,
                                    )
                                logger.debug(
                                    "[pocket_occupancy] pdb_key=%s pocket=%s "
                                    "(%d residue specs, max_distance=%s)",
                                    pdb_key,
                                    chosen_pid,
                                    len(pocket_specifics),
                                    max_d,
                                )
            else:
                pocket_constraint_feature, constrained_tokens = pocket_featurizer.generate(
                    size=pocket_cfg.get("size", 0.0),
                    feature_type=pocket_cfg.get("feature_type", "continuous"),
                    max_distance_range=pocket_cfg.get("max_distance_range", {}),
                    sample_group=pocket_cfg.get("group", "complex"),
                    distance_type=pocket_cfg.get("distance_type", "center_atom"),
                    spec_binder_asym_id=binder_asym_id,
                )
            if __import__("os").environ.get("DEBUG_POCKET_CONSTRAINT", "0") == "1":
                logger.warning(
                    "[DEBUG_POCKET_CONSTRAINT] Just generated pocket: pocket_constraint_feature.shape=%s, "
                    "constrained_tokens (num)=%s, binder_asym_id=%s",
                    getattr(pocket_constraint_feature, "shape", None),
                    len(constrained_tokens) if constrained_tokens else 0,
                    binder_asym_id,
                )
                _debug_pocket_definitions_maybe_trace()
        else:
            if pocket_dbg:
                logger.warning(
                    "[DEBUG_POCKET_DEFINITIONS] skipped: use_random_pocket=False "
                    "(pocket.prob roll); returning empty pocket"
                )
            pocket_constraint_feature, constrained_tokens = pocket_featurizer.generate(
                {}, "continuous", size=0, distance_type=None
            )
        return pocket_constraint_feature, pocket_featurizer, constrained_tokens

    def _generate_contact_features(
        self,
        atom_array: AtomArray,
        token_array: TokenArray,
        generator: torch.Generator,
        features_dict: dict[str, Any],
        pdb_indice: pd.core.series.Series,
    ) -> tuple[torch.Tensor, Any, list[tuple[int, int]], set[int]]:
        """Generate contact constraint features"""
        contact_featurizer = ContactFeaturizer(
            token_array=token_array,
            atom_array=atom_array,
            generator=generator,
        )
        # contact_feature_prob =
        use_random_contact = np.random.rand() < self.constraint.get("contact", {}).get(
            "prob", 0
        )
        selected_pairs = []
        if use_random_contact:
            # collecte all interfaces in cropped_tokens
            interface_asym_pairs = []
            if self.constraint["contact"].get("group", "complex") == "interface":
                # Optional debug info (kept as logger.debug, no prints)
                if isinstance(pdb_indice, pd.Series):
                    chain_1_id_value = pdb_indice.get("chain_1_id", "N/A")
                    chain_2_id_value = pdb_indice.get("chain_2_id", "N/A")
                    pdb_id_value = pdb_indice.get("pdb_id", "N/A")
                else:
                    chain_1_id_value = pdb_indice.get("chain_1_id", "N/A") if isinstance(pdb_indice, dict) else "N/A"
                    chain_2_id_value = pdb_indice.get("chain_2_id", "N/A") if isinstance(pdb_indice, dict) else "N/A"
                    pdb_id_value = pdb_indice.get("pdb_id", "N/A") if isinstance(pdb_indice, dict) else "N/A"
                
                unique_chain_ids = np.unique(atom_array.chain_id)
                logger.debug(
                    f"[constraint] pdb_id={pdb_id_value}, "
                    f"pdb_indice chain_1_id={chain_1_id_value}, chain_2_id={chain_2_id_value}, "
                    f"atom_array unique chain_ids={list(unique_chain_ids)}"
                )

                def _get_asym_id(atom_mask):
                    token_idx = features_dict["atom_to_token_idx"][
                        atom_mask.bool()
                    ].long()
                    if len(token_idx) == 0:
                        logger.warning("[DEBUG constraint] Empty token_idx in _get_asym_id")
                        return None
                    asym_ids = features_dict["asym_id"][token_idx]
                    unique_asym_ids = torch.unique(asym_ids)
                    
                    # Debug: Print detailed info when multiple asym_ids found
                    if len(unique_asym_ids) > 1:
                        chain_ids_in_mask = atom_array.chain_id[atom_mask.bool().cpu().numpy()]
                        unique_chain_ids_in_mask = np.unique(chain_ids_in_mask)
                        asym_id_counts = {uid.item(): torch.sum(asym_ids == uid).item() for uid in unique_asym_ids}
                        
                        # Keep warning: this indicates inconsistent mapping (should be fixed in data prep)
                        logger.warning(
                            f"[constraint] Multiple asym_ids in chain mask! "
                            f"unique_asym_ids={unique_asym_ids.tolist()}, "
                            f"asym_id_counts={asym_id_counts}, "
                            f"chain_ids_in_mask={list(unique_chain_ids_in_mask)}, "
                            f"pdb_indice chain_1_id={chain_1_id_value}"
                        )
                        
                        # Debug: Check if the same chain_id appears with different asym_ids in full atom_array
                        if len(unique_chain_ids_in_mask) == 1:
                            problem_chain_id = unique_chain_ids_in_mask[0]
                            # Find all atoms with this chain_id in the full atom_array
                            full_chain_mask = atom_array.chain_id == problem_chain_id
                            full_asym_ids = features_dict["asym_id"][features_dict["atom_to_token_idx"][full_chain_mask].long()]
                            unique_full_asym_ids = torch.unique(full_asym_ids)
                            logger.warning(
                                f"[constraint] Problem: chain_id='{problem_chain_id}' has {len(unique_full_asym_ids)} different asym_ids in full atom_array: {unique_full_asym_ids.tolist()}"
                            )
                            
                            # Check chain starts for this chain_id
                            chain_starts = get_chain_starts(atom_array, add_exclusive_stop=True)
                            chain_id_starts = []
                            for i in range(len(chain_starts) - 1):
                                start = chain_starts[i]
                                stop = chain_starts[i + 1]
                                if atom_array.chain_id[start] == problem_chain_id:
                                    asym_id_at_start = features_dict["asym_id"][features_dict["atom_to_token_idx"][start].long()].item()
                                    num_atoms = stop - start
                                    chain_id_starts.append((start, stop, asym_id_at_start, num_atoms))
                            
                            logger.warning(
                                f"[constraint] Chain '{problem_chain_id}' appears {len(chain_id_starts)} times:"
                            )
                            for idx, (start, stop, asym_id, num_atoms) in enumerate(chain_id_starts):
                                logger.warning(
                                    f"  [{idx}] start={start}, stop={stop}, num_atoms={num_atoms}, asym_id={asym_id}"
                                )
                            
                            # Check if these chains are actually separate (not continuous)
                            if len(chain_id_starts) > 1:
                                for idx in range(len(chain_id_starts) - 1):
                                    prev_stop = chain_id_starts[idx][1]
                                    next_start = chain_id_starts[idx + 1][0]
                                    gap = next_start - prev_stop
                                    logger.warning(
                                        f"  Gap between chain {idx} and {idx+1}: {gap} atoms"
                                    )
                    assert len(unique_asym_ids) == 1
                    return asym_ids[0].item()

                chain_1_mask, chain_2_mask = self._get_chain_interface_mask(
                    pdb_indice=pdb_indice, atom_array_chain_id=atom_array.chain_id
                )

                interface_asym_pairs = []
                for c1, c2 in zip(chain_1_mask, chain_2_mask):
                    if torch.equal(c1, c2):
                        continue
                    asym_id1 = _get_asym_id(c1)
                    asym_id2 = _get_asym_id(c2)
                    if asym_id1 is not None and asym_id2 is not None:
                        interface_asym_pairs.append((asym_id1, asym_id2))

                if len(interface_asym_pairs) == 0:
                    logger.info("Interface for constraint feature is not found")

            contact_constraint_feature, selected_pairs, constrained_tokens = (
                contact_featurizer.generate(
                    size=self.constraint["contact"].get("size", 0.0),
                    chain_pairs=interface_asym_pairs,
                    feature_type=self.constraint["contact"].get(
                        "feature_type", "continuous"
                    ),
                    max_distance_range=self.constraint["contact"].get(
                        "max_distance_range", {}
                    ),
                    min_distance_threshold=self.constraint["contact"].get(
                        "min_distance", 0.0
                    ),
                    sample_group=self.constraint["contact"].get("group", "complex"),
                    distance_type=self.constraint["contact"].get(
                        "distance_type", "center_atom"
                    ),
                )
            )
        else:
            contact_constraint_feature, selected_pairs, constrained_tokens = (
                contact_featurizer.generate(
                    {}, None, "continuous", size=0, distance_type=None
                )
            )
        
        # Generate ribbon constraints if enabled
        ribbon_enabled = self.constraint.get("ribbon", {}).get("enable", False)
        if ribbon_enabled:
            # logger.info("[Ribbon Constraint] Generating dynamic ribbon constraints")
            ribbon_constraint_dict = ConstraintFeatureGenerator.generate_ribbon_constraints_dynamic(
                atom_array=atom_array,
                token_array=token_array,
                min_distance=self.constraint.get("ribbon", {}).get("min_distance", 4.0),
                max_distance=self.constraint.get("ribbon", {}).get("max_distance", 6.0),
                tolerance=self.constraint.get("ribbon", {}).get("tolerance", 1.0),
                min_common_residues=self.constraint.get("ribbon", {}).get("min_common_residues", 5),
                distance_ratio_threshold=self.constraint.get("ribbon", {}).get("distance_ratio_threshold", 1.5),
            )
            
            # Merge ribbon constraints into contact constraints
            ribbon_contact_feature = ribbon_constraint_dict.get("contact", None)
            if ribbon_contact_feature is not None:
                # ribbon_contact_feature shape: [N_token, N_token, 2] (min_dist, max_dist)
                # contact_constraint_feature shape: [N_token, N_token, 2] (min_dist, max_dist)
                # Merge by taking the tighter (more restrictive) constraints
                # For min_dist: take max (more restrictive lower bound)
                # For max_dist: take min (more restrictive upper bound)
                ribbon_min_dist = ribbon_contact_feature[..., 0]  # [N_token, N_token]
                ribbon_max_dist = ribbon_contact_feature[..., 1]  # [N_token, N_token]
                contact_min_dist = contact_constraint_feature[..., 0]  # [N_token, N_token]
                contact_max_dist = contact_constraint_feature[..., 1]  # [N_token, N_token]
                
                # Find where ribbon constraints are valid (non-padding)
                ribbon_valid_mask = (ribbon_min_dist >= 0) & (ribbon_max_dist > 0)
                
                # Merge: where ribbon is valid, use tighter constraints
                # If contact constraint at a pair is padding (contact_max_dist==0),
                # don't clamp ribbon to 0; directly use ribbon range.
                contact_valid_mask = contact_max_dist > 0

                merged_min_dist = torch.where(
                    ribbon_valid_mask,
                    torch.where(
                        contact_valid_mask,
                        torch.maximum(ribbon_min_dist, contact_min_dist),
                        ribbon_min_dist,
                    ),
                    contact_min_dist,
                )
                merged_max_dist = torch.where(
                    ribbon_valid_mask,
                    torch.where(
                        contact_valid_mask,
                        torch.minimum(ribbon_max_dist, contact_max_dist),
                        ribbon_max_dist,
                    ),
                    contact_max_dist,
                )
                
                # Update contact_constraint_feature
                contact_constraint_feature = torch.stack([merged_min_dist, merged_max_dist], dim=-1)

                # Optional debug: print ribbon constraint stats
                # Enable via env:
                #   DEBUG_RIBBON_CONSTRAINT=1
                #   DEBUG_RIBBON_CONSTRAINT_PRINT_N=10
                global _RIBBON_DEBUG_PRINTED
                if (
                    os.environ.get("DEBUG_RIBBON_CONSTRAINT", "0") == "1"
                    and not _RIBBON_DEBUG_PRINTED
                ):
                    _RIBBON_DEBUG_PRINTED = True
                    try:
                        print_n = int(os.environ.get("DEBUG_RIBBON_CONSTRAINT_PRINT_N", "10"))
                        valid_pairs = ribbon_valid_mask.nonzero(as_tuple=False)
                        n_valid_pairs = valid_pairs.shape[0]
                        contact_valid_pairs = (contact_max_dist > 0).sum().item()
                        n_constrained_tokens = (
                            torch.nonzero(
                                ribbon_valid_mask.any(dim=0) | ribbon_valid_mask.any(dim=1)
                            ).numel()
                        )

                        # Map token index -> asym_id -> chain_id for easier debugging.
                        # NOTE: This runs only once per training job when DEBUG_RIBBON_CONSTRAINT=1.
                        token_asym_id = None
                        asym_id_to_chain = {}
                        try:
                            if "asym_id" in features_dict:
                                token_asym_id = features_dict["asym_id"].to(torch.long)
                            centre_atom_indices = token_array.get_annotation("centre_atom_index")
                            centre_atoms = atom_array[centre_atom_indices]
                            # NOTE: centre_atoms.asym_id_int / chain_id may be numpy-backed.
                            # Use numpy for robust indexing in debug mode.
                            asym_ids_np = np.asarray(centre_atoms.asym_id_int)
                            chain_ids_np = np.asarray(centre_atoms.chain_id)
                            unique_asym_ids = np.unique(asym_ids_np)
                            for aid in unique_asym_ids:
                                mask = asym_ids_np == aid
                                chain_ids_for_aid = chain_ids_np[mask]
                                # chain_ids_for_aid is array of strings; keep unique and sorted
                                chain_unique = sorted(set(map(str, chain_ids_for_aid.tolist())))
                                asym_id_to_chain[int(aid)] = ",".join(chain_unique)
                        except Exception as e:
                            logger.warning(
                                "[DEBUG_RIBBON_CONSTRAINT] token->chain mapping failed: %s",
                                str(e),
                            )

                        logger.info(
                            "[DEBUG_RIBBON_CONSTRAINT] ribbon_contact_feature.shape=%s "
                            "ribbon_valid_pairs=%d contact_valid_pairs=%s n_constrained_tokens=%d",
                            tuple(ribbon_contact_feature.shape),
                            n_valid_pairs,
                            contact_valid_pairs,
                            n_constrained_tokens,
                        )

                        if n_valid_pairs > 0:
                            # Aggregate unique chain pairs for the whole ribbon_valid_mask.
                            # This helps you see "which chain with which chain".
                            chain_pair_counts = {}
                            try:
                                if token_asym_id is not None:
                                    for idx_pair in valid_pairs.tolist():
                                        i, j = idx_pair
                                        ai = int(token_asym_id[i].item())
                                        aj = int(token_asym_id[j].item())
                                        # undirected chain pair
                                        key_asym = tuple(sorted([ai, aj]))
                                        chain_pair_counts[key_asym] = chain_pair_counts.get(key_asym, 0) + 1
                            except Exception as e:
                                logger.warning(
                                    "[DEBUG_RIBBON_CONSTRAINT] chain_pair aggregation failed: %s",
                                    str(e),
                                )

                            if chain_pair_counts:
                                topk = int(os.environ.get("DEBUG_RIBBON_CONSTRAINT_TOPK", "10"))
                                top_items = sorted(chain_pair_counts.items(), key=lambda x: x[1], reverse=True)[:topk]
                                logger.info(
                                    "[DEBUG_RIBBON_CONSTRAINT] top chain pairs (asym_id_a,asym_id_b)->count (valid token pairs):"
                                )
                                for (a0, a1), cnt in top_items:
                                    c0 = asym_id_to_chain.get(a0, f"asym{a0}")
                                    c1 = asym_id_to_chain.get(a1, f"asym{a1}")
                                    logger.info(
                                        "  chain(%s) <-> chain(%s): %d",
                                        c0,
                                        c1,
                                        cnt,
                                    )

                            valid_pairs_to_print = valid_pairs[:print_n]
                            logger.info(
                                "[DEBUG_RIBBON_CONSTRAINT] first %d valid (token_i, token_j) pairs: ",
                                min(print_n, n_valid_pairs),
                            )
                            for k in range(valid_pairs_to_print.shape[0]):
                                i = int(valid_pairs_to_print[k, 0].item())
                                j = int(valid_pairs_to_print[k, 1].item())
                                if token_asym_id is not None:
                                    ai = int(token_asym_id[i].item())
                                    aj = int(token_asym_id[j].item())
                                    ci = asym_id_to_chain.get(ai, f"asym{ai}")
                                    cj = asym_id_to_chain.get(aj, f"asym{aj}")
                                else:
                                    ai = aj = None
                                    ci = cj = None
                                logger.info(
                                    "  pair[%d]: i=%d j=%d (asym=%s<->%s chain=%s<->%s) | ribbon(min,max)=(%.3f, %.3f) | merged(min,max)=(%.3f, %.3f)",
                                    k,
                                    i,
                                    j,
                                    str(ai),
                                    str(aj),
                                    str(ci),
                                    str(cj),
                                    float(ribbon_min_dist[i, j].item()),
                                    float(ribbon_max_dist[i, j].item()),
                                    float(merged_min_dist[i, j].item()),
                                    float(merged_max_dist[i, j].item()),
                                )
                    except Exception as e:
                        logger.warning("[DEBUG_RIBBON_CONSTRAINT] debug failed: %s", str(e))
                
                # Update constrained_tokens to include ribbon constraints
                ribbon_constrained_tokens = torch.nonzero(ribbon_valid_mask.any(dim=0) | ribbon_valid_mask.any(dim=1)).squeeze(-1)
                constrained_tokens = constrained_tokens | set(ribbon_constrained_tokens.tolist())
                
                # logger.info(f"[Ribbon Constraint] Merged ribbon constraints into contact features: "
                #            f"{ribbon_valid_mask.sum().item()} constraint pairs added")
        
        return (
            contact_constraint_feature,
            contact_featurizer,
            selected_pairs,
            constrained_tokens,
        )

    def _generate_contact_atom_features(
        self,
        atom_array: AtomArray,
        token_array: TokenArray,
        generator: torch.Generator,
        features_dict: dict[str, Any],
        pdb_indice: pd.core.series.Series,
    ) -> tuple[torch.Tensor, Any, list[tuple[int, int]], set[int]]:
        """Generate contact atom constraint features"""
        contact_atom_featurizer = ContactAtomFeaturizer(
            token_array=token_array, atom_array=atom_array, generator=generator
        )
        use_random_contact_atom = np.random.rand() < self.constraint.get(
            "contact_atom", {}
        ).get("prob", 0)
        selected_pairs = []
        if use_random_contact_atom:
            # collecte all interfaces in cropped_tokens
            interface_asym_pairs = []
            if self.constraint["contact_atom"].get("group", "complex") == "interface":
                # Optional debug info (kept as logger.debug, no prints)
                if isinstance(pdb_indice, pd.Series):
                    chain_1_id_value = pdb_indice.get("chain_1_id", "N/A")
                    chain_2_id_value = pdb_indice.get("chain_2_id", "N/A")
                    pdb_id_value = pdb_indice.get("pdb_id", "N/A")
                else:
                    chain_1_id_value = pdb_indice.get("chain_1_id", "N/A") if isinstance(pdb_indice, dict) else "N/A"
                    chain_2_id_value = pdb_indice.get("chain_2_id", "N/A") if isinstance(pdb_indice, dict) else "N/A"
                    pdb_id_value = pdb_indice.get("pdb_id", "N/A") if isinstance(pdb_indice, dict) else "N/A"
                
                unique_chain_ids = np.unique(atom_array.chain_id)
                logger.debug(
                    f"[constraint_atom] pdb_id={pdb_id_value}, "
                    f"pdb_indice chain_1_id={chain_1_id_value}, chain_2_id={chain_2_id_value}, "
                    f"atom_array unique chain_ids={list(unique_chain_ids)}"
                )

                def _get_asym_id(atom_mask):
                    token_idx = features_dict["atom_to_token_idx"][
                        atom_mask.bool()
                    ].long()
                    if len(token_idx) == 0:
                        logger.warning("[DEBUG constraint_atom] Empty token_idx in _get_asym_id")
                        return None
                    asym_ids = features_dict["asym_id"][token_idx]
                    unique_asym_ids = torch.unique(asym_ids)
                    
                    # Debug: Print detailed info when multiple asym_ids found
                    if len(unique_asym_ids) > 1:
                        chain_ids_in_mask = atom_array.chain_id[atom_mask.bool().cpu().numpy()]
                        unique_chain_ids_in_mask = np.unique(chain_ids_in_mask)
                        asym_id_counts = {uid.item(): torch.sum(asym_ids == uid).item() for uid in unique_asym_ids}
                        logger.warning(
                            f"[constraint_atom] Multiple asym_ids in chain mask! "
                            f"unique_asym_ids={unique_asym_ids.tolist()}, "
                            f"asym_id_counts={asym_id_counts}, "
                            f"chain_ids_in_mask={list(unique_chain_ids_in_mask)}, "
                            f"pdb_indice chain_1_id={chain_1_id_value}"
                        )
                    assert len(unique_asym_ids) == 1
                    return asym_ids[0].item()

                chain_1_mask, chain_2_mask = self._get_chain_interface_mask(
                    pdb_indice=pdb_indice, atom_array_chain_id=atom_array.chain_id
                )

                interface_asym_pairs = []
                for c1, c2 in zip(chain_1_mask, chain_2_mask):
                    if torch.equal(c1, c2):
                        continue
                    asym_id1 = _get_asym_id(c1)
                    asym_id2 = _get_asym_id(c2)
                    if asym_id1 is not None and asym_id2 is not None:
                        interface_asym_pairs.append((asym_id1, asym_id2))

                if len(interface_asym_pairs) == 0:
                    logger.info("Interface for constraint feature is not found")

            contact_atom_constraint_feature, selected_pairs, constrained_tokens = (
                contact_atom_featurizer.generate(
                    size=self.constraint["contact_atom"].get("size", 0.0),
                    chain_pairs=interface_asym_pairs,
                    feature_type=self.constraint["contact_atom"].get(
                        "feature_type", "continuous"
                    ),
                    max_distance_range=self.constraint["contact_atom"].get(
                        "max_distance_range", {}
                    ),
                    min_distance_threshold=self.constraint["contact_atom"].get(
                        "min_distance", 0.0
                    ),
                    sample_group=self.constraint["contact_atom"].get(
                        "group", "complex"
                    ),
                    distance_type=self.constraint["contact_atom"].get(
                        "distance_type", "atom"
                    ),
                )
            )
        else:
            contact_atom_constraint_feature, selected_pairs, constrained_tokens = (
                contact_atom_featurizer.generate(
                    {}, None, "continuous", size=0, distance_type=None
                )
            )
        return (
            contact_atom_constraint_feature,
            contact_atom_featurizer,
            selected_pairs,
            constrained_tokens,
        )

    def _generate_substructure_features(
        self,
        atom_array: AtomArray,
        token_array: TokenArray,
        generator: torch.Generator,
        sample_indice: pd.core.series.Series,
    ) -> tuple[torch.Tensor, Any, set[int]]:
        """Generate substructure constraint features"""
        substructure_featurizer = SubStructureFeaturizer(
            token_array=token_array,
            atom_array=atom_array,
            generator=generator,
        )
        use_random_substructure = np.random.rand() < self.constraint.get(
            "substructure", {}
        ).get("prob", 0)
        if use_random_substructure:
            spec_asym_id = None
            if self.constraint["substructure"].get("spec_asym_id", False):
                # find ligand chain
                if sample_indice.mol_1_type == "ligand":
                    binder_chain_id = str(sample_indice.chain_1_id)
                    target_chain_id = str(sample_indice.chain_2_id)
                elif sample_indice.mol_2_type == "ligand":
                    binder_chain_id = str(sample_indice.chain_2_id)
                    target_chain_id = str(sample_indice.chain_1_id)
                # find antibody chain
                elif (
                    f"{sample_indice.pdb_id.lower()}_{sample_indice.entity_1_id}"
                    in self.ab_top2_clusters
                ):
                    binder_chain_id = str(sample_indice.chain_1_id)
                    target_chain_id = str(sample_indice.chain_2_id)
                elif (
                    f"{sample_indice.pdb_id.lower()}_{sample_indice.entity_2_id}"
                    in self.ab_top2_clusters
                ):
                    binder_chain_id = str(sample_indice.chain_2_id)
                    target_chain_id = str(sample_indice.chain_1_id)
                else:
                    target_chain_id = binder_chain_id = -1
                    logger.info(f"No specific chain found!")

                chain_choice = self.constraint["substructure"].get(
                    "spec_chain_type", "binder"
                )
                if chain_choice == "binder":
                    spec_chain_id = binder_chain_id
                elif chain_choice == "target":
                    spec_chain_id = target_chain_id
                else:
                    raise ValueError(
                        f"Invalid spec_chain_type: {self.constraint['substructure'].get('spec_chain_type', 'binder')}"
                    )

                if spec_chain_id != -1:
                    spec_asym_id = atom_array.asym_id_int[
                        atom_array.chain_id == spec_chain_id
                    ]
                    num_unique = len(np.unique(spec_asym_id))
                    assert num_unique <= 1
                    logger.info(f"found {chain_choice} chain: {num_unique}")
                    spec_asym_id = spec_asym_id[0] if num_unique == 1 else -1

            substructure_constraint_feature, constrained_tokens = (
                substructure_featurizer.generate(
                    mol_type_pairs=self.constraint["substructure"].get(
                        "mol_type_pairs", {}
                    ),
                    feature_type=self.constraint["substructure"].get(
                        "feature_type", "one_hot"
                    ),
                    size=self.constraint["substructure"].get("size", 0),
                    ratios=self.constraint["substructure"].get(
                        "ratios", {"full": [0.0, 0.5, 1.0], "partial": 0.3}
                    ),
                    coord_noise_scale=self.constraint["substructure"].get(
                        "coord_noise_scale", 0.05
                    ),
                    spec_asym_id=spec_asym_id,
                )
            )
        else:
            substructure_constraint_feature, constrained_tokens = (
                substructure_featurizer.generate(
                    mol_type_pairs={},
                    feature_type="one_hot",
                    size=0,
                    ratios={"full": [0.0, 0.5, 1.0], "partial": 0.3},
                    coord_noise_scale=0.05,
                    spec_asym_id=None,
                )
            )
        return (
            substructure_constraint_feature,
            substructure_featurizer,
            constrained_tokens,
        )

    def _get_feature_statistics(
        self,
        constraint_feature: dict[str, torch.Tensor],
        atom_array: AtomArray,
        token_array: TokenArray,
        features_dict: dict[str, Any],
        contact_atom_featurizer: Any,
        substructure_featurizer: Any,
    ) -> dict[str, Any]:
        """Log statistics about generated features"""
        token_idx_1, token_idx_2 = torch.nonzero(
            torch.triu(constraint_feature["contact"][..., 1]), as_tuple=True
        )
        asym_id_1 = features_dict["asym_id"][token_idx_1]
        asym_id_2 = features_dict["asym_id"][token_idx_2]

        res_id_1 = atom_array.res_id[
            token_array[token_idx_1].get_annotation("centre_atom_index")
        ]
        res_id_2 = atom_array.res_id[
            token_array[token_idx_2].get_annotation("centre_atom_index")
        ]
        contact_distance = constraint_feature["contact"][token_idx_1, token_idx_2, 1]

        # logging contact atom feature
        atom_idx_1, atom_idx_2 = torch.nonzero(
            torch.triu(constraint_feature["contact_atom"][..., 1]), as_tuple=True
        )
        contact_atom_real_distance = (
            contact_atom_featurizer.get_real_distance(atom_idx_1, atom_idx_2)
            if len(atom_idx_1) > 0
            else None
        )
        contact_atom_max_distance = constraint_feature["contact_atom"][
            atom_idx_1, atom_idx_2, 1
        ]
        contact_atom_min_distance = constraint_feature["contact_atom"][
            atom_idx_1, atom_idx_2, 0
        ]
        num_contact_atom = contact_atom_max_distance.shape[0]

        # logging pocket feature
        binder_idx, pocket_idx = torch.nonzero(
            constraint_feature["pocket"].squeeze(-1), as_tuple=True
        )
        binder_idx = binder_idx.unique()
        pocket_idx = pocket_idx.unique()
        if binder_idx.numel() > 1:
            pocket_distance = constraint_feature["pocket"][binder_idx[0], pocket_idx, 0]
        else:
            pocket_distance = -1

        # logging substructure feature
        sub_structure_info = substructure_featurizer.analyze_features(
            constraint_feature["substructure"]
        )

        feature_info = {
            "contact_info": {
                "asym_id": torch.stack([asym_id_1, asym_id_2]),
                "token_id": torch.stack([token_idx_1, token_idx_2]),
                "res_id": torch.tensor(np.stack([res_id_1, res_id_2])),
                "distance": contact_distance,
            },
            "contact_atom_info": {
                "distance": contact_atom_real_distance,
                "max_distance": contact_atom_max_distance,
                "min_distance": contact_atom_min_distance,
                "num_contact_atom": num_contact_atom,
            },
            "pocket_info": {
                "binder_tokenid": binder_idx,
                "pocket_tokenid": pocket_idx,
                "distance": pocket_distance,
            },
            "substructure_info": sub_structure_info,
        }

        return feature_info

    def _log_feature_statistics(self, feature_info: dict[str, Any]) -> dict[str, Any]:
        log_constraint = {}
        if feature_info.get("pocket_info", None) is not None:
            binder_tokens, pocket_tokens, distance = feature_info[
                "pocket_info"
            ].values()
            pocket_msg = ";".join(
                [
                    ",".join(map(str, binder_tokens.flatten().tolist())),
                    ",".join(map(str, pocket_tokens.flatten().tolist())),
                ]
            )
            log_constraint["pocket_msg"] = pocket_msg
            log_constraint["pocket_N_binder"] = torch.tensor(
                feature_info["pocket_info"]["binder_tokenid"].shape[0]
            )
            log_constraint["pocket_N_pocket"] = torch.tensor(
                feature_info["pocket_info"]["pocket_tokenid"].shape[0]
            )
        if feature_info.get("contact_info", None) is not None:
            asym_id, token_id, res_id, distance = feature_info["contact_info"].values()
            N_contact = asym_id.shape[-1]
            contact_msg = ";".join(
                [
                    ",".join(map(str, asym_id.flatten().tolist())),
                    ",".join(map(str, token_id.flatten().tolist())),
                    ",".join(map(str, distance.flatten().tolist())),
                ]
            )
            log_constraint["contact_N_pair"] = torch.tensor(N_contact)
            log_constraint["contact_msg"] = contact_msg
        if feature_info.get("contact_atom_info", None) is not None:
            distance, max_distance, min_distance, num_contact_atom = feature_info[
                "contact_atom_info"
            ].values()
            N_contact = num_contact_atom
            log_constraint["contact_atom_N_pair"] = torch.tensor(N_contact)
            log_constraint["contact_atom_distance"] = distance
            log_constraint["contact_atom_max_distance"] = max_distance
            log_constraint["contact_atom_min_distance"] = min_distance

        if feature_info.get("substructure_info", None) is not None:
            log_constraint["substructure_active_tokens"] = torch.tensor(
                feature_info["substructure_info"]["num_active_tokens"]
            )
            log_constraint["substructure_active_token_ratio"] = torch.tensor(
                feature_info["substructure_info"]["active_token_ratio"]
            )
            log_constraint["substructure_bin0_cnt"] = torch.tensor(
                feature_info["substructure_info"]["distance_distribution"][
                    "bin_0_count"
                ]
            )
            log_constraint["substructure_bin1_cnt"] = torch.tensor(
                feature_info["substructure_info"]["distance_distribution"][
                    "bin_1_count"
                ]
            )
            log_constraint["substructure_bin2_cnt"] = torch.tensor(
                feature_info["substructure_info"]["distance_distribution"][
                    "bin_2_count"
                ]
            )
            log_constraint["substructure_bin3_cnt"] = torch.tensor(
                feature_info["substructure_info"]["distance_distribution"][
                    "bin_3_count"
                ]
            )

        return log_constraint


class ConstraintFeaturizer(object):
    def __init__(
        self,
        token_array: TokenArray,
        atom_array: AtomArray,
        pad_value: float = 0,
        generator=None,
    ):
        self.token_array = token_array
        self.atom_array = atom_array
        self.pad_value = pad_value
        self.generator = generator
        self._get_base_info()

    @staticmethod
    def one_hot_encoder(feature: torch.Tensor, num_classes: int):
        # Create mask for padding values (-1)
        pad_mask = feature == -1

        # Replace -1 with 0 temporarily for F.one_hot
        feature = torch.where(pad_mask, torch.zeros_like(feature), feature)

        # Convert to one-hot
        one_hot = F.one_hot(feature, num_classes=num_classes).float()

        # Zero out the one-hot vectors for padding positions
        one_hot[pad_mask] = 0.0

        return one_hot

    def encode(self, feature: torch.Tensor, feature_type: str, **kwargs):
        if feature_type == "one_hot":
            return ConstraintFeaturizer.one_hot_encoder(
                feature, num_classes=kwargs.get("num_classes", -1)
            )
        elif feature_type == "continuous":
            return feature
        else:
            raise RuntimeError(f"Invalid feature_type: {feature_type}")

    def _get_base_info(self):
        token_centre_atom_indices = self.token_array.get_annotation("centre_atom_index")
        centre_atoms = self.atom_array[token_centre_atom_indices]
        self.asymid = torch.tensor(centre_atoms.asym_id_int, dtype=torch.long)
        self.is_ligand = torch.tensor(centre_atoms.is_ligand, dtype=torch.bool)
        self.is_protein = torch.tensor(centre_atoms.is_protein, dtype=torch.bool)
        self.entity_type_dict = {"P": self.is_protein, "L": self.is_ligand}

    def _get_generation_basics(self, distance_type: str = "center_atom"):
        token_centre_atom_indices = self.token_array.get_annotation("centre_atom_index")
        centre_atoms = self.atom_array[token_centre_atom_indices]

        # is_resolved mask
        self.token_resolved_mask = torch.tensor(
            centre_atoms.is_resolved, dtype=torch.bool
        )
        self.token_resolved_maskmat = (
            self.token_resolved_mask[:, None] * self.token_resolved_mask[None, :]
        )

        # distance matrix
        if distance_type == "center_atom":
            # center atom distance
            self.token_distance = torch.tensor(
                cdist(centre_atoms.coord, centre_atoms.coord), dtype=torch.float64
            )
        elif distance_type == "any_atom":
            # any atom distance
            all_atom_resolved_mask = (
                self.atom_array.is_resolved[:, None]
                * self.atom_array.is_resolved[None, :]
            )
            all_atom_distance = cdist(self.atom_array.coord, self.atom_array.coord)
            all_atom_distance[~all_atom_resolved_mask] = np.inf

            token_atoms_num = [
                len(_atoms)
                for _atoms in self.token_array.get_annotation("atom_indices")
            ]
            atom_token_num = np.repeat(
                np.arange(len(self.token_array)), token_atoms_num
            )

            self.token_distance = torch.zeros(
                (len(centre_atoms), len(centre_atoms)), dtype=torch.float64
            )
            for i, j in np.ndindex(self.token_distance.shape):
                atom_pairs_mask = np.ix_(atom_token_num == i, atom_token_num == j)
                self.token_distance[i, j] = np.min(all_atom_distance[atom_pairs_mask])
        elif distance_type == "atom":
            raise ValueError(
                "Not implement in this class, please use ContactAtomFeaturizer"
            )
        else:
            raise ValueError(f"Not recognized distance_type: {distance_type}")

    def generate(self):
        pass

    def generate_spec_constraint(self):
        pass


class ContactFeaturizer(ConstraintFeaturizer):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def get_valid_contact_feature(
        self,
        valid_contact_type: str,
        max_distance_threshold: float,
        min_distance_threshold: float,
    ) -> torch.Tensor:
        """
        Find all valid pairs of entities that satisfy the given contact distance requirements.

        Parameters:
         - valid_contact_type: A two-charactor type to represent entity pairs under consideration.
            e.g. PP, PL, P_
         - max_distance_threshold: The maximum allowable distance for a contact to be considered valid.
         - min_distance_threshold: The minimum allowable distance for a contact to be considered valid.
        Returns:
         - shape=(N_token, N_token), A matrix generate contact within the specified distance range.
        """
        # get valid contact type
        query_type, key_type = valid_contact_type

        if key_type == "_":
            # intra chain contact
            assert query_type == "P", "only support intra-protein contact for now"

            valid_chain_mask = self.asymid[:, None] == self.asymid[None, :]

            # skip closest squence pairs
            n_neighbor = 20  # TODO: tune this parameter
            valid_chain_mask_right = torch.triu(valid_chain_mask, diagonal=n_neighbor)
            valid_chain_mask_left = torch.tril(valid_chain_mask, diagonal=-n_neighbor)
            valid_chain_mask = valid_chain_mask_right | valid_chain_mask_left

        else:
            # inter chain contact
            assert (
                query_type in "PL" and key_type in "PL"
            ), f"[Error]contact type not support: {valid_contact_type}"

            valid_type_mask = (
                self.entity_type_dict[query_type][None, :]
                & self.entity_type_dict[key_type][:, None]
            )
            valid_type_mask |= (
                self.entity_type_dict[key_type][None, :]
                & self.entity_type_dict[query_type][:, None]
            )
            # get different chain mask
            valid_chain_mask = self.asymid[:, None] != self.asymid[None, :]
            valid_chain_mask &= valid_type_mask

        # get min & max distance threshold
        if min_distance_threshold == -1:  # default false, hard to determine
            # random select a min threshold in [0, max_distance_threshold), but may lead to zero contact pairs
            min_distance_threshold = (
                torch.zeros(1).uniform_(to=max_distance_threshold).item()
            )
        valid_dist_mask = (self.token_distance <= max_distance_threshold) & (
            self.token_distance >= min_distance_threshold
        )

        # make feature
        contact_valid_mask = (
            valid_chain_mask & self.token_resolved_maskmat & valid_dist_mask
        )
        return contact_valid_mask

    def _get_constraint_size(self, group: str, size: Any) -> list[int]:
        """
        If size is not fixed, then we generate it randomly for each group

        Args:
            - group: groups types
        """
        if group == "complex":
            k = 1
        elif group == "interface":
            N_asym = torch.unique(self.asymid).shape[0]
            k = (N_asym * N_asym - N_asym) // 2
        if size < 1 and size > 0:
            samples = torch.zeros(k).geometric_(size).int().tolist()
            return samples
        elif size >= 1 and isinstance(size, int):
            return [size] * k
        else:
            raise NotImplementedError

    def _sample_contacts(
        self,
        contact_valid_mask: torch.Tensor,
        contact_distance_mat: torch.Tensor,
        size: list[int],
        sample_group: str,
        chain_pairs: list[tuple[int, int]],
    ) -> tuple[torch.Tensor, list[tuple[int, int, float, float]]]:
        """
        Randomly select contact from all valid contact pairs

        Args:
            - contact_valid_mask: shape=(N_token, N_token), all valid contact pairs, bool
            - size: how many contacts to sample
            - sample_group: support sample group by 1. whole complex 2. interface
        """
        result_mat = torch.full(
            (contact_valid_mask.shape[0], contact_valid_mask.shape[1], 2),
            fill_value=self.pad_value,
            dtype=torch.float32,
        )
        selected_pairs = []

        if not contact_valid_mask.any().item():
            return result_mat, selected_pairs

        def _sample(valid_mask, cur_size):
            nonlocal selected_pairs

            valid_indices = torch.nonzero(torch.triu(valid_mask))
            selected_indices = valid_indices[
                torch.randperm(valid_indices.shape[0], generator=self.generator)[
                    :cur_size
                ]
            ]

            selected_indices = tuple(zip(*selected_indices))
            if len(selected_indices) == 0:
                return

            # Convert to pairs format
            for i, j in zip(*selected_indices):
                min_dist, max_dist = contact_distance_mat[i, j]
                selected_pairs.append(
                    (i.item(), j.item(), min_dist.item(), max_dist.item())
                )
                selected_pairs.append(
                    (j.item(), i.item(), min_dist.item(), max_dist.item())
                )  # Add symmetric pair

            # add symmetry indices
            selected_indices = (
                selected_indices[0] + selected_indices[1],
                selected_indices[1] + selected_indices[0],
            )
            result_mat[selected_indices] = contact_distance_mat[selected_indices]
            return

        # sample contacts by complex
        if sample_group == "complex":
            _sample(contact_valid_mask, size[0])
        # if group by interface, get all unique interfaces, and sample contact from each interface
        elif sample_group == "interface":
            # sample contacts from each interface iteratively

            idx = 0
            for asym1, asym2 in chain_pairs:
                asym1_mask, asym2_mask = self.asymid == asym1, self.asymid == asym2

                cur_interface_mask = asym1_mask[..., None, :] & asym2_mask[..., :, None]
                cur_interface_mask |= (
                    asym1_mask[..., :, None] & asym2_mask[..., None, :]
                )
                valid_contacts_interface = contact_valid_mask & cur_interface_mask
                _sample(valid_contacts_interface, size[idx])
                idx += 1
        else:
            raise NotImplementedError
        return result_mat, selected_pairs

    def generate(
        self,
        max_distance_range: dict[str, tuple[float, float]],
        sample_group: str,
        feature_type: str,
        size: Any,
        distance_type: str,
        min_distance_threshold: float = 0.0,
        chain_pairs: list[tuple[int, int]] = [],
        **kwargs,
    ) -> tuple[torch.Tensor, list[tuple[int, int, float, float]], set[int]]:
        """
        training & evaluation
        """
        constrained_tokens = set()
        if size == 0:
            return (
                self.encode(
                    torch.full(
                        (self.asymid.shape[0], self.asymid.shape[0], 2),
                        fill_value=self.pad_value,
                        dtype=torch.float32,
                    ),
                    feature_type=feature_type,
                ),
                [],
                constrained_tokens,
            )
        self._get_generation_basics(distance_type=distance_type)

        # contact mask
        n_token = len(self.asymid)
        contact_valid_mask = torch.zeros((n_token, n_token), dtype=torch.bool)
        contact_distance_mat = torch.zeros((n_token, n_token, 2), dtype=torch.float32)

        for contact_type, max_d_range in max_distance_range.items():
            # generate max_distance_mask for different contact type
            max_distance_threshold = torch.zeros(1).uniform_(*max_d_range).item()

            # get all valid contact pairs
            contact_mask = self.get_valid_contact_feature(
                contact_type,
                max_distance_threshold=max_distance_threshold,
                min_distance_threshold=min_distance_threshold,
            )
            contact_valid_mask |= contact_mask
            contact_distance_mat[contact_mask] = torch.tensor(
                [0, max_distance_threshold], dtype=torch.float32
            )

        # random select contact
        size = self._get_constraint_size(sample_group, size)
        sampled_contact_feature, selected_pairs = self._sample_contacts(
            contact_valid_mask,
            contact_distance_mat,
            size,
            sample_group,
            chain_pairs,
        )

        # encode the feature
        contact_feature = self.encode(
            feature=sampled_contact_feature, feature_type=feature_type
        )
        # Track constrained tokens
        constrained_tokens = set()
        for token_i, token_j, _, _ in selected_pairs:
            constrained_tokens.add(token_i)
            constrained_tokens.add(token_j)

        return contact_feature, selected_pairs, constrained_tokens

    def generate_spec_constraint(
        self,
        contact_specifics: list[tuple[int, int, float, float]],
        feature_type: str,
    ) -> tuple[torch.Tensor, set[int]]:
        """
        parse constraint from user specification
        """
        # DEBUG: Check input contact_specifics
        logger.debug(f"ContactFeaturizer.generate_spec_constraint: Number of contact_specifics: {len(contact_specifics)}")
        if len(contact_specifics) > 0:
            logger.debug(f"First contact_specific: {contact_specifics[0]}")

        contact_feature = torch.full(
            (self.asymid.shape[0], self.asymid.shape[0], 2),
            fill_value=self.pad_value,
            dtype=torch.float32,
        )
        constrained_tokens = set()

        for token_list_1, token_list_2, max_distance, min_distance in contact_specifics:
            # Handle both numpy array and list/tuple cases
            token_list_1 = np.asarray(token_list_1)
            token_list_2 = np.asarray(token_list_2)
            
            if len(token_list_1) == 0 or len(token_list_2) == 0:
                logger.debug(f"Empty token list! token_list_1: {len(token_list_1)}, token_list_2: {len(token_list_2)}")
                continue
                
            token_id_1 = int(token_list_1[
                torch.randint(
                    high=len(token_list_1), size=(1,), generator=self.generator
                ).item()
            ])
            token_id_2 = int(token_list_2[
                torch.randint(
                    high=len(token_list_2), size=(1,), generator=self.generator
                ).item()
            ])
            contact_feature[token_id_1, token_id_2, 1] = max_distance
            contact_feature[token_id_2, token_id_1, 1] = max_distance
            contact_feature[token_id_1, token_id_2, 0] = min_distance
            contact_feature[token_id_2, token_id_1, 0] = min_distance

            constrained_tokens.add(token_id_1)
            constrained_tokens.add(token_id_2)

        contact_feature = self.encode(
            feature=contact_feature, feature_type=feature_type
        )
        return contact_feature, constrained_tokens


class PocketFeaturizer(ConstraintFeaturizer):
    def __init__(self, **kargs):
        super().__init__(**kargs)

    def get_valid_pocket_feature(
        self,
        binder_pocket_type: str,
        max_distance_threshold: float,
        asym_list: list[int],
    ) -> torch.Tensor:
        """
        Parameters:
            - binder_pocket_type: PP, LP
            - max_distance_threshold
            - asym list
        Returns:
            - binder_pocket_valid_masks, shape=(N_asym, N_token), A matrix of all pocket tokens within the specified distance range.
            - max_distance_value, shape=(N_asym, N_token), A matrix of the max_distance_threshold to determine the pocket token for each binder chain
        """
        # get valid binder-pocket type, not symmetry
        query_type, key_type = binder_pocket_type
        valid_type_mask = (
            self.entity_type_dict[query_type][:, None]
            & self.entity_type_dict[key_type][None, :]
        )

        # get different chain mask
        diff_chain_mask = self.asymid[:, None] != self.asymid[None, :]

        # get distance mask
        # Note: To simplify the implementation, we only consider token-token distance, instead of any heavy atom distance for each residue for now.
        dist_mask = (
            (self.token_distance <= max_distance_threshold)
            & self.token_resolved_maskmat
            & diff_chain_mask
            & valid_type_mask
        )

        # pocket mask
        n_token = len(self.asymid)
        n_asym = len(asym_list)
        binder_pocket_valid_masks = torch.zeros((n_asym, n_token), dtype=torch.bool)

        for idx, asym_id in enumerate(asym_list):
            cur_chain_dist_mask = (
                self.asymid[:, None].expand(-1, self.asymid.shape[0]) == asym_id
            )
            if cur_chain_dist_mask[:, 0].sum() > 1:  # ensure num of binder tokens > 1
                cur_chain_dist_mask = cur_chain_dist_mask & dist_mask
                cur_valid_mask = cur_chain_dist_mask.any(dim=0)
                binder_pocket_valid_masks[idx] = cur_valid_mask
        return binder_pocket_valid_masks

    def _sample_pocket(
        self,
        binder_pocket_valid_masks: torch.Tensor,
        size: Any,
        asym_list: list[int],
        max_distance_value: torch.Tensor,
        spec_binder_asym_id: Any = None,
        **_,  # do not use
    ) -> torch.Tensor:
        pocket_dist_feature = torch.full(
            (self.asymid.shape[0], self.asymid.shape[0]),
            fill_value=self.pad_value,
            dtype=torch.float32,
        )
        if spec_binder_asym_id is None:
            # random select 1 binder with valid pocket
            binders_with_valid_pocket = torch.nonzero(
                binder_pocket_valid_masks.any(-1), as_tuple=True
            )[0]
            if len(binders_with_valid_pocket) == 0:
                return pocket_dist_feature
            selected_binder = binders_with_valid_pocket[
                torch.randperm(
                    binders_with_valid_pocket.shape[0], generator=self.generator
                )[0]
            ]
        else:
            selected_binder = torch.nonzero(
                asym_list == spec_binder_asym_id, as_tuple=True
            )[0].squeeze()
        selected_all_pocket_res = binder_pocket_valid_masks[selected_binder]
        binder_asym_id = (
            asym_list[selected_binder]
            if spec_binder_asym_id is None
            else spec_binder_asym_id
        )

        # random select k residues within the selected pocket
        selected_pocket_res_mask = torch.zeros(self.asymid.shape[0], dtype=torch.bool)
        valid_pockets = torch.nonzero(selected_all_pocket_res, as_tuple=True)[0]
        selected_pocket_res_indices = torch.randperm(
            valid_pockets.shape[0], generator=self.generator
        )[:size]
        selected_pocket_res_mask[valid_pockets[selected_pocket_res_indices]] = True

        binder_pocket_mask_mat = (self.asymid == binder_asym_id)[
            :, None
        ] * selected_pocket_res_mask[None, :]

        # set distance threshold to selected pocket
        # [binder_mask, pocket_mask]
        binder_dist_mat = max_distance_value[selected_binder][selected_pocket_res_mask][
            None, :
        ].expand((self.asymid == binder_asym_id).sum(), -1)

        pocket_dist_feature[binder_pocket_mask_mat] = binder_dist_mat.reshape(-1)

        return pocket_dist_feature

    def generate(
        self,
        max_distance_range: dict[str, tuple[float, float]],
        feature_type: str,
        size: Any,
        distance_type: Any,
        spec_binder_asym_id: int = None,
        **_,  # do not use
    ) -> tuple[torch.Tensor, set[int]]:
        """
        trainint & evaluation
        """
        constrained_tokens = set()
        if size == 0 or spec_binder_asym_id == -1:
            pocket_dist_feature = self.encode(
                torch.full(
                    (self.asymid.shape[0], self.asymid.shape[0], 1),
                    fill_value=self.pad_value,
                    dtype=torch.float32,
                ),
                feature_type=feature_type,
            )  # [..., N_token, 1]

            return pocket_dist_feature, constrained_tokens
        self._get_generation_basics(distance_type=distance_type)

        # get all binder-pocket pairs & masks
        n_token = len(self.asymid)
        asym_list = torch.unique(self.asymid, sorted=True)
        n_asym = len(asym_list)
        binder_pocket_valid_masks = torch.zeros((n_asym, n_token), dtype=torch.bool)
        max_distance_value = torch.zeros((n_asym, n_token), dtype=torch.float32)
        for binder_pocket_type, max_d_range in max_distance_range.items():
            # generate max_distance_mask for different binder_pocket type
            max_distance_threshold = torch.zeros(1).uniform_(*max_d_range).item()

            # get all valid binder-pocket pairs
            valid_pocket_token_mask = self.get_valid_pocket_feature(
                binder_pocket_type,
                max_distance_threshold=max_distance_threshold,
                asym_list=asym_list,
            )
            binder_pocket_valid_masks |= valid_pocket_token_mask
            max_distance_value[valid_pocket_token_mask] = max_distance_threshold

        # random select k residues from pocket, only consider one binder
        size = self._get_constraint_size(size)
        sampled_pocket_feature = self._sample_pocket(
            binder_pocket_valid_masks,
            size,
            asym_list,
            max_distance_value,
            spec_binder_asym_id,
        ).unsqueeze(-1)

        # encode the feature
        pocket_dist_feature = self.encode(
            feature=sampled_pocket_feature, feature_type=feature_type
        )
        # Track constrained tokens
        nonzero_indices = torch.nonzero(pocket_dist_feature)
        for i, j, _ in nonzero_indices:
            constrained_tokens.add(i.item())
            constrained_tokens.add(j.item())

        return pocket_dist_feature, constrained_tokens

    def generate_spec_constraint(
        self, pocket_specifics, feature_type: str
    ) -> tuple[torch.Tensor, set[int]]:
        """
        parse constraint from user specification
        """
        pocket_dist_mat = torch.full(
            (self.asymid.shape[0], self.asymid.shape[0], 1),
            fill_value=self.pad_value,
            dtype=torch.float32,
        )
        logger.info(
            "[pocket_dist_mat] BEFORE fill: shape=%s, pad_value=%s, nnz=%d",
            tuple(pocket_dist_mat.shape),
            self.pad_value,
            (pocket_dist_mat != self.pad_value).sum().item(),
        )

        for binder_token_list, pocket_token_list, max_distance in pocket_specifics:
            pocket_token_id = pocket_token_list[
                torch.randint(
                    high=pocket_token_list.shape[0], size=(1,), generator=self.generator
                ).item()
            ]

            binder_token_idx = torch.tensor(binder_token_list)[:, None]
            new_val = torch.full(
                (binder_token_idx.shape[0],),
                max_distance,
                dtype=pocket_dist_mat.dtype,
                device=pocket_dist_mat.device,
            )
            current = pocket_dist_mat[binder_token_idx.squeeze(-1), pocket_token_id, 0]
            # If (i,j) already set by another pocket, take min (more restrictive)
            pocket_dist_mat[binder_token_idx.squeeze(-1), pocket_token_id, 0] = (
                torch.where(
                    current == self.pad_value,
                    new_val,
                    torch.minimum(current, new_val),
                )
            )

        logger.info(
            "[pocket_dist_mat] AFTER fill: shape=%s, nnz=%d, min=%s, max=%s",
            tuple(pocket_dist_mat.shape),
            (pocket_dist_mat != self.pad_value).sum().item(),
            pocket_dist_mat[pocket_dist_mat != self.pad_value].min().item()
            if (pocket_dist_mat != self.pad_value).any()
            else None,
            pocket_dist_mat[pocket_dist_mat != self.pad_value].max().item()
            if (pocket_dist_mat != self.pad_value).any()
            else None,
        )

        pocket_dist_feature = self.encode(
            feature=pocket_dist_mat, feature_type=feature_type
        )

        constrained_tokens = set()
        nonzero_indices = torch.nonzero(pocket_dist_feature)
        for binder_token_id, pocket_token_id, _ in nonzero_indices:
            # Use .item() so the set dedupes by token index (scalar tensors hash by identity → inflated len).
            constrained_tokens.add(int(binder_token_id.item()))
            constrained_tokens.add(int(pocket_token_id.item()))
        return pocket_dist_feature, constrained_tokens

    def _get_constraint_size(self, size) -> list[int]:
        """
        If size is not fixed, then we generate it randomly for each group
        """
        if size < 1 and size > 0:
            # TODO: to be determined!
            samples = torch.zeros(1).geometric_(size).int().tolist()[0]
            return samples
        elif size >= 1 and isinstance(size, int):
            return size
        else:
            raise NotImplementedError


class ContactAtomFeaturizer(ContactFeaturizer):
    def __init__(
        self,
        token_array: TokenArray,
        atom_array: AtomArray,
        pad_value: float = 0,
        generator=None,
    ):
        self.token_array = token_array
        self.atom_array = atom_array
        self.pad_value = pad_value
        self.generator = generator
        self._get_base_info()

    def _get_base_info(self):
        self.asymid = torch.tensor(self.atom_array.asym_id_int, dtype=torch.long)
        self.is_ligand = torch.tensor(self.atom_array.is_ligand, dtype=torch.bool)
        self.is_protein = torch.tensor(self.atom_array.is_protein, dtype=torch.bool)
        self.entity_type_dict = {"P": self.is_protein, "L": self.is_ligand}

    def _get_generation_basics(self, distance_type="atom"):
        # is_resolved mask
        self.token_resolved_mask = torch.tensor(
            self.atom_array.is_resolved, dtype=torch.bool
        )
        self.token_resolved_maskmat = (
            self.token_resolved_mask[:, None] * self.token_resolved_mask[None, :]
        )

        # distance matrix
        self.token_distance = torch.tensor(
            cdist(self.atom_array.coord, self.atom_array.coord), dtype=torch.float64
        )

    def generate_spec_constraint(
        self,
        contact_specifics: list[tuple[int, int, float, float]],
        feature_type: str,
        shape: tuple[int, int, int],
    ) -> tuple[torch.Tensor, set[int]]:
        """
        parse constraint from user specification
        """

        contact_feature = torch.full(
            shape, fill_value=self.pad_value, dtype=torch.float32
        )
        for token_id_1, token_id_2, max_distance, min_distance in contact_specifics:
            contact_feature[token_id_1, token_id_2, 1] = max_distance
            contact_feature[token_id_2, token_id_1, 1] = max_distance
            contact_feature[token_id_1, token_id_2, 0] = min_distance
            contact_feature[token_id_2, token_id_1, 0] = min_distance

        contact_feature = self.encode(
            feature=contact_feature, feature_type=feature_type
        )
        constrained_tokens = set()
        for token_id_1, token_id_2, _, _ in contact_specifics:
            constrained_tokens.add(token_id_1)
            constrained_tokens.add(token_id_2)
        return contact_feature, constrained_tokens

    def get_real_distance(self, atom_idx_1: int, atom_idx_2: int) -> float:
        return self.token_distance[atom_idx_1, atom_idx_2]


class SubStructureFeaturizer(ConstraintFeaturizer):
    def __init__(self, **kargs):
        super().__init__(**kargs)
        # Default distance bins
        self.distance_bins = torch.tensor([0, 4, 8, 16, torch.inf])

    def _add_coordinate_noise(self, coord_noise_scale: float = 0.05) -> torch.Tensor:
        """Add Gaussian noise to coordinates"""
        # Convert coordinates to tensor first
        coords = torch.tensor(self.atom_array.coord)
        if coord_noise_scale <= 0:
            return coords
        noisy_coords = (
            coords
            + torch.randn(
                coords.shape,
                generator=self.generator,
            )
            * coord_noise_scale
        )

        return noisy_coords

    def _get_distance_feature(
        self,
        selected_token_mask: torch.Tensor,
        coord_noise_scale: float = 0.05,
        feature_type: str = "one_hot",
    ) -> torch.Tensor:
        """
        Encode pairwise distances between selected tokens into binned one-hot features

        Parameters:
            - selected_token_mask: Boolean mask of selected tokens
            - coord_noise_scale: Scale of Gaussian noise to add to coordinates
            - feature_type: "one_hot" or "continuous"

        Returns:
            - distance_feature: For one_hot: [..., N_token, N_token] tensor with bin indices
                              For continuous: [..., N_token, N_token] tensor with distance values
        """
        n_tokens = len(self.asymid)

        # Initialize output tensor
        distance_feature = torch.full(
            (n_tokens, n_tokens),
            fill_value=-1 if feature_type == "one_hot" else self.pad_value,
            dtype=torch.long if feature_type == "one_hot" else torch.float32,
        )

        # Get selected token indices
        selected_tokens = torch.nonzero(selected_token_mask).squeeze(-1)
        if len(selected_tokens) <= 1:
            return distance_feature

        # Add noise to coordinates and calculate distances
        noisy_coords = self._add_coordinate_noise(coord_noise_scale)

        # Get token center atom indices
        token_centre_atom_indices = torch.tensor(
            self.token_array.get_annotation("centre_atom_index"), dtype=torch.long
        )
        selected_indices = token_centre_atom_indices[selected_tokens.long()]
        selected_coords = noisy_coords[selected_indices]

        # Calculate pairwise distances between selected tokens
        pairwise_distances = torch.cdist(selected_coords, selected_coords)

        if feature_type == "one_hot":
            # Digitize distances into bins
            binned_distances = (
                torch.bucketize(pairwise_distances, self.distance_bins, right=True) - 1
            )

            # Create mask for valid bins
            valid_bins = binned_distances > 0

            # Create indices for the full matrix
            rows = selected_tokens.repeat_interleave(len(selected_tokens))
            cols = selected_tokens.repeat(len(selected_tokens))

            # Get valid indices and their corresponding bin values
            valid_mask = valid_bins.flatten()
            valid_rows = rows[valid_mask]
            valid_cols = cols[valid_mask]
            valid_bins = binned_distances.flatten()[valid_mask]

            # Fill the distance feature matrix
            distance_feature[valid_rows, valid_cols] = valid_bins
            distance_feature[valid_cols, valid_rows] = valid_bins  # symmetric

        else:  # continuous
            # Create indices for the full matrix
            rows = selected_tokens.repeat_interleave(len(selected_tokens))
            cols = selected_tokens.repeat(len(selected_tokens))

            # Fill the distance feature matrix
            distance_feature[rows, cols] = pairwise_distances.flatten()
            distance_feature[cols, rows] = pairwise_distances.flatten()  # symmetric

        return distance_feature

    def get_valid_substructure_feature(
        self, mol_type_pairs: dict[str, float]
    ) -> torch.Tensor:
        """
        Find valid chains that form interfaces based on mol_type_pairs

        Parameters:
            - mol_type_pairs: Dict of type pairs (e.g. {'PP': threshold, 'LP': threshold})
                First type in pair is the one to be selected as substructure, threshold is the distance threshold to determine the interface
        Returns:
            - valid_asym_ids: List of valid asym_ids that can form interfaces
        """
        valid_asym_ids = []

        for type_pair, threshold in mol_type_pairs.items():
            # Parse type pair (e.g. 'PP' -> ('P','P'))
            query_type, key_type = type_pair

            # Get type masks
            valid_type_mask = (
                self.entity_type_dict[query_type][:, None]
                & self.entity_type_dict[key_type][None, :]
            )

            # Get different chain mask
            diff_chain_mask = self.asymid[:, None] != self.asymid[None, :]

            # Get distance mask
            dist_mask = (
                (self.token_distance <= threshold)
                & self.token_resolved_maskmat
                & diff_chain_mask
                & valid_type_mask
            )

            # Find chains that form interfaces
            for asym_id in torch.unique(self.asymid):
                # Only consider chains of query_type
                if not self.entity_type_dict[query_type][self.asymid == asym_id].any():
                    continue

                # Check if this chain forms interface with any chain of key_type
                cur_chain_mask = self.asymid == asym_id
                other_chains_mask = ~cur_chain_mask

                has_interface = (dist_mask[cur_chain_mask][:, other_chains_mask]).any()

                if has_interface:
                    valid_asym_ids.append(asym_id)

        return torch.tensor(valid_asym_ids)

    def _get_constraint_size(self, size: Any) -> int:
        """
        If size is not fixed, then we generate it randomly

        Args:
            - size: If >=1, randomly select chains; if <1, select size chains
        Returns:
            - size: Number of chains/proportion of tokens to select
        """
        if size < 1 and size > 0:
            # For size < 1, use as proportion directly
            return torch.rand(1).geometric_(size, generator=self.generator).int().item()
        elif size >= 1 and isinstance(size, (int, float)):
            # For size >= 1, return the size directly
            return int(size)
        else:
            raise NotImplementedError(f"Invalid size: {size}")

    def _sample_substructure(
        self,
        valid_asym_ids: torch.Tensor,
        size: Any,
        ratios: dict[str, list[float]],
        spec_asym_id: Any = None,
    ) -> torch.Tensor:
        """
        Sample substructure based on size and ratios

        Parameters:
            - valid_asym_ids: List of valid asym_ids
            - size: Total number of chains to select
            - ratios: Dict containing:
                - full: List of possible proportions for full chain selection
                - partial: Proportion of tokens to select for partial chains [0,1]
            - spec_asym_id: If provided, select from this specific chain
        """
        selected_token_mask = torch.zeros(len(self.asymid), dtype=torch.bool)

        if len(valid_asym_ids) == 0:
            return selected_token_mask

        # Handle spec_asym_id case
        if spec_asym_id is not None:
            if spec_asym_id not in valid_asym_ids:
                return selected_token_mask
            # Use partial_ratio for spec_asym_id
            chain_mask = (self.asymid == spec_asym_id) & self.token_resolved_mask
            chain_tokens = torch.nonzero(chain_mask).squeeze()
            if len(chain_tokens) == 0:  # Skip if no resolved tokens
                return selected_token_mask
            num_tokens = max(1, int(len(chain_tokens) * ratios["partial"]))
            selected_tokens = chain_tokens[
                torch.randperm(len(chain_tokens), generator=self.generator)[:num_tokens]
            ]
            selected_token_mask[selected_tokens] = True
            return selected_token_mask

        # Regular case: sample based on size and ratios
        if size == 0:
            return selected_token_mask

        # Randomly select full chain ratio from the list
        full_ratio_idx = torch.randint(
            len(ratios["full"]), (1,), generator=self.generator
        ).item()
        full_ratio = ratios["full"][full_ratio_idx]

        # Calculate number of chains for full and partial selection
        num_full_chains = min(int(size * full_ratio), len(valid_asym_ids))
        num_partial_chains = min(
            size - num_full_chains, len(valid_asym_ids) - num_full_chains
        )

        # Randomly shuffle and split valid_asym_ids
        shuffled_indices = torch.randperm(len(valid_asym_ids), generator=self.generator)
        full_chain_ids = valid_asym_ids[shuffled_indices[:num_full_chains]]
        partial_chain_ids = valid_asym_ids[
            shuffled_indices[num_full_chains : num_full_chains + num_partial_chains]
        ]

        # Select full chains
        for asym_id in full_chain_ids:
            chain_mask = (self.asymid == asym_id) & self.token_resolved_mask
            selected_token_mask |= chain_mask
        # Select partial chains
        for asym_id in partial_chain_ids:
            chain_mask = (self.asymid == asym_id) & self.token_resolved_mask
            chain_tokens = torch.nonzero(chain_mask).squeeze(-1)
            if len(chain_tokens) == 0:  # Skip if no resolved tokens
                continue
            num_tokens = max(
                1,
                int(len(chain_tokens) * torch.rand(1, generator=self.generator).item()),
            )
            selected_tokens = chain_tokens[
                torch.randperm(len(chain_tokens), generator=self.generator)[:num_tokens]
            ]
            selected_token_mask[selected_tokens] = True

        return selected_token_mask

    def generate(
        self,
        mol_type_pairs: dict[str, float],
        feature_type: str,
        size: Any,
        ratios: dict[str, list[float]],
        coord_noise_scale: float,
        spec_asym_id: int = None,
    ) -> tuple[torch.Tensor, set[int]]:
        """
        Generate substructure features

        Parameters:
            - mol_type_pairs: Dict of type pairs and their distance thresholds
            - feature_type: Type of feature encoding
            - size: Number of chains to select
            - ratios: Dict containing:
                - full: List of possible proportions for full chain selection
                - partial: Proportion of tokens to select for partial chains [0,1]
            - coord_noise_scale: Scale of Gaussian noise to add to coordinates
            - spec_asym_id: Specific chain to select from
        """
        constrained_tokens = set()
        if size == 0 or spec_asym_id == -1:
            distance_feature = torch.full(
                (self.asymid.shape[0], self.asymid.shape[0]),
                fill_value=-1 if feature_type == "one_hot" else self.pad_value,
                dtype=torch.long if feature_type == "one_hot" else torch.float32,
            )
            return (
                self.encode(
                    feature=distance_feature,
                    feature_type=feature_type,
                    num_classes=len(self.distance_bins) - 1,
                ),
                constrained_tokens,
            )

        self._get_generation_basics()

        # Get valid asym_ids that form interfaces
        valid_asym_ids = self.get_valid_substructure_feature(mol_type_pairs)

        if len(valid_asym_ids) == 0:
            distance_feature = torch.full(
                (self.asymid.shape[0], self.asymid.shape[0]),
                fill_value=-1 if feature_type == "one_hot" else self.pad_value,
                dtype=torch.long if feature_type == "one_hot" else torch.float32,
            )
            return (
                self.encode(
                    feature=distance_feature,
                    feature_type=feature_type,
                    num_classes=len(self.distance_bins) - 1,
                ),
                constrained_tokens,
            )

        size = self._get_constraint_size(size)

        # Sample tokens based on ratio and spec_asym_id
        selected_token_mask = self._sample_substructure(
            valid_asym_ids, size, ratios, spec_asym_id
        )

        # Get distance features (bin indices)
        distance_feature = self._get_distance_feature(
            selected_token_mask, coord_noise_scale, feature_type
        )

        # Track constrained tokens
        constrained_tokens = set(torch.nonzero(selected_token_mask).flatten().tolist())

        # Encode using base class method
        return (
            self.encode(
                feature=distance_feature,
                feature_type=feature_type,
                num_classes=len(self.distance_bins) - 1,
            ),
            constrained_tokens,
        )

    def analyze_features(self, feature_tensor: torch.Tensor) -> dict[str, Any]:
        """
        Analyze the features generated by the generate method
        """
        is_one_hot = len(feature_tensor.shape) == 3
        n_tokens = feature_tensor.shape[0]
        if is_one_hot:
            # For one-hot features
            # A token is active if it has any non-zero one-hot vector
            has_valid_distance = torch.any(feature_tensor, dim=-1)
            active_tokens = torch.any(has_valid_distance, dim=1)

            # Get distribution of distance bins (excluding zero vectors)
            valid_distances = feature_tensor[has_valid_distance]
            bin_counts = torch.sum(valid_distances, dim=0)  # Sum over all valid pairs
            distance_stats = {
                f"bin_{i}_count": count.item() for i, count in enumerate(bin_counts)
            }

        else:
            # For continuous features
            # A token is active if it has any non-zero distance to other tokens
            has_valid_distance = feature_tensor != 0
            active_tokens = torch.any(has_valid_distance, dim=1)

            # Get distribution of actual distances (excluding zeros)
            valid_distances = feature_tensor[has_valid_distance]
            if len(valid_distances) > 0:
                distance_stats = {
                    "mean": valid_distances.mean().item(),
                    "std": valid_distances.std().item(),
                    "min": valid_distances.min().item(),
                    "max": valid_distances.max().item(),
                }
            else:
                distance_stats = {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}

        stats = {
            "num_active_tokens": active_tokens.sum().item(),
            "active_token_ratio": (active_tokens.sum().item() / n_tokens),
            "distance_distribution": distance_stats,
        }

        return stats

    def generate_spec_constraint(self, substructure_specifics, feature_type):
        """parse constraint from user specification

        Args:
            substructure_specifics (Dict): dictionary speicifing fixed tokens
                token_indices: List[int]
                token_coords: List[List[float]]
            feature_type: 'ont hot' by default
        """
        distance_feature_mat = torch.full(
            (self.asymid.shape[0], self.asymid.shape[0]),
            fill_value=-1 if feature_type == "one_hot" else self.pad_value,
            dtype=torch.long if feature_type == "one_hot" else torch.float32,
        )

        if len(substructure_specifics["token_indices"]) > 0:
            token_indices = torch.tensor(substructure_specifics["token_indices"])
            coords = torch.tensor(substructure_specifics["token_coords"], dtype=torch.float32)

            distance_mat = torch.cdist(coords, coords)
            if feature_type == "one_hot":
                # The training path (_get_distance_feature) bucketizes before one-hot;
                # this path did not, so raw distances were written into a long tensor
                # (truncated to whole Angstroms) and then fed to F.one_hot(num_classes=4),
                # which is out of range for anything >= 4 A. Bin here to match training.
                distance_mat = (
                    torch.bucketize(distance_mat, self.distance_bins, right=True) - 1
                ).clamp_(min=0, max=len(self.distance_bins) - 2)
            distance_feature_mat[token_indices[:, None], token_indices[None, :]] = (
                distance_mat
            )

        distance_feature_mat = self.encode(
            distance_feature_mat, feature_type, num_classes=len(self.distance_bins) - 1
        )

        return distance_feature_mat
