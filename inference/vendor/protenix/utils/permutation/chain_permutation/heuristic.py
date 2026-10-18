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

import os
import random

import torch

from protenix.metrics.rmsd import rmsd, self_aligned_rmsd
from protenix.utils.logger import get_logger
from protenix.utils.permutation.chain_permutation.utils import (
    apply_transform,
    get_optimal_transform,
    num_unique_matches,
)
from protenix.utils.permutation.utils import Checker

logger = get_logger(__name__)

ExtraLabelKeys = [
    "pocket_mask",
    "interested_ligand_mask",
    "chain_1_mask",
    "chain_2_mask",
    "entity_mol_id",
    "mol_id",
    "mol_atom_index",
    "pae_rep_atom_mask",
]


def correct_symmetric_chains(
    pred_dict: dict,
    label_full_dict: dict,
    extra_label_keys: list[str] = ExtraLabelKeys,
    max_num_chains: int = 20,
    permute_label: bool = True,
    **kwargs,
):
    """Inputs

    Args:
        pred_dict (dict[str, torch.Tensor]): A dictionary containing:
            - coordinate: pred_dict["coordinate"]
                shape = [N_cropped_atom, 3] or [Batch, N_cropped_atom, 3].
            - other keys: entity_mol_id, mol_id, mol_atom_index, pae_rep_atom_mask, is_ligand.
                shape = [N_cropped_atom]
        label_full_dict (dict[str, torch.Tensor]): A dictionary containing
            - coordinate: label_full_dict["coordinate"] and label_full_dict["coordinate_mask"]
                shape = [N_atom, 3] and [N_atom] (for coordinate_mask)
            - other keys: entity_mol_id, mol_id, mol_atom_index, pae_rep_atom_mask.
                shape = [N_atom]
            - extra keys: keys specified by extra_feature_keys.
        extra_label_keys (list[str]):
            - Additional features in label_full_dict that should be returned along with the permuted coordinates.
        max_num_chains (int): if the number of chains is more than this number, than skip permutation to
            avoid expensive computations.
        permute_label (bool): if true, permute the groundtruth chains, otherwise premute the prediction chains

    Return:
        output_dict:
            If permute_label=True, this is a dictionary containing
            - coordinate
            - coordinate_mask
            - features specified by extra_label_keys.
            If permute_label=False, this is a dictionary containing
            - coordinate.

        log_dict: statistics.

        permute_pred_indices / permute_label_indices:
            If batch_mode, this is a list of LongTensor. Otherwise, this is a LongTensor.
            The LongTensor gives the indices to permute either prediction or label.
    """

    assert pred_dict["coordinate"].dim() in [2, 3]
    batch_mode = pred_dict["coordinate"].dim() > 2

    if not batch_mode:
        (
            best_match,
            permute_pred_indices,
            permute_label_indices,
            output_dict,
            log_dict,
        ) = _correct_symmetric_chains_for_one_sample(
            pred_dict,
            label_full_dict,
            max_num_chains,
            permute_label,
            extra_label_keys=extra_label_keys,
            **kwargs,
        )
        return output_dict, log_dict, permute_pred_indices, permute_label_indices
    else:
        assert not permute_label, "Only supports prediction permutations in batch mode."
        pred_coord = []
        log_dict = {}
        best_matches = []
        permute_pred_indices = []
        permute_label_indices = []
        # Loop over all samples to find best matches one by one
        for i, pred_coord_i in enumerate(pred_dict["coordinate"]):
            (
                best_match_i,
                permute_pred_indices_i,
                permute_label_indices_i,
                pred_dict_i,
                log_dict_i,
            ) = _correct_symmetric_chains_for_one_sample(
                {**pred_dict, "coordinate": pred_coord_i},
                label_full_dict,
                max_num_chains,
                permute_label=False,
                extra_label_keys=[],
                **kwargs,
            )

            best_matches.append(best_match_i)
            permute_pred_indices.append(permute_pred_indices_i)
            permute_label_indices.append(permute_label_indices_i)
            pred_coord.append(pred_dict_i["coordinate"])
            for key, value in log_dict_i.items():
                log_dict.setdefault(key, []).append(value)

        output_dict = {"coordinate": torch.stack(pred_coord, dim=0)}

        log_dict = {key: sum(value) / len(value) for key, value in log_dict.items()}
        log_dict["N_unique_perm"] = num_unique_matches(best_matches)

        return output_dict, log_dict, permute_pred_indices, permute_label_indices


def _correct_symmetric_chains_for_one_sample(
    pred_dict: dict,
    label_full_dict: dict,
    max_num_chains: int = 20,
    permute_label: bool = False,
    extra_label_keys: list[str] = [],
    **kwargs,
):
    """
    Correct symmetric chains for a single sample by permuting either the predicted or the ground truth coordinates.
    """

    if not permute_label:
        """
        Permutation will act on the predicted coordinate.
        In this case, predicted structures and true structure need to have
        the same number of atoms.
        """
        assert pred_dict["coordinate"].size(-2) == label_full_dict["coordinate"].size(
            -2
        )

    with torch.no_grad():
        # Do not compute gradient while optimizing the permutation
        (
            best_match,
            permute_pred_indices,
            permute_label_indices,
            log_dict,
        ) = MultiChainPermutation(**kwargs)(
            pred_dict=pred_dict,
            label_full_dict=label_full_dict,
            max_num_chains=max_num_chains,
        )

    if permute_label:
        # Permute groundtruth coord and coord mask with indices, along the first dimension.
        indices = permute_label_indices.tolist()
        output_dict = {
            "coordinate": label_full_dict["coordinate"][indices, :],
            "coordinate_mask": label_full_dict["coordinate_mask"][indices],
        }
        # Permute extra label features, along the last dimension.
        output_dict.update(
            {
                k: label_full_dict[k][..., indices]
                for k in extra_label_keys
                if k in label_full_dict
            }
        )

    else:
        # Permute the predicted coord with permuted_indices
        indices = permute_pred_indices.tolist()
        output_dict = {
            "coordinate": pred_dict["coordinate"][indices, :],
        }

    return (
        best_match,
        permute_pred_indices,
        permute_label_indices,
        output_dict,
        log_dict,
    )


class MultiChainPermutation(object):
    """Anchor-based heuristic method.
    Find the best match that maps predicted chains to chains in the true complex.
    Here the predicted chains could be cropped, which could be fewer and shorter than
    those in the true complex.
    """

    def __init__(
        self, use_center_rmsd, find_gt_anchor_first, accept_it_as_it_is, *args, **kwargs
    ):
        self.use_center_rmsd = use_center_rmsd
        self.find_gt_anchor_first = find_gt_anchor_first
        self.accept_it_as_it_is = accept_it_as_it_is

    @staticmethod
    def _build_fibril_chain_table(pred_dict: dict):
        """{atom-level mol_id: (pf_id, layer_id)} for chains that have a protofilament, else None.

        Split out from process_input so it can be tested directly: pf_id/layer_id are TOKEN-level
        while mol_id is ATOM-level, so they must be expanded through atom_to_token_idx. Getting
        that wrong would raise inside the caller's try/except and silently degrade to no
        permutation at all, which is the kind of failure that looks like success in a log.
        """
        try:
            if not all(k in pred_dict for k in ("pf_id", "layer_id", "atom_to_token_idx")):
                return None
            t2a = pred_dict["atom_to_token_idx"].reshape(-1).long()
            pf_a = pred_dict["pf_id"].reshape(-1)[t2a]
            lay_a = pred_dict["layer_id"].reshape(-1)[t2a]
            mol_a = pred_dict["mol_id"].reshape(-1).long()
            if not (pf_a.numel() == lay_a.numel() == mol_a.numel()):
                return None
            tbl = {}
            for a in torch.unique(mol_a).tolist():
                m = mol_a == a
                p = int(pf_a[m][0])
                if p > 0:
                    tbl[a] = (p, int(lay_a[m][0]))
            return tbl or None
        except Exception:
            return None

    @staticmethod
    def dict_of_interested_keys(
        input_dict: dict,
        keys: list = [
            "mol_id",
            "entity_mol_id",
            "mol_atom_index",
            "pae_rep_atom_mask",
            "coordinate",
            "coordinate_mask",
            "is_ligand",
        ],
    ):
        """
        Extract a subset of keys from the input dictionary from the list `keys`.
        """

        return {k: input_dict[k] for k in keys if k in input_dict}

    def process_input(
        self,
        pred_dict: dict[str, torch.Tensor],
        label_full_dict: dict[str, torch.Tensor],
        max_num_chains: int = 20,
    ):
        """Process the input dicts

        Args:
            pred_dict (dict[str, torch.Tensor]): A dictionary containing
                entity_mol_id, mol_id, mol_atom_index, pae_rep_atom_mask, coordinate, is_ligand.
                All Tensors have shape = [N_cropped_atom]
            label_full_dict (dict[str, torch.Tensor]): A dictionary containing
                entity_mol_id, mol_id, mol_atom_index, pae_rep_atom_mask, coordinate, coordinate_mask.
                All Tensors have shape = [N_atom]
            max_num_chains (int): if the number of chains is more than this number, than skip permutation to
                avoid expensive computations.
            permute_label (bool): if true, permute the groundtruth chains, otherwise premute the prediction chains
        """

        log_dict = {}

        # Chain-level (pf_id, layer_id), used only by compute_best_match_fibril. pf_id/layer_id are
        # TOKEN-level while mol_id is ATOM-level, so they must be expanded through
        # atom_to_token_idx -- putting them straight into the atom-level dicts would raise a shape
        # error that the caller's try/except would swallow into a silently skipped permutation.
        self._fibril_chain = self._build_fibril_chain_table(pred_dict)

        for key in ["entity_mol_id", "mol_id", "mol_atom_index"]:
            pred_dict[key] = pred_dict[key].long()
            label_full_dict[key] = label_full_dict[key].long()

        # get original unpermuted match
        pred_mol_id = set(torch.unique(pred_dict["mol_id"]).tolist())
        label_mol_id = set(torch.unique(label_full_dict["mol_id"]).tolist())
        if pred_mol_id.intersection(label_mol_id) != pred_mol_id:
            # if the mol_id in predicted structure is not a subset of label structure,
            # assert they contain the same number of atoms.
            assert pred_dict["coordinate"].size(-2) == label_full_dict[
                "coordinate"
            ].size(-2)
            self.unpermuted_match = self.check_pattern_and_create_mapping(
                pred_dict["mol_id"], label_full_dict["mol_id"]
            )
        else:
            self.unpermuted_match = {
                i: i for i in torch.unique(pred_dict["mol_id"]).tolist()
            }

        if len(torch.unique(label_full_dict["entity_mol_id"])) == len(
            torch.unique(label_full_dict["mol_id"])
        ):
            # No permutation is needed
            has_sym_chain = False
            return self.unpermuted_match, has_sym_chain
        else:
            has_sym_chain = True

        n_label_chain = len(torch.unique(label_full_dict["mol_id"]))
        if n_label_chain > 20:
            logger.warning(f"The label_full_dict contains {n_label_chain} asym chains.")

        if max_num_chains > 0 and n_label_chain > max_num_chains:
            logger.warning(
                f"The label_full_dict contains {n_label_chain} asym chains (max_num_chains: {max_num_chains}). Will skip chain permutation and keep the original chain assignment."
            )
            return self.unpermuted_match, has_sym_chain

        # parse features to token-level
        self.label_token_dict, self.label_asym_dict = self._parse_atom_feature_dict(
            self.dict_of_interested_keys(label_full_dict),
            rep_atom_mask=label_full_dict["pae_rep_atom_mask"],
        )
        self.pred_token_dict, self.pred_asym_dict = self._parse_atom_feature_dict(
            self.dict_of_interested_keys(pred_dict),
            rep_atom_mask=pred_dict["pae_rep_atom_mask"],
        )

        # get mapping between entity_id and asym_id
        self.label_token_dict.update(
            self._get_entity_asym_mapping(
                self.label_token_dict["entity_mol_id"], self.label_token_dict["mol_id"]
            )
        )
        self.pred_token_dict.update(
            self._get_entity_asym_mapping(
                self.pred_token_dict["entity_mol_id"], self.pred_token_dict["mol_id"]
            )
        )
        return None, has_sym_chain

    @staticmethod
    def check_pattern_and_create_mapping(mol_id1: torch.Tensor, mol_id2: torch.Tensor):
        """
        Check if the patterns between two mol_id tensors match and create a mapping between them.

        Args:
            mol_id1 (torch.Tensor): A tensor of mol IDs from the first set.
            mol_id2 (torch.Tensor): A tensor of mol IDs from the second set.

        Returns:
            dict: A dictionary mapping mol IDs from mol_id1 to mol_id2.
        """
        if mol_id1.shape != mol_id2.shape:
            raise ValueError("mol_id1 and mol_id2 must have the same shape")

        pattern_mapping = {}
        for id1, id2 in zip(mol_id1.tolist(), mol_id2.tolist()):
            if id1 in pattern_mapping:
                if pattern_mapping[id1] != id2:
                    raise ValueError(
                        f"Inconsistent pattern: {id1} mapped to different values in mol_id2"
                    )
            else:
                if id2 in pattern_mapping.values():
                    raise ValueError(
                        f"Value {id2} in mol_id2 already mapped to another value"
                    )
                pattern_mapping[id1] = id2

        return pattern_mapping

    def _parse_atom_feature_dict(
        self, atom_features: dict, rep_atom_mask: torch.Tensor
    ):
        """
        Parse the atom feature dictionary and convert it to token features and per-asym token features.

        Args:
            atom_features (dict): A dictionary containing atom features.
            rep_atom_mask (torch.Tensor): The rep atom mask.

        Returns:
            tuple: A tuple containing:
                - token_dict (dict): A dictionary containing the token features corresponding to the rep atoms.
                - asym_token_dict (dict): A dictionary where keys are asym IDs and values are dictionaries of features corresponding to each asym ID.
        """

        # Atom features --> Token features
        token_dict = self._convert_to_token_dict(
            atom_dict=atom_features,
            rep_atom_mask=rep_atom_mask.bool(),
        )

        # Token features --> per asym token features
        asym_token_dict = self._convert_to_per_asym_feature_dict(
            asym_id=token_dict["mol_id"],
            feature_dict={
                "coordinate": token_dict.get("coordinate"),
                "coordinate_mask": token_dict.get("coordinate_mask", None),
                "mol_atom_index": token_dict.get("mol_atom_index"),
            },
        )

        return token_dict, asym_token_dict

    @staticmethod
    def _convert_to_token_dict(
        atom_dict: dict[str, torch.Tensor], rep_atom_mask: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """
        Convert the atom feature dictionary to a token feature dictionary based on the rep atom mask.

        Args:
            atom_dict (dict[str, torch.Tensor]): A dictionary containing atom features.
            rep_atom_mask (torch.Tensor): The rep atom mask.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing the token features corresponding to the rep atoms.
        """

        rep_atom_mask = rep_atom_mask.bool()
        return {k: v[rep_atom_mask] for k, v in atom_dict.items() if v is not None}

    @staticmethod
    def _convert_to_per_asym_feature_dict(asym_id: torch.Tensor, feature_dict: dict):
        """
        Convert the feature dictionary to a dictionary where keys are asym IDs and values are dictionaries of features corresponding to each asym ID.

        Args:
            asym_id (torch.Tensor): A tensor of asym IDs.
            feature_dict (dict): A dictionary containing features for all atoms.

        Returns:
            dict: A dictionary where keys are asym IDs and values are dictionaries of features corresponding to each asym.
        """
        out = {}

        for aid in torch.unique(asym_id):
            mask = asym_id == aid
            out[aid.item()] = {
                k: v[mask] for k, v in feature_dict.items() if v is not None
            }
        return out

    @staticmethod
    def _get_entity_asym_mapping(
        entity_id: torch.Tensor, asym_id: torch.Tensor
    ) -> tuple[dict]:
        """
        Generate mappings between entity IDs and asym IDs.

        Args:
            entity_id (torch.Tensor): A tensor of entity IDs.
            asym_id (torch.Tensor): A tensor of asym IDs.

        Returns:
            tuple[dict]: A tuple containing two dictionaries:
                - entity_to_asym: A dictionary mapping entity IDs to their corresponding asym IDs.
                - asym_to_entity: A dictionary mapping asym IDs to their corresponding entity IDs.
        """

        entity_to_asym = {}
        asym_to_entity = {}
        for ein in torch.unique(entity_id):
            ein = ein.item()
            asyms = torch.unique(asym_id[entity_id == ein])
            entity_to_asym[ein] = asyms
            asym_to_entity.update({a.item(): ein for a in asyms})

        return {"entity_to_asym": entity_to_asym, "asym_to_entity": asym_to_entity}

    def find_anchor_asym_chain_in_predictions(self) -> tuple[int]:
        """
        Find anchor chains in the prediction.

        Ref: AlphaFold3 SI Chapter 4.2. -> AlphaFold Multimer Chapter 7.3.1
        In the alignment phase, we pick a pair of anchor asyms to align,
        one in the ground truth and one in the prediction.
        The ground truth anchor asym a_gt is chosen to be the least ambiguous possible,
        for example in an A3B2 complex an arbitrary B asym is chosen.
        In the event of a tie e.g. A2B2 stoichiometry, the longest asym is chosen,
        with the hope that in general the longer asyms are likely to have higher confident predictions.
        The prediction anchor asym is chosen from the set {a^pred_m} of all prediction asyms
        with the same sequence as the ground truth anchor asym.

        Return:
            anchor_pred_asym_id (int): selected asym chain.
        """

        # Do not consider asym with fewer than 4 tokens in Prediction
        asym_to_asym_length = {
            asym_id: len(asym_dict["coordinate"])
            for asym_id, asym_dict in self.pred_asym_dict.items()
        }
        valid_asyms = [asym_id for asym_id, l in asym_to_asym_length.items() if l >= 4]

        # Do not consider entities with fewer than 4 resolved tokens in GT
        valid_entities = []
        for ent, asyms in self.label_token_dict["entity_to_asym"].items():
            if any(
                self.label_asym_dict[asym.item()]["coordinate_mask"].sum().item() >= 4
                for asym in asyms
            ):
                valid_entities.append(ent)

        valid_entity_asym = [
            (ent, asym.item())
            for ent in valid_entities
            for asym in self.pred_token_dict["entity_to_asym"][ent]
            if asym.item() in valid_asyms
        ]

        candidate_entities = set(ent for ent, _ in valid_entity_asym)

        # Find polymer chains in the prediction
        pred_polymer_entity_id = []
        for ent_id in candidate_entities:
            mask = self.pred_token_dict["entity_mol_id"] == ent_id
            is_ligand = self.pred_token_dict["is_ligand"][mask]
            if (
                torch.sum(is_ligand) <= is_ligand.shape[0] / 2
                and is_ligand.shape[0]
                >= 12  # do not prioritize asym with too few tokens
            ):
                pred_polymer_entity_id.append(ent_id)

        # Prioritize polymer
        if len(pred_polymer_entity_id) > 0:
            candidate_entities = pred_polymer_entity_id

        # Choose entities with fewest asyms in GT
        entity_to_asym_count = {
            k: len(self.label_token_dict["entity_to_asym"][k])
            for k in candidate_entities
        }
        min_asym_count = min(list(entity_to_asym_count.values()))
        candidate_entities = [
            ent
            for ent, count in entity_to_asym_count.items()
            if count == min_asym_count
        ]

        # Choose longest asyms in Prediction
        candidate_asyms = [
            asym_id for ent, asym_id in valid_entity_asym if ent in candidate_entities
        ]
        max_asym_length = max(
            asym_to_asym_length[asym_id] for asym_id in candidate_asyms
        )
        candidate_asyms = [
            asym_id
            for asym_id in candidate_asyms
            if asym_to_asym_length[asym_id] == max_asym_length
        ]

        # If multiple asym chains remain, return a random one.
        anchor_pred_asym_id = random.choice(candidate_asyms)

        return anchor_pred_asym_id

    @staticmethod
    def _select_atoms_by_mol_atom_index(input_dict: dict, mol_atom_index: torch.Tensor):
        """
        Select atoms from the input dictionary based on the specified mol_atom_index.

        Args:
            input_dict (dict): Input dict.
            mol_atom_index (torch.Tensor): A tensor of atom indices.

        Returns:
            dict: A dictionary containing the selected atom features.
        """
        mask = torch.isin(input_dict["mol_atom_index"], mol_atom_index)
        out_dict = {k: v[mask] for k, v in input_dict.items()}
        assert (out_dict["mol_atom_index"] == mol_atom_index).all()
        return out_dict

    def _align_label_by_match(self, match: dict) -> bool:
        """Superpose the groundtruth onto the prediction USING a candidate match, so the candidates
        can be scored with the existing calculate_rmsd.

        calculate_rmsd reads label_asym_dict[...]["aligned_coordinate"], which the anchor heuristic
        fills in as a side effect of its own search -- calling it without that raises KeyError, and
        the caller's try/except turns that into "no chain permutation at all this step". Aligning on
        ALL matched chains at once rather than a single anchor also makes the comparison between
        candidates fair: each candidate is judged by how well the whole assembly fits under it.

        Returns False instead of raising if nothing alignable is left, so the caller can fall back.
        """
        P, G = [], []
        for pred_asym, gt_asym in match.items():
            if pred_asym not in self.pred_asym_dict or gt_asym not in self.label_asym_dict:
                return False
            pd = self.pred_asym_dict[pred_asym]
            gd = self.label_asym_dict[gt_asym]
            # Intersect on mol_atom_index rather than going through
            # _select_atoms_by_mol_atom_index: its assert requires the groundtruth chain to hold
            # exactly the predicted chain's index set, which holds for same-entity anchor pairs but
            # not for the ligand-follows-protein and identity-fallback pairs this enumeration makes.
            pi, gi = pd["mol_atom_index"], gd["mol_atom_index"]
            order = torch.argsort(gi)
            gi_sorted = gi[order]
            pos = torch.searchsorted(gi_sorted, pi).clamp(max=max(len(gi_sorted) - 1, 0))
            hit = len(gi_sorted) > 0
            hit = hit & (gi_sorted[pos] == pi) if hit else torch.zeros_like(pi, dtype=torch.bool)
            if not bool(hit.any()):
                continue
            g_rows = order[pos[hit]]
            m = gd["coordinate_mask"][g_rows].bool()
            if m.any():
                P.append(pd["coordinate"][hit][m])
                G.append(gd["coordinate"][g_rows][m])
        if not P:
            return False
        rot, trans = get_optimal_transform(torch.cat(G, dim=0), torch.cat(P, dim=0))
        aligned = apply_transform(self.label_token_dict["coordinate"], rot, trans)
        for asym_id in self.label_asym_dict:
            self.label_asym_dict[asym_id]["aligned_coordinate"] = aligned[
                self.label_token_dict["mol_id"] == asym_id
            ]
        return True

    def compute_best_match_fibril(self):
        """Best match restricted to the symmetry group of a declared protofilament/layer labelling.

        Needed because pf_id and layer_id are MEASURED from the structure, unlike sym_id, whose
        labelling the official design treats as arbitrary (there is even a `shuffle_sym_ids`
        augmentation for it). Once the input declares "this chain is protofilament 1, layer 0",
        a match that sends it to a groundtruth chain in the other protofilament trains the model
        against its own input. The generic anchor heuristic searches all copies of the entity and
        will happily do that.

        The relative position encoding consumes (same_pf, clip(|Delta layer|)), whose symmetry
        group -- enumerated exhaustively over all 10! relabelings for a 5-layer, 2-protofilament
        assembly -- is exactly four elements: {identity, whole-stack reversal} x {protofilament
        swap}. Whole-stack reversal is in there because the sign of the layer axis has no reliable
        intrinsic definition; the protofilament swap is in there because nothing in the input says
        which stack should be called 1. Both must stay available or a correct prediction gets
        penalised roughly half the time.

        Returns None when the sample is not a two-protofilament fibril, so the caller falls back
        to the official heuristic and non-fibril behaviour is unchanged.
        """
        chain_of = getattr(self, "_fibril_chain", None)
        if not chain_of:
            return None
        mol = self.pred_token_dict["mol_id"]
        if len(chain_of) < 4 or len({p for p, _ in chain_of.values()}) != 2:
            return None
        n_layer = len({l for _, l in chain_of.values()})
        by_key = {v: k for k, v in chain_of.items()}
        if len(by_key) != len(chain_of):
            return None                      # (pf, layer) not unique -> not a clean fibril

        gt_chains = set(torch.unique(self.label_token_dict["mol_id"]).tolist())

        # Ligand copies must follow the protein relabeling, not stay fixed. Once identical copies
        # share an entity (see dataset._merge_identical_ligand_entities) they become permutable, and
        # a ligand bound in a pocket of protofilament 1 layer 2 has to map to the one bound at
        # protofilament 2 layer 2 under the swap candidate -- leaving it identity would ask the model
        # for a protein arrangement and a ligand arrangement that contradict each other.
        #
        # The binding site is a property of the DATA, so it is read off the label coordinates: each
        # ligand chain inherits the (pf, layer) of the nearest protein chain. Pred coordinates would
        # be useless here, being near-random early in training.
        lig_key, lig_by_key = {}, {}
        try:
            prot_xyz = {}
            for a in chain_of:
                d = self.label_asym_dict.get(a)
                if d is None:
                    continue
                m = d["coordinate_mask"].bool()
                if m.any():
                    prot_xyz[a] = d["coordinate"][m]
            for a in torch.unique(mol).tolist():
                if a in chain_of or a not in self.label_asym_dict:
                    continue
                d = self.label_asym_dict[a]
                m = d["coordinate_mask"].bool()
                if not m.any() or not prot_xyz:
                    continue
                lc = d["coordinate"][m]
                nearest = min(
                    prot_xyz,
                    key=lambda p: torch.cdist(lc, prot_xyz[p]).min().item(),
                )
                lig_key[a] = chain_of[nearest]
            for a, k in lig_key.items():
                lig_by_key.setdefault(k, []).append(a)
            # ambiguous if two ligands share a site: fall back to identity for ligands
            if any(len(v) > 1 for v in lig_by_key.values()):
                lig_key, lig_by_key = {}, {}
        except Exception as exc:
            logger.warning("ligand-follows-protein mapping unavailable (%s); ligands kept identity", exc)
            lig_key, lig_by_key = {}, {}

        candidates = []
        cand_names = []
        for swap in (False, True):
            for rev in (False, True):
                match = {}
                for a, (p, l) in chain_of.items():
                    tgt = by_key.get(((3 - p) if swap else p, (n_layer - 1 - l) if rev else l))
                    if tgt is None or tgt not in gt_chains:
                        match = None
                        break
                    match[a] = tgt
                if match:
                    # ligands follow the same transform via their inherited binding site; any
                    # chain we could not place keeps its own identity
                    for a in torch.unique(mol).tolist():
                        if a in match or a not in gt_chains:
                            continue
                        k = lig_key.get(a)
                        if k is not None:
                            p, l = k
                            tk = ((3 - p) if swap else p, (n_layer - 1 - l) if rev else l)
                            tgt = lig_by_key.get(tk, [None])[0]
                            if tgt is not None and tgt in gt_chains:
                                match[a] = tgt
                                continue
                        match[a] = a
                    candidates.append(match)
                    cand_names.append(
                        ("swap" if swap else "") + ("rev" if rev else "") or "identity"
                    )
        if not candidates:
            return None
        best, best_rmsd, best_name = None, torch.inf, "?"
        cand_rmsd = {}
        for match, cname in zip(candidates, cand_names):
            try:
                if not self._align_label_by_match(match):
                    continue
                r = self.calculate_rmsd(match)
            except Exception as exc:
                logger.warning("fibril permutation candidate failed (%s); falling back", exc)
                return None
            cand_rmsd[cname] = float(r)
            if r < best_rmsd:
                best_rmsd, best, best_name = r, match, cname
        if best is None:
            return None

        # Guard: `rev` and `swaprev` are symmetries of the ENCODING (clip|Delta layer| cannot see the
        # direction of the layer axis) but NOT of the geometry -- applying either to the groundtruth
        # and superposing it back costs rise*sqrt((n^2-1)/3), which is 13.6 A for a 5-layer stack at
        # 4.8 A rise (measured: closed form vs 80 atlas fibrils, r=0.93, median |error| 0.06 A).
        # They have to stay available, because a flawless prediction that merely numbers its layers
        # the other way round is only rescued by `rev` (measured: rev wins 95% of such cases).
        # But when the prediction carries no information about layer order -- early training, or a
        # structure whose layer order is scrambled -- all four candidates land within 0.01-0.08 A of
        # each other and the argmin becomes a coin flip that lands on a 13.6 A-wrong label about half
        # the time. So accept a non-symmetry candidate only when the prediction actually prefers it:
        #   genuinely reverse-numbered  ->  rev beats the true symmetries by +11.9 A (median)
        #   coin flip                   ->  by +0.08 A
        # Anywhere in between is safe; the threshold is 1 A.
        guard = float(os.environ.get("PERM_GUARD_A", "1.0"))
        TRUE_SYM = ("identity", "swap")
        if guard > 0 and best_name not in TRUE_SYM and cand_rmsd:
            sym = {k: v for k, v in cand_rmsd.items() if k in TRUE_SYM}
            if sym:
                k_sym = min(sym, key=sym.get)
                if sym[k_sym] - best_rmsd < guard:
                    for match, cname in zip(candidates, cand_names):
                        if cname == k_sym:
                            if os.environ.get("MINI_MONITOR", "0") == "1":
                                print(
                                    "[PERMGUARD] rejected=%s (%.2f) -> %s (%.2f), advantage %.2f < %.2f"
                                    % (best_name, best_rmsd, k_sym, sym[k_sym],
                                       sym[k_sym] - best_rmsd, guard),
                                    flush=True,
                                )
                            best, best_rmsd, best_name = match, sym[k_sym], k_sym
                            self._align_label_by_match(best)
                            break

        # Which candidate wins is decided by the mini-rollout, and only two of the four are true
        # geometric symmetries: applying "rev" or "swaprev" to the groundtruth and superposing it
        # back gives a median 13.5 A RMSD over 60 atlas fibrils (identity 0.00, swap 1.39). They are
        # in the set because |Delta layer| makes reversal a symmetry of the ENCODING, so penalising
        # it would punish the model for something the input cannot tell it. The consequence is that
        # a collapsed rollout -- which fits none of the four -- can select a label 13.5 A from the
        # truth. Log the winner and the margin so that ordering can be checked against collapse.
        if os.environ.get("MINI_MONITOR", "0") == "1":
            others = sorted(v for k, v in cand_rmsd.items() if k != best_name)
            print(
                "[PERMPICK] best=%s rmsd=%.2f margin=%.2f all=%s"
                % (
                    best_name,
                    best_rmsd,
                    (others[0] - best_rmsd) if others else float("nan"),
                    ",".join("%s:%.2f" % (k, cand_rmsd[k]) for k in sorted(cand_rmsd)),
                ),
                flush=True,
            )
        # leave aligned_coordinate consistent with the match actually returned
        self._align_label_by_match(best)
        logger.info(
            "fibril-restricted chain permutation: %d candidate(s) (group is 4 for a clean "
            "2-protofilament assembly), best RMSD %.3f", len(candidates), float(best_rmsd)
        )
        return best

    def compute_best_match_heuristic(self):
        """
        Compute the best chain permutation between prediction and groundtruth.


        Returns:
            dict[int, int]: A dictionary mapping pred chain IDs to those of the groundtruth.
        """

        # Find anchor asym chain in predictions
        anchor_pred_asym_id = self.find_anchor_asym_chain_in_predictions()
        anchor_entity_id = self.pred_token_dict["asym_to_entity"][anchor_pred_asym_id]

        if self.find_gt_anchor_first:
            # Randomly sample a groundtruth asym chain using this entity id
            anchor_gt_asym_id = self.label_token_dict["entity_to_asym"][
                anchor_entity_id
            ].tolist()
            anchor_gt_asym_id = random.choice(anchor_gt_asym_id)

            # The candidate anchors to be matched are from prediction
            candidate_anchors = self.pred_token_dict["entity_to_asym"][anchor_entity_id]
        else:

            # The candidate anchors to be matched are from groundtruth
            candidate_anchors = self.label_token_dict["entity_to_asym"][
                anchor_entity_id
            ]

        # Find best match
        best_rmsd = torch.inf
        best_match = None

        for anchor_k in candidate_anchors:
            anchor_k = anchor_k.item()

            if self.find_gt_anchor_first:
                gt_anchor, pred_anchor = anchor_gt_asym_id, anchor_k
            else:
                gt_anchor, pred_anchor = anchor_k, anchor_pred_asym_id

            # Find atoms in GT chain to match atoms in predicted chain (which could be cropped)
            gt_anchor_dict = MultiChainPermutation._select_atoms_by_mol_atom_index(
                self.label_asym_dict[gt_anchor],
                mol_atom_index=self.pred_asym_dict[pred_anchor]["mol_atom_index"],
            )

            # Align GT Anchor to Pred Anchor
            mask = gt_anchor_dict["coordinate_mask"].bool()  # use GT coordinate_mask
            if not mask.any():
                continue
            rot, trans = get_optimal_transform(
                gt_anchor_dict["coordinate"][mask],
                self.pred_asym_dict[pred_anchor]["coordinate"][mask],
            )

            # Transform all GT coordinates according to the aligment results
            aligned_coordinate = apply_transform(
                self.label_token_dict["coordinate"], rot, trans
            )
            for asym_id in self.label_asym_dict:
                self.label_asym_dict[asym_id]["aligned_coordinate"] = (
                    aligned_coordinate[self.label_token_dict["mol_id"] == asym_id]
                )

            # Greedily matches all remaining chains
            matched_asym = {pred_anchor: gt_anchor}
            to_be_matched = [k for k in self.pred_asym_dict if k != pred_anchor]
            candidate_gt_asym_id = [k for k in self.label_asym_dict if k != gt_anchor]

            # Sort the remaining chains by their length, so that longer chain chooses its match first.
            to_be_matched = sorted(
                to_be_matched,
                key=lambda k: -self.pred_asym_dict[k]["coordinate"].size(-2),
            )

            while len(to_be_matched) > 0:
                cur_pred_asym_id = to_be_matched.pop(0)
                cur_entity_id = self.pred_token_dict["asym_to_entity"][cur_pred_asym_id]
                cur_gt_asym_ids = self.label_token_dict["entity_to_asym"][
                    cur_entity_id
                ].tolist()
                matched_gt_asym_id, _ = self.match_pred_asym_to_gt_asym(
                    cur_pred_asym_id,
                    [asym for asym in cur_gt_asym_ids if asym in candidate_gt_asym_id],
                )
                matched_asym[cur_pred_asym_id] = matched_gt_asym_id
                candidate_gt_asym_id.remove(matched_gt_asym_id)

            assert len(matched_asym) == len(self.pred_asym_dict)

            # Calculate RMSD
            total_rmsd = self.calculate_rmsd(matched_asym)

            if total_rmsd < best_rmsd:
                best_rmsd = total_rmsd
                best_match = matched_asym

        assert best_match is not None

        return best_match

    def calculate_rmsd(self, asym_match: dict):
        """
        Calculate the RMSD given a match.
        """

        return sum(self._calculate_rmsd(a, b) for a, b in asym_match.items()) / len(
            asym_match
        )

    def _calculate_rmsd(self, pred_asym_id: int, gt_asym_id: int):
        """
        Calculate the RMSD between the predicted and ground truth chains, either using the average of the representative atoms or all of them.

        Args:
            pred_asym_id (int): The ID of the predicted asymmetric chain.
            gt_asym_id (int): The ID of the ground truth asymmetric chain.

        Returns:
            float: The calculated RMSD.
        """

        pred_asym_dict = self.pred_asym_dict[pred_asym_id]
        label_asym_dict = MultiChainPermutation._select_atoms_by_mol_atom_index(
            self.label_asym_dict[gt_asym_id], pred_asym_dict["mol_atom_index"]
        )
        mask = label_asym_dict["coordinate_mask"].bool()
        if not mask.any():
            return 0.0
        elif self.use_center_rmsd:
            return rmsd(
                pred_asym_dict["coordinate"][mask].mean(dim=-2, keepdim=True),
                label_asym_dict["aligned_coordinate"][mask].mean(dim=-2, keepdim=True),
            ).item()
        else:
            return rmsd(
                pred_asym_dict["coordinate"][mask],
                label_asym_dict["aligned_coordinate"][mask],
            ).item()

    def match_pred_asym_to_gt_asym(self, pred_asym_id: int, gt_asym_ids: list):
        """
        Match a predicted  chain to the groundtruth chain based on the average of the representative atoms.

        Args:
            pred_asym_id (int): The ID of the predicted asymmetric chain.
            gt_asym_ids (list[int]): A list or tensor of ground truth asymmetric chain IDs.

        Returns:
            tuple: A tuple containing:
                - best_gt_asym_id (int): The ID of the best matched ground truth asymmetric chain.
                - best_error (float): The distance error between the centers of mass of the best matched chains.
        """

        pred_asym_dict = self.pred_asym_dict[pred_asym_id]

        best_error = torch.inf
        best_gt_asym_id = None
        unresolved_gt_asym_id = []
        for gt_asym_id in gt_asym_ids:
            if isinstance(gt_asym_id, torch.Tensor):
                gt_asym_id = gt_asym_id.item()

            # Select cropped atoms by comparing to mol_atom_index in prediction
            label_asym_dict = MultiChainPermutation._select_atoms_by_mol_atom_index(
                self.label_asym_dict[gt_asym_id], pred_asym_dict["mol_atom_index"]
            )
            mask = label_asym_dict["coordinate_mask"].bool()

            if not mask.any():
                # Skip unresolved ones
                unresolved_gt_asym_id.append(gt_asym_id)
                continue

            gt_center = label_asym_dict["aligned_coordinate"][mask].mean(dim=0)
            pred_center = pred_asym_dict["coordinate"][mask].mean(dim=0)

            delta = torch.norm(gt_center - pred_center)

            if delta < best_error:
                best_error = delta
                best_gt_asym_id = gt_asym_id

        if best_gt_asym_id is None:
            # If only unresolved ones remains, return the first one
            assert len(unresolved_gt_asym_id) > 0
            best_gt_asym_id, best_error = gt_asym_ids[0], 0

        return best_gt_asym_id, best_error

    @staticmethod
    def build_permuted_indice(
        pred_dict: dict, label_full_dict: dict, best_match: dict[int, int]
    ):
        """
        Build permutation indices from the pred-gt chain mapping.
        Args:
            pred_dict (dict): A dictionary containing the predicted coordinates.
            label_full_dict (dict): A dictionary containing the true coordinates and their masks.
            best_match (dict[int, int]): {pred_mol_id: gt_mol_id} best match between pred asym chains and gt asym chains

        Returns:
            indices (torch.Tensor): Permutation indices.
        """

        # Get the number of predicted (cropped) atoms
        N_pred_atom = pred_dict["mol_id"].size(0)
        N_label_atom = label_full_dict["mol_id"].size(0)
        indices = pred_dict["mol_id"].new_zeros(size=(N_pred_atom,))
        full_indices = torch.arange(N_label_atom, device=indices.device)

        for pred_asym_id, gt_asym_id in best_match.items():
            # Create a mask for the predicted asym_id
            mask = pred_dict["mol_id"] == pred_asym_id
            mol_atom_index = pred_dict["mol_atom_index"][mask]

            # Creat a mask for the matched gt asym_id
            gt_mask = label_full_dict["mol_id"] == gt_asym_id
            # Extract indices according to 'mol_atom_index'
            gt_asym_dict = MultiChainPermutation._select_atoms_by_mol_atom_index(
                {
                    "mol_atom_index": label_full_dict["mol_atom_index"][gt_mask],
                    "indices": full_indices[gt_mask],
                },
                mol_atom_index,
            )
            indices[mask] = gt_asym_dict["indices"].clone()

        assert len(torch.unique(indices)) == len(indices)
        return indices

    @staticmethod
    def aligned_rmsd(
        pred_dict: dict,
        label_full_dict: dict,
        indices: torch.Tensor,
        reduce: bool = True,
        eps: float = 1e-8,
    ):
        """
        Calculate the global aligned RMSD between predicted and true coordinates.

        Args:
            pred_dict (dict): A dictionary containing the predicted coordinates.
            label_full_dict (dict): A dictionary containing the true coordinates and their masks.
            indices (torch.Tensor): Indices to select from the true coordinates.
            reduce (bool): If True, reduce the RMSD over the batch dimension.
            eps (float): A small value to avoid division by zero.

        Returns:
            float: The aligned RMSD value.
        """

        with torch.amp.autocast("cuda", enabled=False):
            aligned_rmsd, _, _, _ = self_aligned_rmsd(
                pred_pose=pred_dict["coordinate"].to(torch.float32),
                true_pose=label_full_dict["coordinate"][indices, :].to(torch.float32),
                atom_mask=label_full_dict["coordinate_mask"][indices],
                allowing_reflection=False,
                reduce=reduce,
                eps=eps,
            )
        return aligned_rmsd.item()

    def __call__(
        self,
        pred_dict: dict[str, torch.Tensor],
        label_full_dict: dict[str, torch.Tensor],
        max_num_chains: int = 20,
    ):
        """
        Call function for the class

        Args:
            pred_dict (dict): A dictionary containing the predicted coordinates.
            label_full_dict (dict): A dictionary containing the groundtruth and its attributes.
            max_num_chains (int): Maximum number of chains allowed.

        Returns:
            tuple: A tuple containing:
                - best_match (dict[int, int]): The best match between predicted and groundtruth chains.
                - permute_pred_indices (torch.Tensor or None): Indices to permute the predicted coordinates.
                - permuted_indices (torch.Tensor): Indices to permute the groundtruth coordinates.
                - log_dict (dict): A dictionary detailing the permutation information.
        """
        match, has_sym_chain = self.process_input(
            pred_dict, label_full_dict, max_num_chains
        )

        if match is not None:
            """
            Either the structure does not contain symmetric chains, or
            there are too many chains so that the algorithm gives up.
            """
            indices = self.build_permuted_indice(pred_dict, label_full_dict, match)
            pred_indices = torch.argsort(indices)
            return match, pred_indices, indices, {"has_sym_chain": False}

        # Core step: get best mol_id match

        # Fibril samples carry a MEASURED protofilament/layer labelling, so their permutation
        # group is 4 elements rather than all copies of the entity; see compute_best_match_fibril.
        # Returns None for everything else, which keeps the official path byte-for-byte.
        best_match = self.compute_best_match_fibril()
        if best_match is None:
            best_match = self.compute_best_match_heuristic()

        permuted_indices = self.build_permuted_indice(
            pred_dict, label_full_dict, best_match
        )

        log_dict = {
            "has_sym_chain": True,
            "is_permuted": num_unique_matches([best_match, self.unpermuted_match]) > 1,
            "algo:no_permute": num_unique_matches([best_match, self.unpermuted_match])
            == 1,
        }

        if log_dict["algo:no_permute"]:
            # No permutation, return now
            pred_indices = torch.argsort(permuted_indices)
            return best_match, pred_indices, permuted_indices, log_dict

        # Compare rmsd before/after permutation
        unpermuted_indices = self.build_permuted_indice(
            pred_dict, label_full_dict, self.unpermuted_match
        )

        permuted_rmsd = self.aligned_rmsd(pred_dict, label_full_dict, permuted_indices)
        unpermuted_rmsd = self.aligned_rmsd(
            pred_dict, label_full_dict, unpermuted_indices
        )
        improved_rmsd = unpermuted_rmsd - permuted_rmsd
        if improved_rmsd >= 1e-12:
            # Case with better permutation
            log_dict.update(
                {
                    "algo:equivalent_permute": False,
                    "algo:worse_permute": False,
                    "algo:better_permute": True,
                    "algo:better_rmsd": improved_rmsd,
                }
            )
        elif improved_rmsd < 0:
            # Case with worse permutation
            log_dict.update(
                {
                    "algo:equivalent_permute": False,
                    "algo:worse_permute": True,
                    "algo:better_permute": False,
                    "algo:worse_rmsd": -improved_rmsd,
                }
            )
        elif not log_dict["algo:no_permute"]:
            # Case with equivalent permutation
            log_dict.update(
                {
                    "algo:equivalent_permute": True,
                    "algo:worse_permute": False,
                    "algo:better_permute": False,
                }
            )
        else:
            # No permutation
            log_dict["debug:zero_rmsd"] = improved_rmsd

        # Revert worse/equivalent permute to original chain assignment
        if (not self.accept_it_as_it_is) and (
            log_dict["algo:equivalent_permute"] or log_dict["algo:worse_permute"]
        ):
            # Revert to original chain assignment
            best_match = self.unpermuted_match
            permuted_indices = unpermuted_indices
            log_dict["is_permuted"] = False

        if pred_dict["coordinate"].size(-2) == label_full_dict["coordinate"].size(-2):
            Checker.is_permutation(permuted_indices)  # indices to permute/crop label
            permute_pred_indices = torch.argsort(
                permuted_indices
            )  # Indices to permute pred
        else:
            # Hard to `define` permute_pred_indices in this case
            permute_pred_indices = None

        return best_match, permute_pred_indices, permuted_indices, log_dict
