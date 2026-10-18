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
import logging
import os
from collections import defaultdict
from typing import Optional, Union

import numpy as np
import torch
from biotite.structure import Atom, AtomArray, get_residue_starts
from sklearn.neighbors import KDTree

from protenix.data.constants import STD_RESIDUES, STD_RESIDUES_WITH_GAP, get_all_elems
from protenix.data.tokenizer import Token, TokenArray
from protenix.data.utils import get_atom_level_token_mask, get_ligand_polymer_bond_mask
from protenix.utils.geometry import angle_3p, random_transform


def _protofilament_id(centre_atoms) -> torch.Tensor:
    """Per-token protofilament id, detected from geometry. 0 for everything non-fibril.

    Must come from coordinates, NOT chain order: 54% of our training bioassemblies interleave
    the two protofilaments in chain order (010101) while 100% of the inference receptors are
    blocked (0000011111), so any order-based rule is inconsistent between train and test.

    Detection is the same rise-graph used throughout this project: the fibril rise is the
    smallest median CA-CA distance over chain pairs (matched by residue index), and a
    protofilament is a connected component of the graph whose edges are the chain pairs sitting
    within RISE_TOL of that rise. Verified to find 2 protofilaments on 57/57 receptors.

    Encoding is three-state: 1 and 2 for the two protofilaments, 0 for "not applicable" --
    ligands, and every token when detection fails. 0 must be its own state rather than an alias
    for one protofilament: lumping ligands into protofilament 0 would declare them
    "same protofilament" as one stack and "different" from the other, which is arbitrary and
    asymmetric. The paired b_diff_pf fires only when BOTH tokens are 1-or-2 and they differ, so
    ligand pairs and undetected assemblies contribute exactly nothing.
    """
    RISE_TOL = 0.6
    n_tok = len(centre_atoms)
    out = torch.zeros(n_tok, dtype=torch.long)

    # Ablation switch. The 999-step checkpoint improved every 57-case metric far more than a
    # zero-initialised column should in 999 steps, and over those same steps the 139 pretrained
    # columns also moved by a comparable amount -- so "the new channel did it" and "999 steps of
    # ordinary training did it" are not yet separable. Forcing pf_id to zeros makes b_diff_pf
    # identically zero while keeping the exact same weights, which isolates the channel.
    if os.environ.get("PF_ID_OFF", ""):
        return out

    # Causal test of whether b_diff_pf is a live INPUT at inference rather than something the
    # weights merely absorbed during training. PF_ID_SPLIT=k DECLARES a split -- the first k
    # protein chains (order of appearance) become protofilament 1, the rest protofilament 2 --
    # instead of detecting one. Declaring 4+6 or 3+7 then asks the model to honour a grouping it
    # never saw in training, where every sample is 5+5. If the predictions follow the declaration
    # the bit is doing work at test time; if they stay 5+5 the topology was memorised.
    #
    # This is NOT a GT-derived edit: the split is an arbitrary number we choose, not a measurement
    # of the answer, and the interesting outcomes are the ones that disagree with the deposited
    # structure. Also the natural prototype for the JSON-declared pf_id discussed as fix B.
    # PF_ID_SPLIT=alt declares the INTERLEAVED assignment (1,2,1,2,...) instead of a block. This
    # is the discriminating case: an unequal block split like 3+7 never occurs in training, so
    # ignoring it is equally consistent with "the channel is dead at inference" and "the channel
    # works but 3+7 is out of distribution". Interleaved 5+5 is IN distribution -- 54% of the
    # training bioassemblies order their chains 0101... -- while still being a completely
    # different grouping from the blocked 5+5 the receptors use. If the model honours it the
    # channel is live; if it still emits blocked halves the channel does not steer topology.
    split = os.environ.get("PF_ID_SPLIT", "")
    if split:
        is_prot = ~np.asarray(centre_atoms.hetero, dtype=bool)
        asym = np.asarray(centre_atoms.asym_id_int)
        chains = [c for c in dict.fromkeys(asym[is_prot].tolist())]
        if split in ("alt", "interleave"):
            lut = {c: 1 + (i % 2) for i, c in enumerate(chains)}
        else:
            k = int(split)
            lut = {c: (1 if i < k else 2) for i, c in enumerate(chains)}
        for i in range(n_tok):
            out[i] = lut.get(int(asym[i]), 0)
        logging.getLogger(__name__).warning(
            "PF_ID_SPLIT=%s active: DECLARING protofilament ids %s (not detecting). "
            "Diagnostic for whether b_diff_pf is a live inference input.",
            split, [(int(c), lut[c]) for c in chains],
        )
        return out

    # Prefer the annotation set on the FULL assembly before cropping (dataset.py
    # _annotate_protofilament). Re-detecting here works on the cropped fragment, where the rise
    # graph collapses to one component on ~5% of real crops; the pre-crop labels do not have that
    # failure mode. Falls back to detecting locally so the inference path -- which builds features
    # straight from a JSON and never goes through the dataset -- still works.
    if hasattr(centre_atoms, "pf_id_int"):
        try:
            return torch.as_tensor(
                np.asarray(centre_atoms.pf_id_int, dtype=np.int64), dtype=torch.long
            )
        except Exception:
            pass

    try:
        is_prot = ~np.asarray(centre_atoms.hetero, dtype=bool)
        asym = np.asarray(centre_atoms.asym_id_int)
        res = np.asarray(centre_atoms.res_id).astype(int)
        xyz = np.asarray(centre_atoms.coord, dtype=float)
        chains = [c for c in dict.fromkeys(asym[is_prot].tolist())]
        if len(chains) < 4:
            return out
        # residue_index -> coord per chain, so pairs are compared at the same position
        per = {}
        for c in chains:
            m = is_prot & (asym == c)
            per[c] = dict(zip(res[m].tolist(), xyz[m]))
        # Pairwise common residues, not a global intersection. Cropping keeps a different
        # residue range per chain, so requiring residues shared by ALL chains made detection
        # collapse on 16% of real crops; per-pair overlap recovers most of those.
        k = len(chains)
        D = np.full((k, k), np.inf)
        MIN_SHARED = 3
        for a in range(k):
            for b in range(a + 1, k):
                sh = sorted(set(per[chains[a]]) & set(per[chains[b]]))
                if len(sh) < MIN_SHARED:
                    continue
                A = np.array([per[chains[a]][r] for r in sh])
                B = np.array([per[chains[b]][r] for r in sh])
                D[a, b] = D[b, a] = float(np.median(np.linalg.norm(A - B, axis=1)))
        off = D[np.triu_indices(k, 1)]
        off = off[np.isfinite(off)]
        if off.size == 0:
            return out
        rise = float(off.min())
        adj = {i: set() for i in range(k)}
        for a in range(k):
            for b in range(a + 1, k):
                if np.isfinite(D[a, b]) and abs(D[a, b] - rise) <= RISE_TOL:
                    adj[a].add(b); adj[b].add(a)
        comp, cid = {}, 0
        for st0 in range(k):
            if st0 in comp:
                continue
            stack, comp[st0] = [st0], cid
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if v not in comp:
                        comp[v] = cid; stack.append(v)
            cid += 1
        if cid != 2:
            return out
        lut = {chains[i]: comp[i] + 1 for i in range(k)}   # 1 / 2; 0 stays "not applicable"
        for i in range(n_tok):
            out[i] = lut.get(int(asym[i]), 0)
    except Exception:
        return torch.zeros(n_tok, dtype=torch.long)
    return out


def _layer_id(centre_atoms, pf: torch.Tensor) -> torch.Tensor:
    """Per-token layer index INSIDE its protofilament; 0 wherever pf_id is 0.

    Only ever read where pf_id > 0 -- the relative position encoding falls back to the official
    signed sym_id path otherwise -- so "0 elsewhere" needs no sym_id fallback and non-fibril
    samples are untouched.

    The rise graph that produced pf_id already ordered the chains, so this only has to rank the
    chain centroids along each protofilament's own axis. Prefers the pre-crop annotation from
    dataset.py for the same reason pf_id does: re-deriving on a cropped fragment is less reliable.
    """
    n_tok = len(centre_atoms)
    out = torch.zeros(n_tok, dtype=torch.long)

    # DECLARED layer ids, paired with a declared pf_id (PF_ID_SPLIT). At inference the layer index
    # cannot be detected: features are built from the JSON, so the coordinates at this point are
    # CCD reference conformers with every chain copy superimposed, and ranking them gives noise.
    # But it does not need detecting -- "the first 5 copies are protofilament 1, layers 0..4" is a
    # statement of INTENT by whoever wrote the input, exactly like `count: 10`, and carries no
    # information about the answer. Layer is the position within each declared group, in chain
    # order, which is how every receptor in this project is actually built (verified blocked 57/57).
    if os.environ.get("PF_ID_SPLIT", ""):
        try:
            asym = np.asarray(centre_atoms.asym_id_int)
            pfn = pf.numpy()
            seen = {}
            for g in np.unique(pfn[pfn > 0]):
                for c in dict.fromkeys(asym[pfn == g].tolist()):
                    seen[c] = len([k for k, v in seen.items() if pfn[asym == k][0] == g])
            for c, r in seen.items():
                out[torch.as_tensor(asym == c)] = r
            return out
        except Exception:
            return torch.zeros(n_tok, dtype=torch.long)

    if hasattr(centre_atoms, "layer_id_int"):
        try:
            return torch.as_tensor(
                np.asarray(centre_atoms.layer_id_int, dtype=np.int64), dtype=torch.long
            )
        except Exception:
            pass
    if int(pf.max()) == 0:
        return out
    try:
        asym = np.asarray(centre_atoms.asym_id_int)
        xyz = np.asarray(centre_atoms.coord, dtype=float)
        pfn = pf.numpy()
        for g in np.unique(pfn[pfn > 0]):
            m = pfn == g
            ch = list(dict.fromkeys(asym[m].tolist()))
            cen = np.array([xyz[m & (asym == c)].mean(0) for c in ch])
            X = cen - cen.mean(0)
            axis = np.linalg.svd(X)[2][0]
            for rank, i in enumerate(np.argsort(X @ axis)):
                out[torch.as_tensor(asym == ch[i]) & torch.as_tensor(m)] = rank
    except Exception:
        return torch.zeros(n_tok, dtype=torch.long)
    return out


class Featurizer(object):
    def __init__(
        self,
        cropped_token_array: TokenArray,
        cropped_atom_array: AtomArray,
        ref_pos_augment: bool = True,
        lig_atom_rename: bool = False,
        include_discont_poly_poly_bonds: bool = False,
        debug_pdb_id: Optional[str] = None,
    ) -> None:
        """
        Args:
            cropped_token_array (TokenArray): TokenArray object after cropping
            cropped_atom_array (AtomArray): AtomArray object after cropping
            ref_pos_augment (bool): Boolean indicating whether apply random rotation and translation on ref_pos
            lig_atom_rename (bool): Boolean indicating whether rename atom name for ligand atoms
            include_discont_poly_poly_bonds (bool): Boolean indicating whether
                                            include discontinuous polymer-polymer bonds
            debug_pdb_id (str, optional): PDB/sample id for debugging (e.g. when is_ligand mask is all zeros).
        """
        self.cropped_token_array = cropped_token_array
        self.debug_pdb_id = debug_pdb_id

        self.cropped_atom_array = cropped_atom_array
        self.ref_pos_augment = ref_pos_augment
        self.lig_atom_rename = lig_atom_rename
        self.include_discont_poly_poly_bonds = include_discont_poly_poly_bonds

    @staticmethod
    def encoder(
        encode_def_dict_or_list: Optional[Union[dict, list[str]]], input_list: list[str]
    ) -> torch.Tensor:
        """
        Encode a list of input values into a binary format using a specified encoding definition list.

        Args:
            encode_def_dict_or_list (list or dict): A list or dict of encoding definitions.
            input_list (list): A list of input values to be encoded.

        Returns:
            torch.Tensor: A tensor representing the binary encoding of the input values.
        """
        num_keys = len(encode_def_dict_or_list)
        if isinstance(encode_def_dict_or_list, dict):
            items = encode_def_dict_or_list.items()
            assert (
                num_keys == max(encode_def_dict_or_list.values()) + 1
            ), "Do not use discontinuous number, which might causing potential bugs in the code"
        elif isinstance(encode_def_dict_or_list, list):
            items = ((key, idx) for idx, key in enumerate(encode_def_dict_or_list))
        else:
            raise TypeError(
                "encode_def_dict_or_list must be a list or dict, "
                f"but got {type(encode_def_dict_or_list)}"
            )
        onehot_dict = {
            key: [int(i == idx) for i in range(num_keys)] for key, idx in items
        }
        onehot_encoded_data = [onehot_dict[item] for item in input_list]
        onehot_tensor = torch.Tensor(onehot_encoded_data)
        return onehot_tensor

    @staticmethod
    def restype_onehot_encoded(restype_list: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "restype"
        One-hot encoding of the sequence. 32 possible values: 20 amino acids + unknown,
        4 RNA nucleotides + unknown, 4 DNA nucleotides + unknown, and gap.
        Ligands represented as “unknown amino acid”.

        Args:
            restype_list (List[str]): A list of residue types.
                                      The residue type of ligand should be "UNK" in the input list.

        Returns:
            torch.Tensor:  A Tensor of one-hot encoded residue types
        """

        return Featurizer.encoder(STD_RESIDUES_WITH_GAP, restype_list)

    @staticmethod
    def elem_onehot_encoded(elem_list: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "ref_element"
        One-hot encoding of the element atomic number for each atom
        in the reference conformer, up to atomic number 128.

        Args:
            elem_list (List[str]): A list of element symbols.

        Returns:
            torch.Tensor:  A Tensor of one-hot encoded elements
        """
        return Featurizer.encoder(get_all_elems(), elem_list)

    @staticmethod
    def ref_atom_name_chars_encoded(atom_names: list[str]) -> torch.Tensor:
        """
        Ref: AlphaFold3 SI Table 5 "ref_atom_name_chars"
        One-hot encoding of the unique atom names in the reference conformer.
        Each character is encoded as ord(c) − 32, and names are padded to length 4.

        Args:
            atom_name_list (List[str]): A list of atom names.

        Returns:
            torch.Tensor:  A Tensor of character encoded atom names
        """
        onehot_dict = {}
        for index, key in enumerate(range(64)):
            onehot = [0] * 64
            onehot[index] = 1
            onehot_dict[key] = onehot
        # [N_atom, 4, 64]
        mol_encode = []
        for atom_name in atom_names:
            # [4, 64]
            atom_encode = []
            for name_str in atom_name.ljust(4):
                atom_encode.append(onehot_dict[ord(name_str) - 32])
            mol_encode.append(atom_encode)
        onehot_tensor = torch.Tensor(mol_encode)
        return onehot_tensor

    @staticmethod
    def get_prot_nuc_frame(token: Token, centre_atom: Atom) -> tuple[int, list[int]]:
        """
        Ref: AlphaFold3 SI Chapter 4.3.2
        For proteins/DNA/RNA, we use the three atoms [N, CA, C] / [C1', C3', C4']

        Args:
            token (Token): Token object.
            centre_atom (Atom): Biotite Atom object of Token centre atom.

        Returns:
            has_frame (int): 1 if the token has frame, 0 otherwise.
            frame_atom_index (List[int]): The index of the atoms used to construct the frame.
        """
        if centre_atom.mol_type == "protein":
            # For protein
            abc_atom_name = ["N", "CA", "C"]
        else:
            # For DNA and RNA
            abc_atom_name = [r"C1'", r"C3'", r"C4'"]

        idx_in_atom_indices = []
        for i in abc_atom_name:
            if centre_atom.mol_type == "protein" and "N" not in token.atom_names:
                return 0, [-1, -1, -1]
            elif centre_atom.mol_type != "protein" and "C1'" not in token.atom_names:
                return 0, [-1, -1, -1]
            idx_in_atom_indices.append(token.atom_names.index(i))
        # Protein/DNA/RNA always has frame
        has_frame = 1
        frame_atom_index = [token.atom_indices[i] for i in idx_in_atom_indices]
        return has_frame, frame_atom_index

    @staticmethod
    def get_lig_frame(
        token: Token,
        centre_atom: Atom,
        lig_res_ref_conf_kdtree: dict[str, tuple[KDTree, list[int]]],
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> tuple[int, list[int]]:
        """
        Ref: AlphaFold3 SI Chapter 4.3.2
        For ligands, we use the reference conformer of the ligand to construct the frame.

        Args:
            token (Token): Token object.
            centre_atom (Atom): Biotite Atom object of Token centre atom.
            lig_res_ref_conf_kdtree (Dict[str, Tuple[KDTree, List[int]]]): A dictionary of KDTree objects and atom indices.
            ref_pos (torch.Tensor): Atom positions in the reference conformer. Size=[N_atom, 3]
            ref_mask (torch.Tensor): Mask indicating which atom slots are used in the reference conformer. Size=[N_atom]

        Returns:
            tuple[int, List[int]]:
                has_frame (int): 1 if the token has frame, 0 otherwise.
                frame_atom_index (List[int]): The index of the atoms used to construct the frame.
        """
        kdtree, atom_ids = lig_res_ref_conf_kdtree[centre_atom.ref_space_uid]
        b_ref_pos = ref_pos[token.centre_atom_index]
        b_idx = token.centre_atom_index
        if kdtree is None:
            # Atom num < 3
            frame_atom_index = [-1, b_idx, -1]
            has_frame = 0
        else:
            _dist, ind = kdtree.query([b_ref_pos], k=3)
            a_idx, c_idx = atom_ids[ind[0][1]], atom_ids[ind[0][2]]
            frame_atom_index = [a_idx, b_idx, c_idx]

            # Check if reference confomrer vaild
            has_frame = all([ref_mask[idx] for idx in frame_atom_index])

            # Colinear check
            if has_frame:
                vec1 = ref_pos[frame_atom_index[1]] - ref_pos[frame_atom_index[0]]
                vec2 = ref_pos[frame_atom_index[2]] - ref_pos[frame_atom_index[1]]
                # ref_pos can be all zeros, in which case has_frame=0
                is_zero_norm = np.isclose(
                    np.linalg.norm(vec1, axis=-1), 0
                ) or np.isclose(np.linalg.norm(vec2, axis=-1), 0)
                if is_zero_norm:
                    has_frame = 0
                else:
                    theta_degrees = angle_3p(
                        *[ref_pos[idx] for idx in frame_atom_index]
                    )
                    is_colinear = theta_degrees <= 25 or theta_degrees >= 155
                    if is_colinear:
                        has_frame = 0
        return has_frame, frame_atom_index

    @staticmethod
    def get_token_frame(
        token_array: TokenArray,
        atom_array: AtomArray,
        ref_pos: torch.Tensor,
        ref_mask: torch.Tensor,
    ) -> TokenArray:
        """
        Ref: AlphaFold3 SI Chapter 4.3.2
        The atoms (a_i, b_i, c_i) used to construct token i’s frame depend on the chain type of i:
        Protein tokens use their residue’s backbone (N, Cα, C),
        while DNA and RNA tokens use (C1′, C3′, C4′) atoms of their residue.
        All other tokens (small molecules, glycans, ions) contain only one atom per token.
        The token atom is assigned to b_i, the closest atom to the token atom is a_i,
        and the second closest atom to the token atom is c_i.
        If this set of three atoms is close to colinear (less than 25 degree deviation),
        or if three atoms do not exist in the chain (e.g. a sodium ion),
        then the frame is marked as invalid.

        Note: frames constucted from reference conformer

        Args:
            token_array (TokenArray): A list of tokens.
            atom_array (AtomArray): An atom array.
            ref_pos (torch.Tensor): Atom positions in the reference conformer. Size=[N_atom, 3]
            ref_mask (torch.Tensor): Mask indicating which atom slots are used in the reference conformer. Size=[N_atom]

        Returns:
            TokenArray: A TokenArray with updated frame annotations.
                        - has_frame: 1 if the token has frame, 0 otherwise.
                        - frame_atom_index: The index of the atoms used to construct the frame.
        """
        token_array_w_frame = token_array
        atom_level_token_mask = get_atom_level_token_mask(token_array, atom_array)

        # Construct a KDTree for queries to avoid redundant distance calculations
        lig_res_ref_conf_kdtree = {}
        # Ligand and non-standard residues need to use ref to identify frames
        lig_atom_array = atom_array[
            (atom_array.mol_type == "ligand")
            | (~np.isin(atom_array.res_name, list(STD_RESIDUES.keys())))
            | atom_level_token_mask
        ]
        for ref_space_uid in np.unique(lig_atom_array.ref_space_uid):
            # The ref_space_uid is the unique identifier ID for each residue.
            atom_ids = np.where(atom_array.ref_space_uid == ref_space_uid)[0]
            if len(atom_ids) >= 3:
                kdtree = KDTree(ref_pos[atom_ids], metric="euclidean")
            else:
                # Invalid frame
                kdtree = None
            lig_res_ref_conf_kdtree[ref_space_uid] = (kdtree, atom_ids)

        has_frame = []
        for token in token_array_w_frame:
            centre_atom = atom_array[token.centre_atom_index]
            if (
                centre_atom.mol_type != "ligand"
                and centre_atom.res_name in STD_RESIDUES
                and len(token.atom_indices) > 1
            ):
                has_frame, frame_atom_index = Featurizer.get_prot_nuc_frame(
                    token, centre_atom
                )

            else:
                has_frame, frame_atom_index = Featurizer.get_lig_frame(
                    token, centre_atom, lig_res_ref_conf_kdtree, ref_pos, ref_mask
                )

            token.has_frame = has_frame
            token.frame_atom_index = frame_atom_index
        return token_array_w_frame

    def get_token_features(self) -> dict[str, torch.Tensor]:
        """
        Ref: AlphaFold3 SI Chapter 2.8

        Get token features.
        The size of these features is [N_token].

        Returns:
            Dict[str, torch.Tensor]: A dict of token features.
        """
        token_features = {}

        centre_atoms_indices = self.cropped_token_array.get_annotation(
            "centre_atom_index"
        )
        centre_atoms = self.cropped_atom_array[centre_atoms_indices]

        restype = centre_atoms.cano_seq_resname
        restype_onehot = self.restype_onehot_encoded(restype)

        token_features["token_index"] = torch.arange(0, len(self.cropped_token_array))
        token_features["residue_index"] = torch.Tensor(
            centre_atoms.res_id.astype(int)
        ).long()
        token_features["asym_id"] = torch.Tensor(centre_atoms.asym_id_int).long()
        token_features["entity_id"] = torch.Tensor(centre_atoms.entity_id_int).long()
        token_features["sym_id"] = torch.Tensor(centre_atoms.sym_id_int).long()
        token_features["pf_id"] = _protofilament_id(centre_atoms)
        token_features["layer_id"] = _layer_id(centre_atoms, token_features["pf_id"])
        token_features["restype"] = restype_onehot

        return token_features

    def get_chain_perm_features(self) -> dict[str, torch.Tensor]:
        """
        The chain permutation use "entity_mol_id", "mol_id" and "mol_atom_index"
        instead of the "entity_id", "asym_id" and "residue_index".

        The shape of these features is [N_atom].

        Returns:
            Dict[str, torch.Tensor]: A dict of chain permutation features.
        """

        chain_perm_features = {}
        chain_perm_features["mol_id"] = torch.Tensor(
            self.cropped_atom_array.mol_id
        ).long()
        chain_perm_features["mol_atom_index"] = torch.Tensor(
            self.cropped_atom_array.mol_atom_index
        ).long()
        chain_perm_features["entity_mol_id"] = torch.Tensor(
            self.cropped_atom_array.entity_mol_id
        ).long()
        return chain_perm_features

    def get_renamed_atom_names(self) -> np.ndarray:
        """
        Rename the atom names of ligands to avioid information leakage.

        Returns:
            np.ndarray: A numpy array of renamed atom names.
        """
        res_starts = get_residue_starts(
            self.cropped_atom_array, add_exclusive_stop=True
        )
        new_atom_names = copy.deepcopy(self.cropped_atom_array.atom_name)
        for start, stop in zip(res_starts[:-1], res_starts[1:]):
            res_mol_type = self.cropped_atom_array.mol_type[start]
            if res_mol_type != "ligand":
                continue

            elem_count = defaultdict(int)
            new_res_atom_names = []
            for elem in self.cropped_atom_array.element[start:stop]:
                elem_count[elem] += 1
                new_res_atom_names.append(f"{elem.upper()}{elem_count[elem]}")
            new_atom_names[start:stop] = new_res_atom_names
        return new_atom_names

    def get_reference_features(self) -> dict[str, torch.Tensor]:
        """
        Ref: AlphaFold3 SI Chapter 2.8

        Get reference features.
        The size of these features is [N_atom].

        Returns:
            Dict[str, torch.Tensor]: a dict of reference features.
        """
        # Global ref positions (shape preserved, one global center) for fix-protein diffusion, same as inference:
        # inference saves global_ref_pos before per-residue centralization; we do one global center so structure keeps shape.
        raw_ref_pos = np.asarray(self.cropped_atom_array.ref_pos, dtype=np.float64)
        global_ref_pos = raw_ref_pos - raw_ref_pos.mean(axis=0)

        ref_pos = []
        for ref_space_uid in np.unique(self.cropped_atom_array.ref_space_uid):
            res_ref_pos = random_transform(
                self.cropped_atom_array.ref_pos[
                    self.cropped_atom_array.ref_space_uid == ref_space_uid,
                ],
                apply_augmentation=self.ref_pos_augment,
                centralize=True,
            )
            ref_pos.append(res_ref_pos)
        ref_pos = np.concatenate(ref_pos)

        ref_features = {}
        ref_features["global_ref_pos"] = torch.Tensor(global_ref_pos)
        ref_features["ref_pos"] = torch.Tensor(ref_pos)
        ref_features["ref_mask"] = torch.Tensor(self.cropped_atom_array.ref_mask).long()
        ref_features["ref_element"] = Featurizer.elem_onehot_encoded(
            self.cropped_atom_array.element
        ).long()
        ref_features["ref_charge"] = torch.Tensor(
            self.cropped_atom_array.ref_charge
        ).long()

        if self.lig_atom_rename:
            atom_names = self.get_renamed_atom_names()
        else:
            atom_names = self.cropped_atom_array.atom_name

        ref_features["ref_atom_name_chars"] = Featurizer.ref_atom_name_chars_encoded(
            atom_names
        ).long()
        ref_features["ref_space_uid"] = torch.Tensor(
            self.cropped_atom_array.ref_space_uid
        ).long()

        token_array_with_frame = self.get_token_frame(
            token_array=self.cropped_token_array,
            atom_array=self.cropped_atom_array,
            ref_pos=ref_features["ref_pos"],
            ref_mask=ref_features["ref_mask"],
        )
        ref_features["has_frame"] = torch.Tensor(
            token_array_with_frame.get_annotation("has_frame")
        ).long()  # [N_token]
        ref_features["frame_atom_index"] = torch.Tensor(
            token_array_with_frame.get_annotation("frame_atom_index")
        ).long()  # [N_token, 3]
        return ref_features

    def get_bond_features(self) -> dict[str, torch.Tensor]:
        """
        Ref: AlphaFold3 SI Chapter 2.8
        A 2D matrix indicating if there is a bond between any atom in token i and token j,
        restricted to just polymer-ligand and ligand-ligand bonds and bonds less than 2.4 Å during training.
        The size of bond feature is [N_token, N_token].
        Returns:
            Dict[str, torch.Tensor]: A dict of bond features.
        """
        bond_array = self.cropped_atom_array.bonds.as_array()
        bond_atom_i = bond_array[:, 0]
        bond_atom_j = bond_array[:, 1]
        ref_space_uid = self.cropped_atom_array.ref_space_uid
        polymer_mask = np.isin(
            self.cropped_atom_array.mol_type, ["protein", "dna", "rna"]
        )
        std_res_mask = (
            np.isin(self.cropped_atom_array.res_name, list(STD_RESIDUES.keys()))
            & polymer_mask
        )
        unstd_res_mask = ~std_res_mask & polymer_mask
        # the polymer-polymer (std-std, std-unstd, and inter-unstd) bond will not be included in token_bonds.
        std_std_bond_mask = std_res_mask[bond_atom_i] & std_res_mask[bond_atom_j]
        std_unstd_bond_mask = (
            std_res_mask[bond_atom_i] & unstd_res_mask[bond_atom_j]
        ) | (std_res_mask[bond_atom_j] & unstd_res_mask[bond_atom_i])
        inter_unstd_bond_mask = (
            unstd_res_mask[bond_atom_i] & unstd_res_mask[bond_atom_j]
        ) & (ref_space_uid[bond_atom_i] != ref_space_uid[bond_atom_j])

        kept_mask = ~(std_std_bond_mask | std_unstd_bond_mask | inter_unstd_bond_mask)
        if self.include_discont_poly_poly_bonds:
            # include discontinuous polymer-polymer bonds
            res_id_i = self.cropped_atom_array.res_id[bond_atom_i]
            res_id_j = self.cropped_atom_array.res_id[bond_atom_j]
            chain_i = self.cropped_atom_array.chain_id[bond_atom_i]
            chain_j = self.cropped_atom_array.chain_id[bond_atom_j]
            is_discont = (np.abs(res_id_i - res_id_j) > 1) | (chain_i != chain_j)
            kept_mask |= is_discont

        kept_bonds = bond_array[kept_mask]

        # -1 means the atom is not in any token
        atom_idx_to_token_idx = np.zeros(len(self.cropped_atom_array), dtype=int) - 1
        for idx, token in enumerate(self.cropped_token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_idx_to_token_idx[atom_idx] = idx
        assert np.all(atom_idx_to_token_idx >= 0), "Some atoms are not in any token"
        num_tokens = len(self.cropped_token_array)
        token_adj_matrix = np.zeros((num_tokens, num_tokens), dtype=int)
        bond_token_i, bond_atom_j = (
            atom_idx_to_token_idx[kept_bonds[:, 0]],
            atom_idx_to_token_idx[kept_bonds[:, 1]],
        )
        for i, j in zip(bond_token_i, bond_atom_j):
            token_adj_matrix[i, j] = 1
            token_adj_matrix[j, i] = 1
        bond_features = {"token_bonds": torch.Tensor(token_adj_matrix)}
        return bond_features

    def get_extra_features(self) -> dict[str, torch.Tensor]:
        """
        Get other features not listed in AlphaFold3 SI Chapter 2.8 Table 5.
        The size of these features is [N_atom].

        Returns:
            Dict[str, torch.Tensor]: a dict of extra features.
        """
        atom_to_token_idx_dict = {}
        for idx, token in enumerate(self.cropped_token_array.tokens):
            for atom_idx in token.atom_indices:
                atom_to_token_idx_dict[atom_idx] = idx

        # Ensure the order of the atom_to_token_idx is the same as the atom_array
        atom_to_token_idx = [
            atom_to_token_idx_dict[atom_idx]
            for atom_idx in range(len(self.cropped_atom_array))
        ]

        extra_features = {}
        extra_features["atom_to_token_idx"] = torch.Tensor(atom_to_token_idx).long()
        extra_features["atom_to_tokatom_idx"] = torch.Tensor(
            self.cropped_atom_array.tokatom_idx
        ).long()

        atom_array = self.cropped_atom_array
        if hasattr(atom_array, "mol_type"):
            # Optional debug: check whether mol_type and res_name look as expected
            if os.getenv("DEBUG_TRAIN_NOISE", "false").lower() == "true":
                try:
                    import numpy as _np
                    mt_vals, mt_cnts = _np.unique(atom_array.mol_type, return_counts=True)
                    print(f"[DEBUG_FEAT] mol_type unique={mt_vals}, counts={mt_cnts}")
                    rn_vals, rn_cnts = _np.unique(atom_array.res_name, return_counts=True)
                    print(f"[DEBUG_FEAT] res_name unique(head)={rn_vals[:10]}, counts(head)={rn_cnts[:10]}")
                except Exception:
                    print("[DEBUG_FEAT] mol_type/res_name present but failed to summarize.")
            mol_type = atom_array.mol_type
            extra_features["is_ligand"] = torch.Tensor(
                (mol_type == "ligand").astype(int)
            ).long()
            extra_features["is_protein"] = torch.Tensor(
                (mol_type == "protein").astype(int)
            ).long()
        else:
            extra_features["is_protein"] = torch.Tensor(
                atom_array.is_protein
            ).long()
            extra_features["is_ligand"] = torch.Tensor(
                atom_array.is_ligand
            ).long()

        # Debug: when is_ligand mask is all zeros (DEBUG_IS_LIGAND_ZERO=1: print; DEBUG_IS_LIGAND_ZERO_PDB=1: print + pdb).
        # pdb here only works with num_workers=0 (otherwise workers have no TTY / can deadlock).
        if os.getenv("DEBUG_IS_LIGAND_ZERO", "0") == "1":
            n_lig = int(extra_features["is_ligand"].sum().item())
            if n_lig == 0:
                _aa = self.cropped_atom_array
                _ann = list(_aa.get_annotation_categories()) if hasattr(_aa, "get_annotation_categories") else []
                _has_mt = hasattr(_aa, "mol_type")
                _mt_uniq = np.unique(_aa.mol_type).tolist() if _has_mt else []
                _lig_sum = int(np.asarray(getattr(_aa, "is_ligand", np.array(0))).sum()) if hasattr(_aa, "is_ligand") else None
                print(f"[DEBUG_IS_LIGAND_ZERO] pdb_id={getattr(self, 'debug_pdb_id', None)}")
                print(f"[DEBUG_IS_LIGAND_ZERO] annotations={_ann}, has mol_type={_has_mt}, mol_type unique={_mt_uniq}, is_ligand.sum={_lig_sum}")
        extra_features["is_dna"] = torch.Tensor(atom_array.is_dna).long()
        extra_features["is_rna"] = torch.Tensor(atom_array.is_rna).long()
        if "resolution" in self.cropped_atom_array._annot:
            extra_features["resolution"] = torch.Tensor(
                [self.cropped_atom_array.resolution[0]]
            )
        else:
            extra_features["resolution"] = torch.Tensor([-1])
        return extra_features

    @staticmethod
    def get_lig_pocket_mask(
        atom_array: AtomArray, lig_label_asym_id: Union[str, list]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Ref: AlphaFold3 Chapter Methods.Metrics

        the pocket is defined as all heavy atoms within 10 Å of any heavy atom of the ligand,
        restricted to the primary polymer chain for the ligand or modified residue being scored,
        and further restricted to only backbone atoms for proteins. The primary polymer chain is defined variously:
        for PoseBusters it is the protein chain with the most atoms within 10 Å of the ligand,
        for bonded ligand scores it is the bonded polymer chain and for modified residues it
        is the chain that the residue is contained in (minus that residue).

        Args:
            atom_array (AtomArray): atoms in the complex.
            lig_label_asym_id (Union[str, List]): The label_asym_id of the ligand of interest.

        Returns:
            tuple[torch.Tensor, torch.Tensor]: A tuple of ligand pocket mask and pocket mask.
        """

        if isinstance(lig_label_asym_id, str):
            lig_label_asym_ids = [lig_label_asym_id]
        else:
            lig_label_asym_ids = list(lig_label_asym_id)

        # Get backbone mask
        prot_backbone = (
            atom_array.is_protein & np.isin(atom_array.atom_name, ["C", "N", "CA"])
        ).astype(bool)

        kdtree = KDTree(atom_array.coord)

        ligand_mask_list = []
        pocket_mask_list = []
        for lig_label_asym_id in lig_label_asym_ids:
            assert np.isin(
                lig_label_asym_id, atom_array.label_asym_id
            ), f"{lig_label_asym_id} is not in the label_asym_id of the cropped atom array."

            ligand_mask = atom_array.label_asym_id == lig_label_asym_id
            lig_pos = atom_array.coord[ligand_mask & atom_array.is_resolved]

            # Get atoms in 10 Angstrom radius
            near_atom_indices = np.unique(
                np.concatenate(kdtree.query_radius(lig_pos, 10.0))
            )
            near_atoms = [
                (
                    True
                    if ((i in near_atom_indices) and atom_array.is_resolved[i])
                    else False
                )
                for i in range(len(atom_array))
            ]

            # Get primary chain (protein backone in 10 Angstrom radius)
            primary_chain_candidates = near_atoms & prot_backbone
            primary_chain_candidates_atoms = atom_array[primary_chain_candidates]

            max_atom = 0
            primary_chain_asym_id_int = None
            for asym_id_int in np.unique(primary_chain_candidates_atoms.asym_id_int):
                n_atoms = np.sum(
                    primary_chain_candidates_atoms.asym_id_int == asym_id_int
                )
                if n_atoms > max_atom:
                    max_atom = n_atoms
                    primary_chain_asym_id_int = asym_id_int
            assert (
                primary_chain_asym_id_int is not None
            ), f"No primary chain found for ligand ({lig_label_asym_id=})."

            pocket_mask = primary_chain_candidates & (
                atom_array.asym_id_int == primary_chain_asym_id_int
            )
            ligand_mask_list.append(ligand_mask)
            pocket_mask_list.append(pocket_mask)

        ligand_mask_by_pockets = torch.Tensor(
            np.array(ligand_mask_list).astype(int)
        ).long()
        pocket_mask_by_pockets = torch.Tensor(
            np.array(pocket_mask_list).astype(int)
        ).long()
        return ligand_mask_by_pockets, pocket_mask_by_pockets

    def get_mask_features(self) -> dict[str, torch.Tensor]:
        """
        Generate mask features for the cropped atom array.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing various mask features.
        """
        mask_features = {}

        mask_features["pae_rep_atom_mask"] = torch.Tensor(
            self.cropped_atom_array.centre_atom_mask
        ).long()

        mask_features["plddt_m_rep_atom_mask"] = torch.Tensor(
            self.cropped_atom_array.plddt_m_rep_atom_mask
        ).long()  # [N_atom]

        mask_features["distogram_rep_atom_mask"] = torch.Tensor(
            self.cropped_atom_array.distogram_rep_atom_mask
        ).long()  # [N_atom]

        mask_features["modified_res_mask"] = torch.Tensor(
            self.cropped_atom_array.modified_res_mask
        ).long()

        lig_polymer_bonds = get_ligand_polymer_bond_mask(self.cropped_atom_array)
        num_atoms = len(self.cropped_atom_array)
        bond_mask_mat = np.zeros((num_atoms, num_atoms))
        for i, j, _ in lig_polymer_bonds:
            bond_mask_mat[i, j] = 1
            bond_mask_mat[j, i] = 1
        mask_features["bond_mask"] = torch.Tensor(
            bond_mask_mat
        ).long()  # [N_atom, N_atom]
        return mask_features

    def get_all_input_features(self):
        """
        Get input features from cropped data.

        Returns:
            Dict[str, torch.Tensor]: a dict of features.
        """
        features = {}
        token_features = self.get_token_features()
        features.update(token_features)

        bond_features = self.get_bond_features()
        features.update(bond_features)

        reference_features = self.get_reference_features()
        features.update(reference_features)

        extra_features = self.get_extra_features()
        features.update(extra_features)

        chain_perm_features = self.get_chain_perm_features()
        features.update(chain_perm_features)

        mask_features = self.get_mask_features()
        features.update(mask_features)
        return features

    def get_labels(self) -> dict[str, torch.Tensor]:
        """
        Get the input labels required for the training phase.

        Returns:
            Dict[str, torch.Tensor]: a dict of labels.
        """

        labels = {}

        coord = self.cropped_atom_array.coord
        # Diagnostic: verify whether NaN originates here (run with DEBUG_NAN=1)
        if os.environ.get("DEBUG_NAN", "0") == "1":
            coord_nan = np.isnan(np.asarray(coord, dtype=np.float64)).any(axis=1)
            if np.any(coord_nan):
                n_nan = coord_nan.sum()
                resolved_nan = np.asarray(self.cropped_atom_array.is_resolved, dtype=bool)[coord_nan].sum()
                logging.getLogger(__name__).warning(
                    f"[DEBUG_NAN] get_labels: cropped_atom_array.coord has NaN: atoms_with_nan={n_nan}, "
                    f"of_which_is_resolved={resolved_nan}"
                )

        labels["coordinate"] = torch.Tensor(coord)  # [N_atom, 3]

        labels["coordinate_mask"] = torch.Tensor(
            self.cropped_atom_array.is_resolved.astype(int)
        ).long()  # [N_atom]
        return labels

    def get_atom_permutation_list(
        self,
    ) -> list[list[int]]:
        """
        Generate info of permutations.

        Returns:
            List[List[int]]: a list of atom permutations.
        """
        atom_perm_list = []
        for i in self.cropped_atom_array.res_perm:
            # Decode list[str] -> list[list[int]]
            atom_perm_list.append([int(j) for j in i.split("_")])

        # Atoms connected to different residue are fixed.
        # Bonds array: [[atom_idx_i, atom_idx_j, bond_type]]
        idx_i = self.cropped_atom_array.bonds._bonds[:, 0]
        idx_j = self.cropped_atom_array.bonds._bonds[:, 1]
        diff_mask = (
            self.cropped_atom_array.ref_space_uid[idx_i]
            != self.cropped_atom_array.ref_space_uid[idx_j]
        )
        inter_residue_bonds = self.cropped_atom_array.bonds._bonds[diff_mask]
        fixed_atom_mask = np.isin(
            np.arange(len(self.cropped_atom_array)),
            np.unique(inter_residue_bonds[:, :2]),
        )

        # Get fixed atom permutation for each residue.
        fixed_atom_perm_list = []
        res_starts = get_residue_starts(
            self.cropped_atom_array, add_exclusive_stop=True
        )
        for r_start, r_stop in zip(res_starts[:-1], res_starts[1:]):
            atom_res_perm = np.array(
                atom_perm_list[r_start:r_stop]
            )  # [N_res_atoms, N_res_perm]
            res_fixed_atom_mask = fixed_atom_mask[r_start:r_stop]

            if np.sum(res_fixed_atom_mask) == 0:
                # If all atoms in the residue are not fixed, e.g. ions
                fixed_atom_perm_list.extend(atom_res_perm.tolist())
                continue

            # Create a [N_res_atoms, N_res_perm] template of indices
            n_res_atoms, n_perm = atom_res_perm.shape
            indices_template = (
                atom_res_perm[:, 0].reshape(n_res_atoms, 1).repeat(n_perm, axis=1)
            )

            # Identify the column where the positions of the fixed atoms remain unchanged
            fixed_atom_perm = atom_res_perm[
                res_fixed_atom_mask
            ]  # [N_fixed_res_atoms, N_res_perm]
            fixed_indices_template = indices_template[
                res_fixed_atom_mask
            ]  # [N_fixed_res_atoms, N_res_perm]
            unchanged_columns_mask = np.all(
                fixed_atom_perm == fixed_indices_template, axis=0
            )

            # Remove the columns related to the position changes of fixed atoms.
            fixed_atom_res_perm = atom_res_perm[:, unchanged_columns_mask]
            fixed_atom_perm_list.extend(fixed_atom_res_perm.tolist())
        return fixed_atom_perm_list

    @staticmethod
    def get_gt_full_complex_features(
        atom_array: AtomArray,
        cropped_atom_array: AtomArray = None,
        get_cropped_asym_only: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Get full ground truth complex features.
        It is used for multi-chain permutation alignment.

        Args:
            atom_array (AtomArray): all atoms in the complex.
            cropped_atom_array (AtomArray, optional): cropped atoms. Defaults to None.
            get_cropped_asym_only (bool, optional): Defaults to True.
                - If true, a chain is returned only if its asym_id (mol_id) appears in the
                cropped_atom_array. It should be a favored setting for the spatial cropping.
                - If false, a chain is returned if its entity_id (entity_mol_id) appears in
                the cropped_atom_array.

        Returns:
            Dict[str, torch.Tensor]: a dictionary containing
                coordinate, coordinate_mask, etc.
        """
        gt_features = {}

        if cropped_atom_array is not None:
            # Get the cropped part of gt entities
            entity_atom_set = set(
                zip(
                    cropped_atom_array.entity_mol_id,
                    cropped_atom_array.mol_atom_index,
                )
            )
            mask = [
                (entity, atom) in entity_atom_set
                for (entity, atom) in zip(
                    atom_array.entity_mol_id, atom_array.mol_atom_index
                )
            ]

            if get_cropped_asym_only:
                # Restrict to asym chains appeared in cropped_atom_array
                asyms = np.unique(cropped_atom_array.mol_id)
                mask = mask * np.isin(atom_array.mol_id, asyms)
            atom_array = atom_array[mask]

        gt_features["coordinate"] = torch.Tensor(atom_array.coord)
        gt_features["coordinate_mask"] = torch.Tensor(atom_array.is_resolved).long()
        gt_features["entity_mol_id"] = torch.Tensor(atom_array.entity_mol_id).long()
        gt_features["mol_id"] = torch.Tensor(atom_array.mol_id).long()
        gt_features["mol_atom_index"] = torch.Tensor(atom_array.mol_atom_index).long()
        gt_features["pae_rep_atom_mask"] = torch.Tensor(
            atom_array.centre_atom_mask
        ).long()
        return gt_features, atom_array
