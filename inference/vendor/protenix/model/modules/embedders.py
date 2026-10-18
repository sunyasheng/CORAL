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

from typing import Any, Optional, Union

import os
import torch

from protenix.utils.logger import get_logger

logger = get_logger(__name__)
import torch.nn as nn
import torch.nn.functional as F

from protenix.model.modules.primitives import LinearNoBias
from protenix.model.modules.transformer import AtomAttentionEncoder


class InputFeatureEmbedder(nn.Module):
    """
    Implements Algorithm 2 in AF3
    """

    def __init__(
        self,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 384,
        esm_configs: dict = {},
    ) -> None:
        """
        Args:
            c_atom (int, optional): atom embedding dim. Defaults to 128.
            c_atompair (int, optional): atom pair embedding dim. Defaults to 16.
            c_token (int, optional): token embedding dim. Defaults to 384.
        """
        super(InputFeatureEmbedder, self).__init__()
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.atom_attention_encoder = AtomAttentionEncoder(
            c_atom=c_atom,
            c_atompair=c_atompair,
            c_token=c_token,
            has_coords=False,
        )

        self.esm_configs = {
            "enable": esm_configs.get("enable", False),
            "embedding_dim": esm_configs.get("embedding_dim", 2560),
        }
        if self.esm_configs["enable"]:
            self.linear_esm = LinearNoBias(
                self.esm_configs["embedding_dim"],
                self.c_token + 32 + 32 + 1,
            )
            nn.init.zeros_(self.linear_esm.weight)

        # Line2
        self.input_feature = {"restype": 32, "profile": 32, "deletion_mean": 1}

    def forward(
        self,
        input_feature_dict: dict[str, Any],
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): dict of input features
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: token embedding
                [..., N_token, 384 (c_token) + 32 + 32 + 1 :=449]
        """
        # Embed per-atom features.
        a, _, _, _ = self.atom_attention_encoder(
            input_feature_dict["atom_to_token_idx"],
            input_feature_dict["ref_pos"],
            input_feature_dict["ref_charge"],
            input_feature_dict["ref_mask"],
            input_feature_dict["ref_atom_name_chars"],
            input_feature_dict["ref_element"],
            input_feature_dict["d_lm"],
            input_feature_dict["v_lm"],
            input_feature_dict["pad_info"],
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )  # [..., N_token, c_token]
        # Concatenate the per-token features.
        batch_shape = input_feature_dict["restype"].shape[:-1]
        s_inputs = torch.cat(
            [a]
            + [
                input_feature_dict[name].reshape(*batch_shape, d)
                for name, d in self.input_feature.items()
            ],
            dim=-1,
        )

        if self.esm_configs["enable"]:
            # Add esm embedding to s_inputs if enable.
            esm_embeddings = self.linear_esm(input_feature_dict["esm_token_embedding"])
            s_inputs = s_inputs + esm_embeddings

        return s_inputs




_SYM_MODE_WARNED = set()


def _pf_rewrite_sym_id(sym_id, input_feature_dict, mode):
    """Rewrite protein sym_id at inference. Off unless PF_SYM_MODE is set.

    Exists to test whether sym_id matters at all -- the official comment at
    dataset.py "Lig_atom_rename/shuffle_mols/shuffle_sym_ids do not affect the performance
    very much" says it may not, and an earlier ablation that claimed otherwise turned out to
    be a no-op (its env var was never read by any code, so both arms were plain replicates).
    Hence: warn loudly when active, and report what was actually rewritten.

    modes
      collapse  every protein chain gets sym_id 0 -> d_chain carries no copy identity at all
      shuffle   permute sym_id among protein chains, deterministic given the chain count
      offset:N  add N to the second half of the protein chains (the other protofilament)
      stride    sym_id = 2 * layer + protofilament, so same-PF offsets are even and
                cross-PF odd -- parity separates them with no artificial gap

    Ligand tokens are never touched: their sym_id is an independent 0-based index, so
    selecting by value would silently renumber them too.
    """
    a2t = input_feature_dict.get("atom_to_token_idx", None)
    is_lig = input_feature_dict.get("is_ligand", None)
    asym = input_feature_dict.get("asym_id", None)
    if a2t is None or is_lig is None or asym is None:
        return sym_id
    tok_lig = torch.zeros(sym_id.shape[-1], dtype=torch.bool, device=sym_id.device)
    idx = a2t.reshape(-1).long()[is_lig.reshape(-1).bool()]
    if idx.numel():
        tok_lig[idx] = True
    prot_asym = torch.unique(asym.reshape(-1)[~tok_lig])
    n = prot_asym.numel()
    if n < 2:
        return sym_id

    out = sym_id.clone()
    flat_asym = asym.reshape(-1)
    half = n // 2
    if mode == "collapse":
        new_of = {int(a): 0 for a in prot_asym.tolist()}
    elif mode == "shuffle":
        # deterministic permutation so replicates of this arm are comparable
        g = torch.Generator().manual_seed(1234 + n)
        perm = torch.randperm(n, generator=g).tolist()
        new_of = {int(a): perm[k] for k, a in enumerate(prot_asym.tolist())}
    elif mode.startswith("offset"):
        N = int(mode.split(":")[1]) if ":" in mode else 100
        new_of = {int(a): (k if k < half else (k - half) + N)
                  for k, a in enumerate(prot_asym.tolist())}
    elif mode == "stride":
        # chain order groups the protofilaments (verified 57/57: pattern 0000011111)
        new_of = {int(a): (2 * k if k < half else 2 * (k - half) + 1)
                  for k, a in enumerate(prot_asym.tolist())}
    else:
        return sym_id

    for a, v in new_of.items():
        m = (~tok_lig) & (flat_asym == a)
        out[..., m] = v
    if mode not in _SYM_MODE_WARNED:
        _SYM_MODE_WARNED.add(mode)
        # log the MAPPING, not sorted(values): a permutation leaves the value set unchanged,
        # so sorted(values) is always [0..n-1] and hides whether anything happened.
        logger.warning("PF_SYM_MODE=%s ACTIVE: %d protein chains, asym->sym %s "
                       "(ligands untouched). Diagnostic intervention; record it with any result.",
                       mode, n, [(a, new_of[a]) for a in sorted(new_of)])
    return out



def _pf_shift_sym_id(sym_id, input_feature_dict, offset):
    """Add `offset` to the sym_id of the second half of the PROTEIN chains.

    Chain order groups the protofilaments in our receptors (verified 57/57: the membership
    pattern is 0000011111), so "second half of the protein asym_ids" is the second
    protofilament. Ligand tokens are left untouched.
    """
    a2t = input_feature_dict.get("atom_to_token_idx", None)
    is_lig = input_feature_dict.get("is_ligand", None)
    asym = input_feature_dict.get("asym_id", None)
    if a2t is None or is_lig is None or asym is None:
        return sym_id
    tok_lig = torch.zeros(sym_id.shape[-1], dtype=torch.bool, device=sym_id.device)
    idx = a2t.reshape(-1).long()[is_lig.reshape(-1).bool()]
    if idx.numel():
        tok_lig[idx] = True
    prot_asym = torch.unique(asym.reshape(-1)[~tok_lig])
    if prot_asym.numel() < 2:
        return sym_id
    pf1 = prot_asym[prot_asym.numel() // 2:]
    shift = (~tok_lig) & torch.isin(asym.reshape(-1), pf1)
    out = sym_id.clone()
    out[..., shift] = out[..., shift] + offset
    return out


class RelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3
    """

    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128, layer_max=None,
                 use_pf_flag: bool = False, use_pf_level: bool = False) -> None:
        """
        Args:
            r_max (int, optional): Relative position indices clip value. Defaults to 32.
            s_max (int, optional): Relative chain indices clip value. Defaults to 2.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        """
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        # b_diff_pf is appended LAST so every pre-existing column keeps its index and the
        # checkpoint can be widened by appending one zero column -- no reindexing, and the
        # forward pass is bit-identical at init.
        self.use_pf_flag = use_pf_flag
        # use_pf_level replaces the flat b_diff_pf column with a hierarchical
        # entity -> protofilament -> layer bucket structure; see the comment at d_chain. The two
        # are mutually exclusive -- the level scheme already distinguishes same/different
        # protofilament, so an extra flag column would be redundant.
        self.use_pf_level = use_pf_level
        assert not (use_pf_flag and use_pf_level), "use_pf_flag and use_pf_level are exclusive"
        # layer_max decouples the LAYER clip from s_max. s_max also indexes the official
        # signed-Delta-sym buckets, so raising it would rewrite their meaning for every dataset,
        # including non-fibril data the released weights were trained on. layer_max touches only the
        # appended pf/layer buckets. Default = s_max keeps the previous 145-wide layout bit-identical.
        # Training set is 3-6 layers, so layer_max = 5 clips nothing.
        self.layer_max = int(layer_max) if layer_max is not None else int(s_max)
        n_chain_bucket = (
            (2 * self.s_max + 2) + 2 * (self.layer_max + 1)
            if use_pf_level
            else (2 * self.s_max + 2)
        )
        self.linear_no_bias = LinearNoBias(
            in_features=(4 * self.r_max + 5 + n_chain_bucket + int(use_pf_flag)),
            out_features=self.c_z,
        )
        self.input_feature = {
            "asym_id": 1,
            "residue_index": 1,
            "entity_id": 1,
            "sym_id": 1,
            "token_index": 1,
        }

    def generate_relp(self, input_feature_dict: dict[str, Any]) -> dict[str, Any]:
        """asym_id, residue_index, entity_id, token_index, sym_id"""
        with torch.no_grad():
            asym_id = input_feature_dict["asym_id"]
            residue_index = input_feature_dict["residue_index"]
            entity_id = input_feature_dict["entity_id"]
            token_index = input_feature_dict["token_index"]
            sym_id = input_feature_dict["sym_id"]

            # Optional protofilament-aware sym_id offset (no-op unless PF_SYM_OFFSET is set).
            #
            # d_chain = clip(sym_id[i]-sym_id[j]+s_max, ...) collides for us: with sym_id 0..9
            # and PF0={0..4}, PF1={5..9}, the cross-PF offsets {1..9} strictly CONTAIN the
            # within-PF offsets {1..4}, so chain E <-> chain F (delta=1) shares a bucket with a
            # genuine adjacent layer while sitting ~66 A apart. No value of s_max fixes that --
            # the collision is in the numbering, not the clipping (verified up to s_max=100).
            # Offsetting the second protofilament by a constant separates them: within-PF stays
            # {1..4} while cross-PF becomes {96..104}, which needs s_max>=5 to stay collision
            # free after clipping.
            #
            # Only PROTEIN chains are offset. Ligand copies carry their own sym_id starting at 0,
            # so selecting by sym_id value would silently renumber them too -- their entity split
            # is a separate problem.
            _mode = os.environ.get("PF_SYM_MODE", "")
            if _mode:
                sym_id = _pf_rewrite_sym_id(sym_id, input_feature_dict, _mode)
            else:
                _pf_off = os.environ.get("PF_SYM_OFFSET", "")
                if _pf_off:
                    sym_id = _pf_shift_sym_id(sym_id, input_feature_dict, int(_pf_off))

            b_same_chain = (
                asym_id[..., :, None] == asym_id[..., None, :]
            ).long()  # [..., N_token, N_token]
            b_same_residue = (
                residue_index[..., :, None] == residue_index[..., None, :]
            ).long()  # [..., N_token, N_token]
            b_same_entity = (
                entity_id[..., :, None] == entity_id[..., None, :]
            ).long()  # [..., N_token, N_token]
            d_residue = torch.clip(
                input=residue_index[..., :, None]
                - residue_index[..., None, :]
                + self.r_max,
                min=0,
                max=2 * self.r_max,
            ) * b_same_chain + (1 - b_same_chain) * (
                2 * self.r_max + 1
            )  # [..., N_token, N_token]
            a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))
            d_token = torch.clip(
                input=token_index[..., :, None]
                - token_index[..., None, :]
                + self.r_max,
                min=0,
                max=2 * self.r_max,
            ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
                2 * self.r_max + 1
            )  # [..., N_token, N_token]
            a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))
            d_chain = torch.clip(
                input=sym_id[..., :, None] - sym_id[..., None, :] + self.s_max,
                min=0,
                max=2 * self.s_max,
            ) * b_same_entity + (1 - b_same_entity) * (
                2 * self.s_max + 1
            )  # [..., N_token, N_token]
            n_chain_bucket = 2 * (self.s_max + 1)
            if self.use_pf_level:
                # Hierarchical entity -> protofilament -> layer, in the same idiom the official
                # code already uses for the entity level: a gated relative index plus dedicated
                # buckets for the "not applicable" relation.
                #
                #   0 .. 2*s_max          official signed Delta sym_id  (kept verbatim, and the
                #                         ONLY path taken when pf_id is absent, so non-fibril
                #                         samples stay bit-identical to the released weights)
                #   2*s_max+1             different entity              (kept verbatim)
                #   2*s_max+2 .. 3*s_max+2   same entity, SAME protofilament, clip(|Delta layer|)
                #   3*s_max+3 .. 4*s_max+3   same entity, DIFF protofilament, clip(|Delta layer|)
                #
                # Why |Delta layer| and not the signed difference: which end of a protofilament
                # is layer 0 is consistent WITHIN a structure (rotation-invariant on 57/57) but
                # agrees across structures only 51% of the time against the chain's own principal
                # frame -- a coin flip, measured, see dataset._annotate_protofilament. A signed
                # index would train half the samples on the opposite convention. Why cross-protofilament
                # pairs carry |Delta layer| too, instead of collapsing to one bucket: collapsing
                # lets the two protofilaments be reversed INDEPENDENTLY, which the encoding then
                # cannot distinguish from the same-polarity arrangement real C2-related fibrils
                # have. Keeping it makes the permutation symmetry group exactly
                # {identity, whole-stack reversal} x {protofilament swap} = 4 elements, and tells
                # the model which layer faces which across the interface.
                base_same = 2 * self.s_max + 2
                base_diff = base_same + (self.layer_max + 1)
                n_chain_bucket = base_diff + (self.layer_max + 1)
                pf = input_feature_dict.get("pf_id", None)
                layer = input_feature_dict.get("layer_id", None)
                if pf is not None and layer is not None:
                    pf = pf.to(sym_id.device).long()
                    layer = layer.to(sym_id.device).long()
                    both = (pf > 0)[..., :, None] & (pf > 0)[..., None, :]
                    same_pf = pf[..., :, None] == pf[..., None, :]
                    a_layer = torch.clip(
                        (layer[..., :, None] - layer[..., None, :]).abs(),
                        min=0,
                        max=self.layer_max,
                    )
                    pf_bucket = torch.where(
                        same_pf, base_same + a_layer, base_diff + a_layer
                    )
                    # only where BOTH ends have a protofilament AND they share an entity;
                    # everything else keeps the official bucket it already had
                    use = both & b_same_entity.bool()
                    d_chain = torch.where(use, pf_bucket, d_chain)
            a_rel_chain = F.one_hot(d_chain, n_chain_bucket)

            parts = [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain]
            if self.use_pf_flag:
                # A DIFFERENCE flag, not a same-flag: absent pf_id -> all zeros -> this column
                # contributes exactly nothing, so every non-fibril sample is untouched and the
                # new weight only receives gradient where protofilaments actually exist.
                # Chain ORDER cannot be used to infer this: 54% of our training bioassemblies
                # interleave the two protofilaments (010101) while 100% of the inference
                # receptors are blocked (0000011111), so pf_id must come from geometry.
                pf = input_feature_dict.get("pf_id", None)
                if pf is None:
                    b_diff_pf = torch.zeros_like(b_same_entity)
                else:
                    pf = pf.to(sym_id.device).long()
                    # pf_id is three-state: 1/2 = the two protofilaments, 0 = not applicable
                    # (ligands, or detection failed). Fire only when BOTH ends are assigned,
                    # so ligand-protein pairs never claim a protofilament relationship.
                    ok = (pf > 0)
                    b_diff_pf = (
                        (pf[..., :, None] != pf[..., None, :])
                        & ok[..., :, None]
                        & ok[..., None, :]
                    ).long()
                parts.append(b_diff_pf[..., None])
            relp = torch.cat(parts, dim=-1).float()
            input_feature_dict["relp"] = relp
        return input_feature_dict

    def forward(self, relp_feature) -> torch.Tensor:
        """
        Args:
            asym_id / residue_index / entity_id / sym_id / token_index
                [..., N_tokens]
        Returns:
            torch.Tensor: relative position encoding
                [..., N_token, N_token, c_z]
        """
        return self.linear_no_bias(relp_feature)


class FourierEmbedding(nn.Module):
    """
    Implements Algorithm 22 in AF3
    """

    def __init__(self, c: int, seed: int = 42) -> None:
        """
        Args:
            c (int): embedding dim.
        """
        super(FourierEmbedding, self).__init__()
        self.c = c
        self.seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)
        w_value = torch.randn(size=(c,), generator=generator)
        self.w = nn.Parameter(w_value, requires_grad=False)
        b_value = torch.randn(size=(c,), generator=generator)
        self.b = nn.Parameter(b_value, requires_grad=False)

    def forward(self, t_hat_noise_level: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]

        Returns:
            torch.Tensor: the output fourier embedding
                [..., N_sample, c]
        """
        return torch.cos(
            input=2 * torch.pi * (t_hat_noise_level.unsqueeze(dim=-1) * self.w + self.b)
        )


class SubstructureEmbedder(nn.Module):
    """
    Implements Substructure Embedder
    """

    def __init__(
        self,
        n_classes: int,
        c_pair_dim: int,
        architecture: str = "mlp",
        hidden_dim: int = 256,
        n_layers: int = 3,
        dropout: float = 0.1,
    ) -> None:
        """
        Args:
            n_classes (int): Number of distance classes in input
            c_pair_dim (int): Output pair embedding dimension
            architecture (str): Either 'mlp' or 'transformer'
            hidden_dim (int): Hidden dimension for both architectures
            n_layers (int): Number of layers (MLP or Transformer)
            dropout (float): Dropout rate
        """
        super().__init__()
        self.architecture = architecture.lower()
        if self.architecture == "mlp":
            layers = []
            layers.append(LinearNoBias(n_classes, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            for _ in range(n_layers - 2):
                layers.append(LinearNoBias(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))

            # Output layer
            layers.append(LinearNoBias(hidden_dim, c_pair_dim))

            self.network = nn.Sequential(*layers)

        elif self.architecture == "transformer":
            self.input_proj = LinearNoBias(n_classes, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.output_proj = LinearNoBias(hidden_dim, c_pair_dim)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input distance map
                shape: [..., N_token, N_token, N_classes]
        Returns:
            torch.Tensor: Output pair embeddings
                shape: [..., N_token, N_token, c_pair_dim]
        """
        if self.architecture == "mlp":
            return self.network(x)
        else:  # transformer
            # Reshape for transformer
            orig_shape = x.shape
            x = x.view(-1, orig_shape[-3], orig_shape[-2], orig_shape[-1])
            batch_size, n_token, _, n_classes = x.shape

            # Project and reshape to sequence
            x = self.input_proj(x)
            x = x.reshape(batch_size, n_token * n_token, -1)

            # Apply transformer
            x = self.transformer(x)

            # Project to output dim and reshape back
            x = self.output_proj(x)
            x = x.reshape(*orig_shape[:-1], -1)

            return x


class ConstraintEmbedder(nn.Module):
    """
    Implements Constraint Embedder
    """

    def __init__(
        self,
        pocket_embedder: dict[str:int],
        contact_embedder: dict[str:int],
        contact_atom_embedder: dict[str:int],
        substructure_embedder: dict[str:int],
        c_constraint_z: int,
        initialize_method: str = "zero",
        **kwarg,
    ) -> None:
        """

        Args:
            pocket_embedder (dict[str:int]): pocket embedder config
            contact_embedder (dict[str:int]): contact embedder config
            contact_atom_embedder (dict[str:int]): contact atom embedder config
            substructure_embedder (dict[str:int]): substructure embedder config
            c_constraint_z (int): constraint z dimension
            initialize_method (str): initialize method
        """
        super(ConstraintEmbedder, self).__init__()
        self.pocket_embedder_config = pocket_embedder
        self.contact_embedder_config = contact_embedder
        self.contact_atom_embedder_config = contact_atom_embedder
        self.substructure_embedder_config = substructure_embedder
        # pocket embedder
        if self.pocket_embedder_config.get("enable", False):
            self.pocket_z_embedder = LinearNoBias(
                in_features=self.pocket_embedder_config.get("c_z_input", 1),
                out_features=c_constraint_z,
            )

        # token contact embedder
        if self.contact_embedder_config.get("enable", False):
            self.contact_z_embedder = LinearNoBias(
                in_features=contact_embedder["c_z_input"], out_features=c_constraint_z
            )

        # atom contact embedder
        if self.contact_atom_embedder_config.get("enable", False):
            self.contact_atom_z_embedder = LinearNoBias(
                in_features=contact_atom_embedder["c_z_input"],
                out_features=c_constraint_z,
            )

        # substructure embedder
        #
        if self.substructure_embedder_config.get("enable", False):
            self.substructure_z_embedder = SubstructureEmbedder(
                n_classes=self.substructure_embedder_config.get("n_classes", 4),
                c_pair_dim=c_constraint_z,
                architecture=self.substructure_embedder_config.get(
                    "architecture", "mlp"
                ),
                hidden_dim=self.substructure_embedder_config.get("hidden_dim", 256),
                n_layers=self.substructure_embedder_config.get("n_layers", 3),
            )

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if initialize_method == "zero":
                    nn.init.zeros_(module.weight)

    def forward(
        self,
        constraint_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
    ) -> torch.Tensor:
        """

        Args:
            constraint_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): dict of input features

        Returns:
            torch.Tensor: token embedding
                [..., N_token, c_s]
        """
        z_constraint = None

        # if  constraint_feature_dict["pocket"].max().item() > 0:
        #     import pdb; pdb.set_trace()  # pocket-constraint has non-zero entries
        if self.pocket_embedder_config.get("enable", False):
            z_constraint = self.pocket_z_embedder(constraint_feature_dict["pocket"])

        if self.contact_embedder_config.get("enable", False):
            z_contact = self.contact_z_embedder(constraint_feature_dict["contact"])
            z_constraint = (
                z_contact if z_constraint is None else z_constraint + z_contact
            )

        if self.contact_atom_embedder_config.get("enable", False):
            z_contact_atom = self.contact_atom_z_embedder(
                constraint_feature_dict["contact_atom"]
            )
            z_constraint = (
                z_contact_atom
                if z_constraint is None
                else z_constraint + z_contact_atom
            )

        # substructure embedder
        # SUBSTRUCT_OFF=1 drops this term at inference (module still constructed, so the checkpoint
        # loads unchanged). Diagnostic, not a fix: the data side sets substructure.prob = 0.0, so the
        # feature is always the dummy tensor, yet the addition is gated only on `enable` -- a zero
        # input through a transformer with biases and LayerNorm still emits a non-zero z, its bias
        # path still gets gradient, and those 14 parameters sit in the 0.0018 finetune group (18x the
        # 1e-4 used elsewhere). Measured base -> step 249: self_attn.in_proj_bias moved 4.30x its own
        # norm while input_proj.weight, the only one consuming the input, moved 0.05. Downstream
        # z_constraint is multiplied by constraint_weight, which inference sets to 20.
        if self.substructure_embedder_config.get("enable", False) and \
                os.environ.get("SUBSTRUCT_OFF", "") != "1":
            z_substructure = self.substructure_z_embedder(
                constraint_feature_dict["substructure"]
            )
            z_constraint = (
                z_substructure
                if z_constraint is None
                else z_constraint + z_substructure
            )
        return z_constraint
