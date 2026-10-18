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

import json
import logging
import os
import time
import traceback
import warnings
from typing import Any, Mapping

import torch
from biotite.structure import AtomArray
from torch.utils.data import DataLoader, Dataset, DistributedSampler

from protenix.data.data_pipeline import DataPipeline
from protenix.data.esm_featurizer import ESMFeaturizer
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.utils.distributed import DIST_WRAPPER
from protenix.utils.torch_utils import collate_fn_identity, dict_to_tensor

# Import CIF loader for protein ref_pos update
try:
    # update_ref_pos_from_cif is deliberately NOT imported: its three-stage guessing is no longer
    # reachable from inference. It stays in the module for its own self-test only.
    from protenix.data.cif_ref_pos_loader import (load_protein_coords_from_cif,
                                                  load_ref_map, update_ref_pos_from_ref_map)
    CIF_LOADER_AVAILABLE = True
except ImportError:
    CIF_LOADER_AVAILABLE = False
    logger.warning("CIF ref_pos loader not available")

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="biotite")


def get_inference_dataloader(configs: Any) -> DataLoader:
    """
    Creates and returns a DataLoader for inference using the InferenceDataset.

    Args:
        configs: A configuration object containing the necessary parameters for the DataLoader.

    Returns:
        A DataLoader object configured for inference.
    """
    inference_dataset = InferenceDataset(
        input_json_path=configs.input_json_path,
        configs=configs,
        dump_dir=configs.dump_dir,
        use_msa=configs.use_msa,
        cif_file_path=getattr(configs, 'cif_file_path', None),
    )
    sampler = DistributedSampler(
        dataset=inference_dataset,
        num_replicas=DIST_WRAPPER.world_size,
        rank=DIST_WRAPPER.rank,
        shuffle=False,
    )
    dataloader = DataLoader(
        dataset=inference_dataset,
        batch_size=1,
        sampler=sampler,
        collate_fn=collate_fn_identity,
        num_workers=configs.num_workers,
    )
    return dataloader


class InferenceDataset(Dataset):
    def __init__(
        self,
        input_json_path: str,
        dump_dir: str,
        use_msa: bool = True,
        configs=None,
        cif_file_path: str = None,
    ) -> None:

        self.input_json_path = input_json_path
        self.dump_dir = dump_dir
        self.use_msa = use_msa
        self.cif_file_path = cif_file_path
        self.configs = configs
        with open(self.input_json_path, "r") as f:
            self.inputs = json.load(f)
        json_task_name = os.path.basename(self.input_json_path).split(".")[0]
        esm_info = configs.get("esm", {})
        configs.esm.embedding_dir = f"./esm_embeddings/{configs.esm.model_name}"
        configs.esm.sequence_fpath = (
            f"./esm_embeddings/{json_task_name}_prot_sequences.csv"
        )
        self.esm_enable = esm_info.get("enable", False)
        if self.esm_enable:
            os.makedirs(configs.esm.embedding_dir, exist_ok=True)
            os.makedirs(os.path.dirname(configs.esm.sequence_fpath), exist_ok=True)
            ESMFeaturizer.precompute_esm_embedding(
                self.inputs,
                configs.esm.model_name,
                configs.esm.embedding_dir,
                configs.esm.sequence_fpath,
                configs.load_checkpoint_dir,
            )
            self.esm_featurizer = ESMFeaturizer(
                embedding_dir=esm_info.embedding_dir,
                sequence_fpath=esm_info.sequence_fpath,
                embedding_dim=esm_info.embedding_dim,
                error_dir="./esm_embeddings/",
            )

    def process_one(
        self,
        single_sample_dict: Mapping[str, Any],
    ) -> tuple[dict[str, torch.Tensor], AtomArray, dict[str, float]]:
        """
        Processes a single sample from the input JSON to generate features and statistics.

        Args:
            single_sample_dict: A dictionary containing the sample data.

        Returns:
            A tuple containing:
                - A dictionary of features.
                - An AtomArray object.
                - A dictionary of time tracking statistics.
        """
        # general features
        t0 = time.time()
        save_rdkit_sdf_dir = getattr(self.configs, "save_rdkit_ligand_conformer_dir", None) or None
        if save_rdkit_sdf_dir:
            save_rdkit_sdf_dir = os.path.abspath(save_rdkit_sdf_dir)
        sample_name = single_sample_dict.get("name", "")
        sample2feat = SampleDictToFeatures(
            single_sample_dict,
            save_rdkit_sdf_dir=save_rdkit_sdf_dir,
            sample_name=sample_name,
        )
        
        # Load CIF file if provided (for later use)
        cif_atom_array = None
        if self.cif_file_path and CIF_LOADER_AVAILABLE and os.path.exists(self.cif_file_path):
            logger.info(f"Loading protein coordinates from CIF file: {self.cif_file_path}")
            cif_atom_array = load_protein_coords_from_cif(self.cif_file_path)
            if cif_atom_array is not None:
                # Debug: Print CIF coordinates
                print(f"[DEBUG] CIF file loaded:")
                print(f"  CIF coords range: X=[{cif_atom_array.coord[:, 0].min():.2f}, {cif_atom_array.coord[:, 0].max():.2f}], "
                      f"Y=[{cif_atom_array.coord[:, 1].min():.2f}, {cif_atom_array.coord[:, 1].max():.2f}], "
                      f"Z=[{cif_atom_array.coord[:, 2].min():.2f}, {cif_atom_array.coord[:, 2].max():.2f}]")
                print(f"  CIF coords center: ({cif_atom_array.coord[:, 0].mean():.2f}, {cif_atom_array.coord[:, 1].mean():.2f}, {cif_atom_array.coord[:, 2].mean():.2f})")
            else:
                logger.warning(f"Failed to load CIF file {self.cif_file_path}, using CCD ref_pos")
        
        # Get features (this will apply centralization to ref_pos from CCD)
        features_dict, atom_array, token_array = sample2feat.get_feature_dict()

        # Debug: Print ref_pos after get_feature_dict (should be centralized)
        print(f"[DEBUG] ref_pos after get_feature_dict (centralized from CCD):")
        print(f"  ref_pos range: X=[{atom_array.ref_pos[:, 0].min():.2f}, {atom_array.ref_pos[:, 0].max():.2f}], "
              f"Y=[{atom_array.ref_pos[:, 1].min():.2f}, {atom_array.ref_pos[:, 1].max():.2f}], "
              f"Z=[{atom_array.ref_pos[:, 2].min():.2f}, {atom_array.ref_pos[:, 2].max():.2f}]")
        print(f"  ref_pos center: ({atom_array.ref_pos[:, 0].mean():.2f}, {atom_array.ref_pos[:, 1].mean():.2f}, {atom_array.ref_pos[:, 2].mean():.2f})")
        
        # If CIF file was provided, update ref_pos from CIF and re-apply centralization
        if cif_atom_array is not None:
            # Get fresh atom_array (before any CIF update) to update ref_pos
            # We need to get the atom_array that has the same structure but with CIF coordinates
            atom_array_for_update = sample2feat.get_atom_array()
            
            # Replace CCD ref_pos with the reference structure's global coordinates, and record
            # which atoms actually received them.
            #
            # When the protein is pinned, the correspondence MUST come from the explicit map rather
            # than be recovered by matching. The matching path guesses in three stages and, on the
            # nine of 57 fibril receptors whose protofilaments resolve different residue sets, it
            # pinned ~9 of 64 residues per chain at the WRONG residues' coordinates while marking
            # them fixed -- producing runs that completed and wrote plausible-looking output with
            # half the assembly silently wrong. A refusal to start is strictly better than that, so
            # a missing map is fatal here instead of falling back.
            # Supplying a reference at all requires supplying its correspondence -- whether or not
            # the protein is pinned. Keeping a heuristic fallback "only for the unpinned case" is how
            # this went unnoticed: the co-folding arm passed CIF_DIR too, matched ${PDBID}.cif (the
            # full 6-chain deposition) against a 10-chain input, and silently replaced ref_pos on all
            # 114 runs. Pinned or not, ref_pos is model input, and input assembled by guesswork is
            # not worth the runs spent on it.
            ref_map = load_ref_map(self.cif_file_path)
            if ref_map is None:
                raise ValueError(
                    f"A reference structure was given ({self.cif_file_path}) but no explicit "
                    f"reference map beside it (expected "
                    f"{os.path.splitext(self.cif_file_path)[0]}.refmap.json). "
                    "Generate it with scripts/make_ref_map.py. The JSON's `count: N` builds N "
                    "byte-identical copies, so per-chain residue numbering is absent by "
                    "construction and cannot be recovered from the inputs alone."
                )
            atom_array_for_update, is_structured_protein = update_ref_pos_from_ref_map(
                atom_array_for_update, cif_atom_array, ref_map
            )
            
            # ODesign-style: optionally center input frame by protein centroid (before per-residue centralization).
            # When "protein", subtract protein centroid so the diffusion sees protein near origin; loop still centers by all atoms.
            inference_center_method = getattr(self.configs, "inference_center_method", "global")
            if inference_center_method == "protein" and is_structured_protein.any():
                import numpy as np
                protein_center = np.mean(atom_array_for_update.ref_pos[is_structured_protein], axis=0)
                atom_array_for_update.ref_pos = atom_array_for_update.ref_pos - protein_center
                logger.info(
                    f"Applied protein-center input frame (inference_center_method=protein): "
                    f"center=({protein_center[0]:.2f}, {protein_center[1]:.2f}, {protein_center[2]:.2f})"
                )
            
            # Debug: Print ref_pos after CIF update (before centralization)
            print(f"[DEBUG] ref_pos after CIF update (before centralization):")
            print(f"  ref_pos range: X=[{atom_array_for_update.ref_pos[:, 0].min():.2f}, {atom_array_for_update.ref_pos[:, 0].max():.2f}], "
                  f"Y=[{atom_array_for_update.ref_pos[:, 1].min():.2f}, {atom_array_for_update.ref_pos[:, 1].max():.2f}], "
                  f"Z=[{atom_array_for_update.ref_pos[:, 2].min():.2f}, {atom_array_for_update.ref_pos[:, 2].max():.2f}]")
            print(f"  ref_pos center: ({atom_array_for_update.ref_pos[:, 0].mean():.2f}, {atom_array_for_update.ref_pos[:, 1].mean():.2f}, {atom_array_for_update.ref_pos[:, 2].mean():.2f})")
            
            # Save global coordinates for diffusion initialization (before centralization)
            global_ref_pos = atom_array_for_update.ref_pos.copy()  # [N_atom, 3] - global coordinates
            features_dict["global_ref_pos"] = torch.Tensor(global_ref_pos)
            
            # Save mask indicating which atoms have coordinates from CIF (structured protein atoms)
            features_dict["is_structured_protein"] = torch.Tensor(is_structured_protein).bool()
            
            # Apply centralization to ref_pos for conditioning input (as in training)
            # ref_pos should be per-residue centralized (local coordinates) for network conditioning
            # This matches the training data format where ref_pos is used as features
            from protenix.utils.geometry import random_transform
            import numpy as np
            
            if "ref_space_uid" in atom_array_for_update.get_annotation_categories():
                ref_space_uid = atom_array_for_update.ref_space_uid
                centralized_ref_pos = []
                for uid in np.unique(ref_space_uid):
                    mask = ref_space_uid == uid
                    res_ref_pos = atom_array_for_update.ref_pos[mask]
                    # Apply centralization (local coordinates) for conditioning input
                    centralized_res_ref_pos = random_transform(
                        res_ref_pos,
                        apply_augmentation=False,  # No augmentation during inference
                        centralize=True,  # Per-residue centralization (local coordinates)
                    )
                    centralized_ref_pos.append(centralized_res_ref_pos)
                atom_array_for_update.ref_pos = np.concatenate(centralized_ref_pos)
                
                # Update ref_pos in features_dict (local coordinates for conditioning)
                features_dict["ref_pos"] = torch.Tensor(atom_array_for_update.ref_pos)
                
                # Update atom_array for consistency
                atom_array = atom_array_for_update
                
                # Re-compute token frames with centralized ref_pos (local coordinates)
                from protenix.data.featurizer import Featurizer
                token_array_with_frame = Featurizer.get_token_frame(
                    token_array=token_array,
                    atom_array=atom_array,
                    ref_pos=features_dict["ref_pos"],
                    ref_mask=features_dict["ref_mask"],
                )
                features_dict["has_frame"] = torch.Tensor(
                    token_array_with_frame.get_annotation("has_frame")
                ).long()
                features_dict["frame_atom_index"] = torch.Tensor(
                    token_array_with_frame.get_annotation("frame_atom_index")
                ).long()
                
                # Debug: Print both coordinate systems
                print(f"[DEBUG] Coordinate systems:")
                print(f"  global_ref_pos (for diffusion init) range: X=[{global_ref_pos[:, 0].min():.2f}, {global_ref_pos[:, 0].max():.2f}], "
                      f"Y=[{global_ref_pos[:, 1].min():.2f}, {global_ref_pos[:, 1].max():.2f}], "
                      f"Z=[{global_ref_pos[:, 2].min():.2f}, {global_ref_pos[:, 2].max():.2f}]")
                print(f"  global_ref_pos center: ({global_ref_pos[:, 0].mean():.2f}, {global_ref_pos[:, 1].mean():.2f}, {global_ref_pos[:, 2].mean():.2f})")
                print(f"  ref_pos (for conditioning, local) range: X=[{atom_array.ref_pos[:, 0].min():.2f}, {atom_array.ref_pos[:, 0].max():.2f}], "
                      f"Y=[{atom_array.ref_pos[:, 1].min():.2f}, {atom_array.ref_pos[:, 1].max():.2f}], "
                      f"Z=[{atom_array.ref_pos[:, 2].min():.2f}, {atom_array.ref_pos[:, 2].max():.2f}]")
                print(f"  ref_pos center: ({atom_array.ref_pos[:, 0].mean():.2f}, {atom_array.ref_pos[:, 1].mean():.2f}, {atom_array.ref_pos[:, 2].mean():.2f})")
                
                logger.info("Updated protein coordinates: global_ref_pos for diffusion init, ref_pos (local) for conditioning")
            else:
                logger.warning("ref_space_uid not found, skipping centralization")
        
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array.distogram_rep_atom_mask
        ).long()
        entity_poly_type = sample2feat.entity_poly_type
        t1 = time.time()

        # Msa features
        entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array)
        msa_features = (
            InferenceMSAFeaturizer.make_msa_feature(
                bioassembly=single_sample_dict["sequences"],
                entity_to_asym_id=entity_to_asym_id,
                token_array=token_array,
                atom_array=atom_array,
            )
            if self.use_msa
            else {}
        )

        # Esm features
        if self.esm_enable:
            x_esm = self.esm_featurizer(
                token_array=token_array,
                atom_array=atom_array,
                bioassembly_dict=single_sample_dict,
                inference_mode=True,
            )
            features_dict["esm_token_embedding"] = x_esm

        # Make dummy features for not implemented features
        dummy_feats = ["template"]
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        # Transform to right data type
        feat = data_type_transform(feat_or_label_dict=features_dict)

        t2 = time.time()

        data = {}
        data["input_feature_dict"] = feat

        # Add dimension related items
        N_token = feat["token_index"].shape[0]
        N_atom = feat["atom_to_token_idx"].shape[0]
        N_msa = feat["msa"].shape[0]

        stats = {}
        for mol_type in ["ligand", "protein", "dna", "rna"]:
            mol_type_mask = feat[f"is_{mol_type}"].bool()
            stats[f"{mol_type}/atom"] = int(mol_type_mask.sum(dim=-1).item())
            stats[f"{mol_type}/token"] = len(
                torch.unique(feat["atom_to_token_idx"][mol_type_mask])
            )

        N_asym = len(torch.unique(data["input_feature_dict"]["asym_id"]))
        data.update(
            {
                "N_asym": torch.tensor([N_asym]),
                "N_token": torch.tensor([N_token]),
                "N_atom": torch.tensor([N_atom]),
                "N_msa": torch.tensor([N_msa]),
            }
        )

        def formatted_key(key):
            type_, unit = key.split("/")
            if type_ == "protein":
                type_ = "prot"
            elif type_ == "ligand":
                type_ = "lig"
            else:
                pass
            return f"N_{type_}_{unit}"

        data.update(
            {
                formatted_key(k): torch.tensor([stats[k]])
                for k in [
                    "protein/atom",
                    "ligand/atom",
                    "dna/atom",
                    "rna/atom",
                    "protein/token",
                    "ligand/token",
                    "dna/token",
                    "rna/token",
                ]
            }
        )
        data.update({"entity_poly_type": entity_poly_type})
        t3 = time.time()
        # Ensure bonds on atom_array for ligand SDF saving in inference dumper.
        # Do not use the CONECT-appending-to-PDB strategy from train_nft; inference saves SDF via save_ligand_sdf.
        if atom_array is not None and (not hasattr(atom_array, "bonds") or atom_array.bonds is None):
            try:
                aa_with_bonds = sample2feat.get_atom_array()
                if hasattr(aa_with_bonds, "bonds") and aa_with_bonds.bonds is not None and len(aa_with_bonds) == len(atom_array):
                    atom_array.bonds = aa_with_bonds.bonds
            except Exception:
                pass

        time_tracker = {
            "crop": t1 - t0,
            "featurizer": t2 - t1,
            "added_feature": t3 - t2,
        }

        return data, atom_array, time_tracker

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], AtomArray, str]:
        try:
            single_sample_dict = self.inputs[index]
            sample_name = single_sample_dict["name"]
            logger.info(f"Featurizing {sample_name}...")

            data, atom_array, _ = self.process_one(
                single_sample_dict=single_sample_dict
            )
            error_message = ""
        except Exception as e:
            data, atom_array = {}, None
            error_message = f"{e}:\n{traceback.format_exc()}"
        data["sample_name"] = single_sample_dict["name"]
        data["sample_index"] = index
        return data, atom_array, error_message
