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

from typing import Any, Callable, Optional
import logging
import pdb

import torch
import os

from protenix.model.utils import (
    centre_random_augmentation_with_params,
    centre_random_augmentation_rtc,
    reverse_centre_random_augmentation_rtc,
    apply_augmentation_params,
)

logger = logging.getLogger(__name__)


def check_condition_atom_coords(x_denoised, x_gt, condition_mask, threshold=5e-3):
    """
    验证条件原子的坐标是否与 ground truth 匹配
    
    Args:
        x_denoised: 去噪后的坐标 [..., N_atom, 3]
        x_gt: ground truth 坐标 [N_atom, 3] 或 [..., N_atom, 3]
        condition_mask: 条件掩码 [..., N_atom]
        threshold: 允许的最大差异（默认 0.005 Å）
    """
    if condition_mask.any():
        # 确保 x_gt 和 x_denoised 的维度匹配
        # x_denoised 可能是 [chunk_n_sample, N_atom, 3] 或 [..., chunk_n_sample, N_atom, 3]
        # x_gt 可能是 [N_atom, 3] 或 [..., chunk_n_sample, N_atom, 3]
        
        x_denoised_shape = x_denoised.shape
        x_gt_shape = x_gt.shape
        
        # 如果维度不匹配，扩展 x_gt
        if x_gt.dim() < x_denoised.dim():
            # x_gt 维度更少，需要扩展
            x_gt_expanded = x_gt
            # 在开头添加维度直到匹配
            while x_gt_expanded.dim() < x_denoised.dim():
                x_gt_expanded = x_gt_expanded.unsqueeze(0)
            # 扩展到匹配的形状（除了最后两个维度 N_atom 和 3）
            expand_shape = x_denoised_shape[:-2] + (-1, -1)
            x_gt_expanded = x_gt_expanded.expand(*expand_shape)
        elif x_gt.dim() == x_denoised.dim():
            # 维度相同，检查形状是否匹配
            if x_gt_shape != x_denoised_shape:
                # 形状不匹配，尝试扩展
                # 检查是否可以广播
                if x_gt_shape[-2:] == x_denoised_shape[-2:]:  # 最后两个维度匹配
                    expand_shape = x_denoised_shape[:-2] + (-1, -1)
                    x_gt_expanded = x_gt.expand(*expand_shape)
                else:
                    x_gt_expanded = x_gt
            else:
                x_gt_expanded = x_gt
        else:
            # x_gt 维度更多，直接使用（可能已经是正确的形状）
            x_gt_expanded = x_gt
        
        # 计算条件原子的坐标差异
        diff = (x_denoised[condition_mask] - x_gt_expanded[condition_mask]).abs()
        
        # 检查最大差异
        max_diff = diff.max().item()
        if max_diff > threshold:
            raise ValueError(
                f"Condition atom coords set error: max difference = {max_diff:.6f} Å "
                f"(threshold = {threshold:.6f} Å)"
            )


class TrainingNoiseSampler:
    """
    Sample the noise-level of of training samples
    """

    def __init__(
        self,
        p_mean: float = -1.2,
        p_std: float = 1.5,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Sampler for training noise-level

        Args:
            p_mean (float, optional): gaussian mean. Defaults to -1.2.
            p_std (float, optional): gaussian std. Defaults to 1.5.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.p_mean = p_mean
        self.p_std = p_std
        print(f"train scheduler {self.sigma_data}")

    def __call__(
        self, size: torch.Size, device: torch.device = torch.device("cpu")
    ) -> torch.Tensor:
        """Sampling

        Args:
            size (torch.Size): the target size
            device (torch.device, optional): target device. Defaults to torch.device("cpu").

        Returns:
            torch.Tensor: sampled noise-level
        """
        rnd_normal = torch.randn(size=size, device=device)
        noise_level = (rnd_normal * self.p_std + self.p_mean).exp() * self.sigma_data
        return noise_level


class InferenceNoiseScheduler:
    """
    Scheduler for noise-level (time steps)
    """

    def __init__(
        self,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        rho: float = 7,
        sigma_data: float = 16.0,  # NOTE: in EDM, this is 1.0
    ) -> None:
        """Scheduler parameters

        Args:
            s_max (float, optional): maximal noise level. Defaults to 160.0.
            s_min (float, optional): minimal noise level. Defaults to 4e-4.
            rho (float, optional): the exponent numerical part. Defaults to 7.
            sigma_data (float, optional): scale. Defaults to 16.0, but this is 1.0 in EDM.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.rho = rho
        print(f"inference scheduler {self.sigma_data}")

    def __call__(
        self,
        N_step: int = 200,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Schedule the noise-level (time steps). No sampling is performed.

        Args:
            N_step (int, optional): number of time steps. Defaults to 200.
            device (torch.device, optional): target device. Defaults to torch.device("cpu").
            dtype (torch.dtype, optional): target dtype. Defaults to torch.float32.

        Returns:
            torch.Tensor: noise-level (time_steps)
                [N_step+1]
        """
        step_size = 1 / N_step
        step_indices = torch.arange(N_step + 1, device=device, dtype=dtype)
        t_step_list = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.rho)
                + step_indices
                * step_size
                * (self.s_min ** (1 / self.rho) - self.s_max ** (1 / self.rho))
            )
            ** self.rho
        )
        # replace the last time step by 0
        t_step_list[..., -1] = 0  # t_N = 0

        return t_step_list


_PF_REP_WARNED = False


def _pf_repulsion(x_l, input_feature_dict, step_idx, n_steps):
    """Externally push the two emergent protofilaments apart during sampling.

    Why this exists outside the model. None of the four constraint channels can express
    "keep these two groups 47 A apart": contact/pocket have only an attraction column
    (the min_distance column is exactly zero in protenix_base_constraint_v0.5.0), and both
    channels feed a LinearNoBias, so the value scales the attraction rather than flipping
    its sign -- measured directly: slot0=60 pulled cross-PF pairs to 11.7 A, slot0=5 to
    13.5 A, control 30.2 A. substructure bins everything past 16 A into one class, so 47 A
    and 17 A are indistinguishable to it. BASE never saw fibrils (upstream has no
    amyloid data, examples or constraints), so this is a domain gap, not a bug.

    Assignment-free by construction: re-cluster the chains into two groups from the CURRENT
    coordinates every step, so the model decides which chains form which protofilament and
    we only widen the gap it chose. Fixing the grouping a priori would fight the model,
    since without a constraint copy indices carry no protofilament meaning.

    Guidance tapers to zero over the last TAPER fraction of steps so the final geometry is
    the model's own and we do not leave a rigid artefact at the end of the trajectory.

    Off unless PF_REPULSION_TARGET is set (Angstroms). PF_REPULSION_MAXSTEP caps the
    per-step displacement; PF_REPULSION_TAPER is the fraction of the schedule over which
    the strength decays to 0.
    """
    target = os.environ.get("PF_REPULSION_TARGET", "")
    if not target:
        return x_l
    try:
        target = float(target)
    except ValueError:
        return x_l
    max_step = float(os.environ.get("PF_REPULSION_MAXSTEP", "2.0"))
    taper = float(os.environ.get("PF_REPULSION_TAPER", "0.3"))

    # Loudly mark any run that uses this. It is a DIAGNOSTIC PROBE, not a method: it edits
    # coordinates by hand at inference and its target is derived from GT (46.8 A is the median
    # GT protofilament separation), so numbers produced with it must never be reported as model
    # performance. Warn once per process so the provenance is in every log.
    global _PF_REP_WARNED
    if not _PF_REP_WARNED:
        _PF_REP_WARNED = True
        logger.warning(
            "PF_REPULSION ACTIVE (target=%s A, maxstep=%s, taper=%s) -- coordinates are being "
            "edited by hand during sampling and the target comes from GT. Diagnostic only; "
            "do NOT report these outputs as model performance.",
            target, max_step, taper,
        )

    is_lig = input_feature_dict.get("is_ligand", None)
    a2t = input_feature_dict.get("atom_to_token_idx", None)
    asym = input_feature_dict.get("asym_id", None)
    if is_lig is None or a2t is None or asym is None:
        return x_l

    # strength: full until (1 - taper) of the schedule, then linearly to 0
    frac = step_idx / max(1, n_steps - 1)
    cut = 1.0 - taper
    strength = 1.0 if frac <= cut else max(0.0, (1.0 - frac) / max(1e-6, taper))
    if strength <= 0:
        return x_l

    a2t_f = a2t.reshape(-1).long()
    asym_f = asym.reshape(-1).long()
    atom_asym = asym_f[a2t_f]                       # [N_atom]
    prot = ~is_lig.reshape(-1).bool()
    if prot.sum() < 2:
        return x_l

    flat = x_l.reshape(-1, x_l.shape[-2], 3)        # [B, N_atom, 3]
    ids = torch.unique(atom_asym[prot])
    if ids.numel() < 2:
        return x_l

    for b in range(flat.shape[0]):
        cen, keep = [], []
        for cid in ids:
            m = prot & (atom_asym == cid)
            if m.sum() >= 3:
                cen.append(flat[b][m].mean(0)); keep.append(cid)
        if len(cen) < 2:
            continue
        C = torch.stack(cen)                        # [n_chain, 3]
        # 2-means on chain centroids, seeded by the most distant pair (deterministic)
        D = torch.cdist(C, C)
        i, j = divmod(int(torch.argmax(D)), D.shape[0])
        m0, m1 = C[i].clone(), C[j].clone()
        for _ in range(8):
            lab = (torch.norm(C - m0, dim=1) > torch.norm(C - m1, dim=1)).long()
            if lab.sum() == 0 or lab.sum() == len(lab):
                break
            m0, m1 = C[lab == 0].mean(0), C[lab == 1].mean(0)
        if lab.sum() == 0 or lab.sum() == len(lab):
            continue
        u = m1 - m0
        cur = float(torch.norm(u))
        if cur < 1e-3 or cur >= target:
            continue
        u = u / cur
        shift = min(max_step, (target - cur)) * strength * 0.5
        for cid, lb in zip(keep, lab.tolist()):
            m = prot & (atom_asym == cid)
            flat[b][m] += (u * shift) if lb == 1 else (-u * shift)

    return flat.reshape(x_l.shape)


def sample_diffusion(
    denoise_net: Callable,
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    noise_schedule: torch.Tensor,
    N_sample: int = 1,
    gamma0: float = 0.8,
    gamma_min: float = 1.0,
    noise_scale_lambda: float = 1.003,
    step_scale_eta: float = 1.5,
    diffusion_chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    attn_chunk_size: Optional[int] = None,
    enable_efficient_fusion: bool = False,
    fix_protein_atoms: bool = False,
) -> torch.Tensor:
    """Implements Algorithm 18 in AF3.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        noise_schedule (torch.Tensor): noise-level schedule (which is also the time steps) since sigma=t.
            [N_iterations]
        N_sample (int): number of generated samples
        gamma0 (float): params in Alg.18.
        gamma_min (float): params in Alg.18.
        noise_scale_lambda (float): params in Alg.18.
        step_scale_eta (float): params in Alg.18.
        diffusion_chunk_size (Optional[int]): Chunk size for diffusion operation. Defaults to None.
        inplace_safe (bool): Whether to use inplace operations safely. Defaults to False.
        attn_chunk_size (Optional[int]): Chunk size for attention operation. Defaults to None.

    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    # Optional inference-time overrides (debug / ablations).
    # These apply whenever env vars are set, and default to the function args.
    # Typical use:
    #   export INFER_GAMMA0=0
    #   export INFER_STEP_SCALE_ETA=1.0
    gamma0_eff = float(os.environ.get("INFER_GAMMA0", str(gamma0)))
    step_scale_eta_eff = float(os.environ.get("INFER_STEP_SCALE_ETA", str(step_scale_eta)))
    noise_scale_lambda_eff = float(os.environ.get("INFER_NOISE_SCALE_LAMBDA", str(noise_scale_lambda)))
    if (
        os.environ.get("INFER_GAMMA0") is not None
        or os.environ.get("INFER_STEP_SCALE_ETA") is not None
        or os.environ.get("INFER_NOISE_SCALE_LAMBDA") is not None
    ):
        logger.info(
            "[sample_diffusion] overrides: gamma0=%.6f step_scale_eta=%.6f noise_scale_lambda=%.6f",
            gamma0_eff,
            step_scale_eta_eff,
            noise_scale_lambda_eff,
        )

    N_atom = input_feature_dict["atom_to_token_idx"].size(-1)
    batch_shape = s_inputs.shape[:-2]
    device = s_inputs.device
    dtype = s_inputs.dtype

    def _chunk_sample_diffusion(chunk_n_sample, inplace_safe):
        # Initialize x_l_gt for fixed protein atoms
        x_l_gt_original = None  # default: no GT track unless fix_protein_atoms=True
        if fix_protein_atoms:
            if "global_ref_pos" not in input_feature_dict:
                raise ValueError(
                    "global_ref_pos not found in input_feature_dict. "
                    "When fix_protein_atoms=True, global coordinates must be provided via CIF file."
                )
            
            # is_structured_protein: which atoms are held fixed during sampling (typically protein with valid GT).
            if "is_structured_protein" in input_feature_dict:
                is_structured_protein = input_feature_dict["is_structured_protein"].bool()  # [N_atom] or [B, N_atom]
                while is_structured_protein.dim() > 1 and is_structured_protein.shape[0] == 1:
                    is_structured_protein = is_structured_protein.squeeze(0)
            else:
                # Fallback: use all protein atoms
                is_protein = input_feature_dict["is_protein"].bool()  # [N_atom]
                is_structured_protein = is_protein
                logger.warning("is_structured_protein not found, using all protein atoms")

            # Never condition on unresolved atoms (no trustworthy global_ref_pos in the crystal frame).
            # NFT train_nft already sets is_structured_protein = protein & resolved; this intersects again when
            # is_resolved / coordinate_mask is present so inference or other callers that pass "all protein"
            # still do not clamp OXT etc. to x_l_augment.
            resolved_mask = None
            if "is_resolved" in input_feature_dict:
                resolved_mask = input_feature_dict["is_resolved"].bool()
            elif "coordinate_mask" in input_feature_dict:
                cm = input_feature_dict["coordinate_mask"]
                resolved_mask = (
                    cm.bool() if cm.dtype == torch.bool else (cm > 0)
                )
            if resolved_mask is not None:
                while resolved_mask.dim() > 1 and resolved_mask.shape[0] == 1:
                    resolved_mask = resolved_mask.squeeze(0)
                if resolved_mask.shape[0] == is_structured_protein.shape[0]:
                    is_structured_protein = is_structured_protein & resolved_mask.to(
                        device=is_structured_protein.device, dtype=torch.bool
                    )
                else:
                    logger.warning(
                        "sample_diffusion: is_resolved/coordinate_mask length %s != is_structured_protein %s; "
                        "skipping resolved intersection",
                        resolved_mask.shape[0],
                        is_structured_protein.shape[0],
                    )

            global_ref_pos = input_feature_dict["global_ref_pos"]  # [N_atom, 3] or [B, N_atom, 3]
            while global_ref_pos.dim() > 2 and global_ref_pos.shape[0] == 1:
                global_ref_pos = global_ref_pos.squeeze(0)
            
            # Prepare x_l_gt_original: expand global_ref_pos to match x_l shape.
            # Used for: (1) initial fill of x_l so gt is in x_l before first aug (ODesign-style);
            # (2) verification after reverse (check conditioned atoms match original frame).
            # In-loop conditioning uses x_l_augment (augmented x_l) instead of transforming x_l_gt.
            # Shape: [N_atom, 3] -> [..., chunk_n_sample, N_atom, 3]
            x_l_gt_original = global_ref_pos  # [N_atom, 3]
            for _ in range(len(batch_shape)):
                x_l_gt_original = x_l_gt_original.unsqueeze(0)
            x_l_gt_original = x_l_gt_original.unsqueeze(0).expand(*batch_shape, chunk_n_sample, -1, -1)  # [..., chunk_n_sample, N_atom, 3]
            x_l_gt = x_l_gt_original  # for initial torch.where only
            
            # Pre-compute condition_mask for structured protein atoms
            # Shape: [N_atom] -> [..., chunk_n_sample, N_atom]
            condition_mask = is_structured_protein  # [N_atom]
            for _ in range(len(batch_shape)):
                condition_mask = condition_mask.unsqueeze(0)
            condition_mask = condition_mask.unsqueeze(0).expand(*batch_shape, chunk_n_sample, -1)  # [..., chunk_n_sample, N_atom]
        else:
            x_l_gt = None
            is_structured_protein = None
            condition_mask = None

        # Build ligand mask (if available) in the same broadcasted shape as x_l / x_noisy
        # lig_mask: [..., chunk_n_sample, N_atom, 1], True for ligand atoms, False for protein
        lig_mask = input_feature_dict.get("is_ligand", None)
        if lig_mask is not None:
            lig_mask = lig_mask.to(device=device, dtype=torch.bool)
            while lig_mask.dim() < len(batch_shape) + 1:
                lig_mask = lig_mask.unsqueeze(0)
            lig_mask = lig_mask.unsqueeze(-1)  # [..., 1?, N_atom, 1]
            lig_mask = lig_mask.expand(*batch_shape, chunk_n_sample, N_atom, 1)

        # Initialize x_l:
        # - inference + fix_protein_atoms=True:
        #     ligand: random noise (same as standard EDM init)
        #     protein: start from GT (x_l_gt_original)
        # - otherwise (e.g. training confidence mini-rollout): pure noise for all atoms
        x_l = noise_schedule[0] * torch.randn(
            size=(*batch_shape, chunk_n_sample, N_atom, 3), device=device, dtype=dtype
        )  # NOTE: set seed in distributed training

        # Effective noise schedule used in the sampler loop.
        # Default: use the provided schedule (EDM/AF3).
        # If we override ligand init to be much smaller than noise_schedule[0], we must also
        # scale the schedule accordingly; otherwise step=0 will inject ~O(noise_schedule[0])
        # noise (delta_noise_level based on t_hat/c_tau_last), instantly throwing ligand far away.
        noise_schedule_eff = noise_schedule

        if fix_protein_atoms and lig_mask is not None and x_l_gt_original is not None:
            # Only ligand atoms keep noisy init; protein atoms start from GT.
            #
            # NOTE: ligand init can be overridden (to match training distribution better) via env:
            # - LIG_INIT_MODE:
            #     - "edm" (default): ligand init = noise_schedule[0] * N(0,1)  (original behavior)
            #     - "small_noise": ligand init = LIG_INIT_SIGMA * N(0,1)      (start near origin)
            #     - "zero": ligand init = 0                                   (start at origin)
            #     - "ref_pos": ligand init = base(ref_pos) + small noise      (base from global_ref_pos if available else 0)
            # - LIG_INIT_SIGMA (float, default=20.0): used for "small_noise" and "ref_pos"
            lig_init_mode = os.environ.get("LIG_INIT_MODE", "edm").lower()
            lig_init_sigma = float(os.environ.get("LIG_INIT_SIGMA", "20.0"))
            lig_init = x_l  # default: EDM noise already in x_l for ligand
            if lig_init_mode in ("small_noise", "small", "low_noise", "lownoise"):
                lig_init = lig_init_sigma * torch.randn_like(x_l, dtype=dtype)
            elif lig_init_mode in ("zero", "zeros", "origin"):
                lig_init = torch.zeros_like(x_l, dtype=dtype)
            elif lig_init_mode in ("ref_pos", "refpos", "ref"):
                # Prefer global_ref_pos (already in inference input frame); fall back to origin.
                base = input_feature_dict.get("global_ref_pos", None)
                if base is not None:
                    while base.dim() > 2 and base.shape[0] == 1:
                        base = base.squeeze(0)
                    # base: [N_atom, 3] -> [..., chunk_n_sample, N_atom, 3]
                    base_exp = base
                    for _ in range(len(batch_shape)):
                        base_exp = base_exp.unsqueeze(0)
                    base_exp = base_exp.unsqueeze(0).expand(*batch_shape, chunk_n_sample, -1, -1)
                else:
                    base_exp = torch.zeros_like(x_l, dtype=dtype)
                lig_init = base_exp + lig_init_sigma * torch.randn_like(x_l, dtype=dtype)
            # else: "edm" keeps lig_init=x_l

            # If we used a non-EDM init, rescale the sampler schedule so c_tau_last starts near lig_init_sigma.
            # This keeps the first "gamma" noise injection consistent with the actual init distribution.
            if lig_init_mode not in ("edm", "default", ""):
                try:
                    s0 = float(noise_schedule[0].item())
                    if s0 > 0 and lig_init_sigma > 0:
                        scale = lig_init_sigma / s0
                        noise_schedule_eff = noise_schedule * scale
                except Exception as e:
                    logger.warning("Failed to scale noise_schedule for LIG_INIT_MODE=%s: %s", lig_init_mode, str(e))

            x_l = torch.where(lig_mask, lig_init, x_l_gt_original)
        elif fix_protein_atoms and x_l_gt is not None and condition_mask is not None:
            # Fallback: original behavior when lig_mask is unavailable
            x_l = torch.where(
                condition_mask[..., None],  # [..., chunk_n_sample, N_atom, 1]
                x_l_gt,  # Use ground truth for structured protein atoms
                x_l,     # Use noise for non-structured atoms
            )

        for step_idx, (c_tau_last, c_tau) in enumerate(
            zip(noise_schedule_eff[:-1], noise_schedule_eff[1:])
        ):
            # Apply data augmentation to all atoms.
            # ODesign-style when fix_protein_atoms: full SE(3) (centre + rotation + translation)
            # on whole x_l so gt (already in x_l) is augmented together; else centre_only.
            x_l_augmented, trans, rot, x_center = centre_random_augmentation_rtc(
                x_input_coords=x_l,
                N_sample=1,
                centre_only=not fix_protein_atoms,
                dtype=dtype
            )
            x_l = x_l_augmented.squeeze(dim=-3).to(dtype)
            # When fix_protein_atoms: use augmented x_l as conditioning reference this step
            # (gt was inside x_l and got rotated/translated together; no separate x_l_gt transform).
            if fix_protein_atoms:
                x_l_augment = x_l.clone()

            # Denoise with a predictor-corrector sampler
            # 1. Add noise to move x_{c_tau_last} to x_{t_hat}
            gamma = float(gamma0_eff) if c_tau > gamma_min else 0
            t_hat = c_tau_last * (gamma + 1)

            delta_noise_level = torch.sqrt(t_hat**2 - c_tau_last**2)
            # Add noise to move from c_tau_last to t_hat.
            # Train-consistent semantics:
            # - ligand atoms: add noise
            # - protein atoms: stay at current GT track (no extra noise)
            noise = noise_scale_lambda_eff * delta_noise_level * torch.randn(
                size=x_l.shape, device=device, dtype=dtype
            )
            if lig_mask is not None:
                x_noisy = torch.where(lig_mask, x_l + noise, x_l)
            else:
                x_noisy = x_l + noise

            # 2. Denoise from x_{t_hat} to x_{c_tau}
            # Euler step only
            t_hat = (
                t_hat.reshape((1,) * (len(batch_shape) + 1))
                .expand(*batch_shape, chunk_n_sample)
                .to(dtype)
            )

            x_denoised = denoise_net(
                x_noisy=x_noisy,
                t_hat_noise_level=t_hat,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                chunk_size=attn_chunk_size,
                inplace_safe=inplace_safe,
                enable_efficient_fusion=enable_efficient_fusion,
            )

            # Compute score function approximation (gradient)
            delta = (x_noisy - x_denoised) / t_hat[
                ..., None, None
            ]  # Line 9 of AF3 uses 'x_l_hat' instead, which we believe  is a typo.
            dt = c_tau - t_hat
            
            # Take Euler step: x_new = x_noisy + eta * dt * delta
            x_l = x_noisy + step_scale_eta_eff * dt[..., None, None] * delta
            
            # External protofilament-separation guidance (no-op unless PF_REPULSION_TARGET set)
            x_l = _pf_repulsion(x_l, input_feature_dict, step_idx,
                                len(noise_schedule_eff) - 1)
            
            # Enforce conditioning: replace structured protein atoms with augmented gt (ODesign-style)
            if fix_protein_atoms and condition_mask is not None:
                x_l = torch.where(
                    condition_mask[..., None],  # [..., chunk_n_sample, N_atom, 1]
                    x_l_augment,  # Use augmented gt for structured protein atoms
                    x_l           # Use updated prediction for others
                )
            
            # Reverse SE(3) augmentation to return to original frame (ODesign: only reverse x_l)
            x_l_expanded = x_l.unsqueeze(dim=-3)  # [..., 1, chunk_n_sample, N_atom, 3]
            
            # trans: [..., 1, 3], rot: [..., 1, 3, 3], x_center: [..., 1, 3]
            # These match the expected shapes for reverse function
            x_l_reversed = reverse_centre_random_augmentation_rtc(
                x_l_expanded, trans, rot, x_center
            )
            x_l = x_l_reversed.squeeze(dim=-3).to(dtype)  # Remove N_sample dimension
            
            # Verification: conditioned atoms (after reverse) must match original gt
            if fix_protein_atoms and x_l_gt_original is not None and condition_mask is not None:
                check_condition_atom_coords(x_l, x_l_gt_original, condition_mask)
        
        return x_l

    if diffusion_chunk_size is None:
        x_l = _chunk_sample_diffusion(N_sample, inplace_safe=inplace_safe)
    else:
        x_l = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            chunk_n_sample = (
                diffusion_chunk_size
                if i < no_chunks - 1
                else N_sample - i * diffusion_chunk_size
            )
            chunk_x_l = _chunk_sample_diffusion(
                chunk_n_sample, inplace_safe=inplace_safe
            )
            x_l.append(chunk_x_l)
        x_l = torch.cat(x_l, -3)  # [..., N_sample, N_atom, 3]
    return x_l


def sample_diffusion_training(
    noise_sampler: TrainingNoiseSampler,
    denoise_net: Callable,
    label_dict: dict[str, Any],
    input_feature_dict: dict[str, Any],
    s_inputs: torch.Tensor,
    s_trunk: torch.Tensor,
    z_trunk: torch.Tensor,
    pair_z: torch.Tensor,
    p_lm: torch.Tensor,
    c_l: torch.Tensor,
    N_sample: int = 1,
    diffusion_chunk_size: Optional[int] = None,
    use_conditioning: bool = True,
    enable_efficient_fusion: bool = False,
) -> tuple[torch.Tensor, ...]:
    """Implements diffusion training as described in AF3 Appendix at page 23.
    It performances denoising steps from time 0 to time T.
    The time steps (=noise levels) are given by noise_schedule.

    Args:
        denoise_net (Callable): the network that performs the denoising step.
        label_dict (dict, optional) : a dictionary containing the followings.
            "coordinate": the ground-truth coordinates
                [..., N_atom, 3]
            "coordinate_mask": whether true coordinates exist.
                [..., N_atom]
        input_feature_dict (dict[str, Any]): input meta feature dict
        s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
            [..., N_tokens, c_s_inputs]
        s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
            [..., N_tokens, c_s]
        z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
            [..., N_tokens, N_tokens, c_z]
        N_sample (int): number of training samples
    Returns:
        torch.Tensor: the denoised coordinates of x in inference stage
            [..., N_sample, N_atom, 3]
    """
    batch_size_shape = label_dict["coordinate"].shape[:-2]
    device = label_dict["coordinate"].device
    dtype = label_dict["coordinate"].dtype
    # Create N_sample versions with random R/t; return (trans, rot, x_center) so we can reverse for loss
    x_gt_augment, trans, rot, x_center = centre_random_augmentation_rtc(
        x_input_coords=label_dict["coordinate"],
        N_sample=N_sample,
        mask=label_dict["coordinate_mask"],
        dtype=dtype,
    )  # x_gt_augment [..., N_sample, N_atom, 3]; trans [..., N_sample, 3]; rot [..., N_sample, 3, 3]; x_center [..., 1, 3]

    # Add independent noise to each structure
    # sigma: independent noise-level [..., N_sample]
    sigma = noise_sampler(size=(*batch_size_shape, N_sample), device=device).to(dtype)
    # noise: [..., N_sample, N_atom, 3]
    noise = torch.randn_like(x_gt_augment, dtype=dtype) * sigma[..., None, None]

    # Training input: add noise to all atoms (protein + ligand), consistent with原始 AF3 语义。
    # 之前我们只对 ligand 加噪、protein 用 GT；现在按你的要求先关闭这个特殊逻辑。
    x_noisy = x_gt_augment + noise
    # lig_mask = input_feature_dict.get("is_ligand", None)
    # if lig_mask is not None:
    #     # lig_mask: [N_atom] (0/1 or bool) -> broadcast to [..., N_sample, N_atom, 1]
    #     lig_mask = lig_mask.to(device=device, dtype=torch.bool)
    #     while lig_mask.dim() < x_gt_augment.dim() - 1:
    #         lig_mask = lig_mask.unsqueeze(0)
    #     lig_mask = lig_mask.unsqueeze(-1)
    #     # Only ligand atoms get noise; protein atoms stay at GT (x_gt_augment)
    #     x_noisy = torch.where(lig_mask, x_gt_augment + noise, x_gt_augment)

    # Get denoising outputs [..., N_sample, N_atom, 3]
    if diffusion_chunk_size is None:
        x_denoised = denoise_net(
            x_noisy=x_noisy,
            t_hat_noise_level=sigma,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            pair_z=pair_z,
            p_lm=p_lm,
            c_l=c_l,
            use_conditioning=use_conditioning,
            enable_efficient_fusion=enable_efficient_fusion,
        )
    else:
        x_denoised = []
        no_chunks = N_sample // diffusion_chunk_size + (
            N_sample % diffusion_chunk_size != 0
        )
        for i in range(no_chunks):
            x_noisy_i = x_noisy[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size, :, :
            ]
            t_hat_noise_level_i = sigma[
                ..., i * diffusion_chunk_size : (i + 1) * diffusion_chunk_size
            ]
            x_denoised_i = denoise_net(
                x_noisy=x_noisy_i,
                t_hat_noise_level=t_hat_noise_level_i,
                input_feature_dict=input_feature_dict,
                s_inputs=s_inputs,
                s_trunk=s_trunk,
                z_trunk=z_trunk,
                pair_z=pair_z,
                p_lm=p_lm,
                c_l=c_l,
                use_conditioning=use_conditioning,
                enable_efficient_fusion=enable_efficient_fusion,
            )
            x_denoised.append(x_denoised_i)
        x_denoised = torch.cat(x_denoised, dim=-3)

    # Reverse augmentation so coordinates are in original frame (same as label_dict["coordinate"]).
    x_denoised = reverse_centre_random_augmentation_rtc(
        x_denoised, trans, rot, x_center
    )

    return x_gt_augment, x_denoised, sigma
