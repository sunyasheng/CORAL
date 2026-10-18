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

from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation

from protenix.utils.scatter_utils import scatter


def centre_random_augmentation(
    x_input_coords: torch.Tensor,
    N_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: torch.Tensor = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Implements Algorithm 19 in AF3

    Args:
        x_input_coords (torch.Tensor): input coords
            [..., N_atom, 3]
        N_sample (int, optional): the total number of augmentation. Defaults to 1.
        s_trans (float, optional): scale factor of trans. Defaults to 1.0.
        centre_only (bool, optional): if set true, will only perform centering without applying random translation and rotation.
        mask (torch.Tensor, optional): masking for the coords
            [..., N_atom]
        eps (float, optional): small number used for masked mean
    Returns:
        torch.Tensor:  the Augmentation version of input coords
            [..., N_sample, N_atom, 3]
    """

    N_atom = x_input_coords.size(-2)
    device = x_input_coords.device

    # Move to origin [..., N_atom, 3]
    if mask is None:
        x_input_coords = x_input_coords - torch.mean(
            input=x_input_coords, dim=-2, keepdim=True
        )
    else:
        center = (x_input_coords * mask.unsqueeze(dim=-1)).sum(dim=-2) / (
            mask.sum(dim=-1) + eps
        )
        x_input_coords = x_input_coords - center.unsqueeze(dim=-2)

    # Expand to [..., N_sample, N_atom, 3]
    x_input_coords = expand_at_dim(x_input_coords, dim=-3, n=N_sample)

    if centre_only:
        return x_input_coords

    # N_augment = batch_size * N_sample
    N_augment = torch.numel(x_input_coords[..., 0, 0])

    # Generate N_augment (rot, trans) pairs
    batch_size_shape = x_input_coords.shape[:-3]
    rot_matrix_random = (
        uniform_random_rotation(N_sample=N_augment)
        .to(device)
        .reshape(*batch_size_shape, N_sample, 3, 3)
    ).detach()  # [..., N_sample, 3, 3]
    trans_random = s_trans * torch.randn(
        size=(*batch_size_shape, N_sample, 3), device=device
    )  # [..., N_sample, 3]
    x_augment_coords = (
        rot_vec_mul(
            r=expand_at_dim(rot_matrix_random, dim=-3, n=N_atom), t=x_input_coords
        )
        + trans_random[..., None, :]
    )  # [..., N_sample, N_atom, 3]

    if mask is not None:
        x_augment_coords = x_augment_coords * mask[..., None, :, None]
    return x_augment_coords


def centre_random_augmentation_with_params(
    x_input_coords: torch.Tensor,
    N_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: torch.Tensor = None,
    eps: float = 1e-12,
    return_params: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Implements Algorithm 19 in AF3 with transform parameters returned
    
    This function is similar to centre_random_augmentation, but also returns
    the transformation parameters (center, rotation matrix, translation) so
    that the same transformation can be applied to other coordinates.

    Args:
        x_input_coords (torch.Tensor): input coords
            [..., N_atom, 3]
        N_sample (int, optional): the total number of augmentation. Defaults to 1.
        s_trans (float, optional): scale factor of trans. Defaults to 1.0.
        centre_only (bool, optional): if set true, will only perform centering without applying random translation and rotation.
        mask (torch.Tensor, optional): masking for the coords
            [..., N_atom]
        eps (float, optional): small number used for masked mean
        return_params (bool, optional): whether to return transformation parameters. Defaults to True.
    
    Returns:
        tuple[torch.Tensor, dict]: 
            - Augmented coordinates [..., N_sample, N_atom, 3]
            - Dictionary containing transformation parameters:
                - 'center': [..., 3] - the center point that was subtracted
                - 'rot_matrix': [..., N_sample, 3, 3] - rotation matrices (None if centre_only=True)
                - 'translation': [..., N_sample, 3] - translation vectors (None if centre_only=True)
    """
    N_atom = x_input_coords.size(-2)
    device = x_input_coords.device
    batch_size_shape = x_input_coords.shape[:-2]

    # Compute center and move to origin [..., N_atom, 3]
    if mask is None:
        center = torch.mean(input=x_input_coords, dim=-2, keepdim=False)  # [..., 3]
        x_centered = x_input_coords - center.unsqueeze(dim=-2)  # [..., N_atom, 3]
    else:
        center = (x_input_coords * mask.unsqueeze(dim=-1)).sum(dim=-2) / (
            mask.sum(dim=-1, keepdim=True) + eps
        )  # [..., 3]
        x_centered = x_input_coords - center.unsqueeze(dim=-2)  # [..., N_atom, 3]

    # Expand to [..., N_sample, N_atom, 3]
    x_centered_expanded = expand_at_dim(x_centered, dim=-3, n=N_sample)

    # Initialize transform parameters
    transform_params = {
        'center': center,  # [..., 3]
        'rot_matrix': None,
        'translation': None,
    }

    if centre_only:
        return x_centered_expanded, transform_params

    # N_augment = batch_size * N_sample
    N_augment = torch.numel(x_centered_expanded[..., 0, 0])

    # Generate N_augment (rot, trans) pairs
    rot_matrix_random = (
        uniform_random_rotation(N_sample=N_augment)
        .to(device)
        .reshape(*batch_size_shape, N_sample, 3, 3)
    ).detach()  # [..., N_sample, 3, 3]
    trans_random = s_trans * torch.randn(
        size=(*batch_size_shape, N_sample, 3), device=device
    )  # [..., N_sample, 3]
    
    # Apply rotation and translation
    x_augment_coords = (
        rot_vec_mul(
            r=expand_at_dim(rot_matrix_random, dim=-3, n=N_atom), t=x_centered_expanded
        )
        + trans_random[..., None, :]
    )  # [..., N_sample, N_atom, 3]

    if mask is not None:
        x_augment_coords = x_augment_coords * mask[..., None, :, None]
    
    # Store transform parameters
    transform_params['rot_matrix'] = rot_matrix_random  # [..., N_sample, 3, 3]
    transform_params['translation'] = trans_random  # [..., N_sample, 3]

    return x_augment_coords, transform_params


def apply_augmentation_params(
    x_input_coords: torch.Tensor,
    transform_params: dict,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Apply saved transformation parameters to new coordinates
    
    This function applies the same transformation (centering, rotation, translation)
    that was computed by centre_random_augmentation_with_params to a new set of coordinates.

    Args:
        x_input_coords (torch.Tensor): input coords to transform
            [..., N_atom, 3]
        transform_params (dict): transformation parameters from centre_random_augmentation_with_params
            Must contain:
                - 'center': [..., 3] - the center point to subtract
                - 'rot_matrix': [..., N_sample, 3, 3] - rotation matrices (can be None)
                - 'translation': [..., N_sample, 3] - translation vectors (can be None)
        mask (torch.Tensor, optional): masking for the coords
            [..., N_atom]
    
    Returns:
        torch.Tensor: transformed coordinates
            [..., N_sample, N_atom, 3]
    """
    device = x_input_coords.device
    N_atom = x_input_coords.size(-2)
    
    # Extract parameters
    center = transform_params['center']  # [..., 3]
    rot_matrix = transform_params.get('rot_matrix', None)  # [..., N_sample, 3, 3] or None
    translation = transform_params.get('translation', None)  # [..., N_sample, 3] or None
    
    # Step 1: Center the coordinates
    x_centered = x_input_coords - center.unsqueeze(dim=-2)  # [..., N_atom, 3]
    
    # Determine N_sample from rot_matrix or translation
    if rot_matrix is not None:
        N_sample = rot_matrix.size(-3)
    elif translation is not None:
        N_sample = translation.size(-2)
    else:
        N_sample = 1
    
    # Expand to [..., N_sample, N_atom, 3]
    x_centered_expanded = expand_at_dim(x_centered, dim=-3, n=N_sample)
    
    # Step 2: Apply rotation if available
    if rot_matrix is not None:
        x_rotated = rot_vec_mul(
            r=expand_at_dim(rot_matrix, dim=-3, n=N_atom), 
            t=x_centered_expanded
        )  # [..., N_sample, N_atom, 3]
    else:
        x_rotated = x_centered_expanded
    
    # Step 3: Apply translation if available
    if translation is not None:
        x_transformed = x_rotated + translation[..., None, :]  # [..., N_sample, N_atom, 3]
    else:
        x_transformed = x_rotated
    
    # Apply mask if provided
    if mask is not None:
        x_transformed = x_transformed * mask[..., None, :, None]
    
    return x_transformed


# Comment: Rotation.random is not supported by torch.compile()
def uniform_random_rotation(N_sample: int = 1) -> torch.Tensor:
    """Generate random rotation matrices with scipy.spatial.transform.Rotation

    Args:
        N_sample (int, optional): the total number of augmentation. Defaults to 1.

    Returns:
        torch.Tensor: N_sample rot matrics
            [N_sample, 3, 3]
    """
    rotation = Rotation.random(num=N_sample)
    rot_matrix = torch.from_numpy(rotation.as_matrix()).float()  # [N_sample, 3, 3]
    return rot_matrix


def centre_random_augmentation_rtc(
    x_input_coords: torch.Tensor,
    N_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: torch.Tensor = None,
    eps: float = 1e-12,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Implements Algorithm 19 in AF3 with return transform components (rtc)
    
    This function is similar to centre_random_augmentation, but returns
    the transformation components (translation, rotation, center) along with
    the augmented coordinates. This allows for later reversal of the transformation.

    Args:
        x_input_coords (torch.Tensor): input coords
            [..., N_atom, 3]
        N_sample (int, optional): the total number of augmentation. Defaults to 1.
        s_trans (float, optional): scale factor of trans. Defaults to 1.0.
        centre_only (bool, optional): if set true, will only perform centering without applying random translation and rotation.
        mask (torch.Tensor, optional): masking for the coords
            [..., N_atom]
        eps (float, optional): small number used for masked mean
        dtype (torch.dtype, optional): data type for output tensors. Defaults to torch.float32.
    
    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: 
            - x_augment_coords: the Augmentation version of input coords [..., N_sample, N_atom, 3]
            - trans_random: translation vectors [..., N_sample, 3]
            - rot_matrix_random: rotation matrices [..., N_sample, 3, 3]
            - x_center: center point that was subtracted [..., 1, 3]
    """

    N_atom = x_input_coords.size(-2)
    device = x_input_coords.device

    # Move to origin [..., N_atom, 3]
    # Compute center of coordinates (with optional masking)
    if mask is None:
        x_center = torch.mean(
            input=x_input_coords, dim=-2, keepdim=True
        )
    else:
        x_center = ((x_input_coords * mask.unsqueeze(dim=-1)).sum(dim=-2) / (
            mask.sum(dim=-1) + eps
        )).unsqueeze(dim=-2)

    x_input_coords = x_input_coords - x_center 

    # Expand to [..., N_sample, N_atom, 3]
    x_input_coords = expand_at_dim(x_input_coords, dim=-3, n=N_sample)

    if centre_only:
        return (
            x_input_coords.to(dtype),
            torch.zeros((*x_input_coords.shape[:-3], N_sample, 3), device=device, dtype=dtype),
            torch.eye(3, device=device, dtype=dtype).expand((*x_input_coords.shape[:-3], N_sample, 3, 3)),
            x_center.to(dtype)
        )

    # N_augment = batch_size * N_sample
    N_augment = torch.numel(x_input_coords[..., 0, 0])

    # Generate N_augment (rot, trans) pairs
    batch_size_shape = x_input_coords.shape[:-3]
    rot_matrix_random = (
        uniform_random_rotation(N_sample=N_augment)
        .to(device)
        .reshape(*batch_size_shape, N_sample, 3, 3)
    ).detach()  # [..., N_sample, 3, 3]
    trans_random = s_trans * torch.randn(
        size=(*batch_size_shape, N_sample, 3), device=device
    )  # [..., N_sample, 3]
    x_augment_coords = (
        rot_vec_mul(
            r=expand_at_dim(rot_matrix_random, dim=-3, n=N_atom), t=x_input_coords
        )
        + trans_random[..., None, :]
    )  # [..., N_sample, N_atom, 3]

    if mask is not None:
        x_augment_coords = x_augment_coords * mask[..., None, :, None]

    # Convert to specified dtype for consistency
    return (
        x_augment_coords.to(dtype), 
        trans_random.to(dtype), 
        rot_matrix_random.to(dtype), 
        x_center.to(dtype)
    )


def reverse_centre_random_augmentation_rtc(
    x_augment_coords: torch.Tensor, 
    trans: torch.Tensor, 
    rot: torch.Tensor, 
    x_center: torch.Tensor
) -> torch.Tensor:
    """
    Reverse the centre random augmentation to recover original coordinates.
    
    This function inverts the transformation applied by centre_random_augmentation_rtc
    by applying inverse rotation, inverse translation, and restoring the original center.
    
    Args:
        x_augment_coords (torch.Tensor): Augmented coordinates to reverse.
            Shape: [..., N_sample, N_atom, 3]
        trans (torch.Tensor): Translation vector used in augmentation.
            Shape: [..., N_sample, 3]
        rot (torch.Tensor): Rotation matrix used in augmentation.
            Shape: [..., N_sample, 3, 3]
        x_center (torch.Tensor): Original center of the coordinates.
            Shape: [..., 1, 3] or [..., N_atom, 3]
            
    Returns:
        torch.Tensor: Reversed coordinates in original space.
            Shape: [..., N_sample, N_atom, 3]
    """
    N_atom = x_augment_coords.size(-2)

    # Apply inverse translation (subtract the translation vector)
    x_reversed = x_augment_coords - trans[..., None, :]
    
    # Apply inverse rotation (transpose of rotation matrix)
    x_reversed = rot_vec_mul(
        r=expand_at_dim(rot.transpose(-1, -2), dim=-3, n=N_atom), 
        t=x_reversed
    )
    
    # Restore to original center
    x_reversed = x_reversed + x_center[..., None, :]
    
    return x_reversed.view_as(x_augment_coords)


# this is from openfold.utils.rigid_utils import rot_vec_mul
# precision is added
def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply rot matrix to vector
    Applies a rotation to a vector. Written out by hand to avoid transfer
    to avoid AMP downcasting.

    Args:
        r (torch.Tensor): the rotation matrices
            [..., 3, 3]
        t (torch.Tensor): the coordinate tensors
            [..., 3]

    Returns:
        torch.Tensor: the rotated coordinates
    """
    if t.dtype != torch.float32:
        t = t.to(dtype=torch.float32)
    if r.dtype != torch.float32:
        r = r.to(dtype=torch.float32)

    x, y, z = torch.unbind(input=t, dim=-1)
    return torch.stack(
        tensors=[
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


# from openfold.utils.tensor_utils.permute_final_dims
# from openfold.utils.tensor_utils.flatten_final_dims
def permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    """Permute final dims of tensor

    Args:
        tensor (torch.Tensor): the input tensor
            [...]
        inds (List[int]): the dim to permute

    Returns:
        torch.Tensor: the permuted tensor
    """
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, num_dims: int) -> torch.Tensor:
    """Flatten final dims of tensor

    Args:
        t (torch.Tensor): the input tensor
            [...]
        num_dims (int): the number of final dims to flatten

    Returns:
        torch.Tensor: the flattened tensor
    """
    return t.reshape(shape=t.shape[:-num_dims] + (-1,))


def one_hot(
    x: torch.Tensor, lower_bins: torch.Tensor, upper_bins: torch.Tensor
) -> torch.Tensor:
    """Get one hot embedding of x from lower_bins and upper_bins
    Args:
        x (torch.Tensor): the input x
            [...]
        lower_bins (torch.Tensor): the lower bounds of bins
            [bins]
        upper_bins (torch.Tensor): the upper bounds of bins
            [bins]
    Returns:
        torch.Tensor: the one hot embedding of x from v_bins
            [..., bins]
    """
    dgram = (x[..., None] > lower_bins) * (x[..., None] < upper_bins).float()
    return dgram


# this is mostly from openfold.utils.torch_utils import batched_gather
def batched_gather(
    data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0
) -> torch.Tensor:
    """Gather data according to indices specify by inds

    Args:
        data (torch.Tensor): the input data
            [..., K, ...]
        inds (torch.Tensor): the indices for gathering data
            [..., N]
        dim (int, optional): along which dimension to gather data by inds (the dim of "K" "N"). Defaults to 0.
        no_batch_dims (int, optional): length of dimensions before the "dim" dimension. Defaults to 0.

    Returns:
        torch.Tensor: gathered data
            [..., N, ...]
    """

    # for the naive case
    if len(inds.shape) == 1 and no_batch_dims == 0 and dim == 0:
        return data[inds]

    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def broadcast_token_to_atom(
    x_token: torch.Tensor, atom_to_token_idx: torch.Tensor
) -> torch.Tensor:
    """Broadcast token-level embeddings to atom-level embeddings

    Args:
        x_token (torch.Tensor): token embedding
            [..., N_token, d]
        atom_to_token_idx (torch.Tensor): map atom idx to token idx
            [..., N_atom] or [N_atom]

    Returns:
        torch.Tensor: atom embedding
            [..., N_atom, d]
    """

    if len(atom_to_token_idx.shape) == 1:
        # shape = [N_atom], easy index
        return x_token[..., atom_to_token_idx, :]
    else:
        assert atom_to_token_idx.shape[:-1] == x_token.shape[:-2]

    return batched_gather(
        data=x_token,
        inds=atom_to_token_idx,
        dim=-2,
        no_batch_dims=len(x_token.shape[:-2]),
    )


def aggregate_atom_to_token(
    x_atom: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    n_token: Optional[int] = None,
    reduce: str = "mean",
) -> torch.Tensor:
    """Aggregate atom embedding to obtain token embedding

    Args:
        x_atom (torch.Tensor): atom-level embedding
            [..., N_atom, d]
        atom_to_token_idx (torch.Tensor): map atom to token idx
            [..., N_atom] or [N_atom]
        n_token (int, optional): number of tokens in total. Defaults to None.
        reduce (str, optional): aggregation method. Defaults to "mean".

    Returns:
        torch.Tensor: token-level embedding
            [..., N_token, d]
    """

    # Broadcasting in the given dim.
    out = scatter(
        src=x_atom, index=atom_to_token_idx, dim=-2, dim_size=n_token, reduce=reduce
    )

    return out


def sample_indices(
    n: int,
    device: torch.device = torch.device("cpu"),
    lower_bound=1,
    strategy: str = "random",
) -> torch.Tensor:
    """Sample msa indices k from uniform[1,n]

    Args:
        n (int): the msa num
        strategy (str): the strategy to sample msa index, random or topk

    Returns:
        torch.Tensor: the sampled indices k
    """
    assert strategy in ["random", "topk"]
    sample_size = torch.randint(low=min(lower_bound, n), high=n + 1, size=(1,)).item()
    if strategy == "random":
        indices = torch.randperm(n=n, device=device)[:sample_size]
    if strategy == "topk":
        indices = torch.arange(sample_size, device=device)
    return indices


def sample_msa_feature_dict_random_without_replacement(
    feat_dict: dict[str, torch.Tensor],
    dim_dict: dict[str, int],
    cutoff: int = 512,
    lower_bound: int = 1,
    strategy: str = "random",
) -> dict[str, torch.Tensor]:
    """Sample a dict of MSA features randomly without replacement.

    Args:
        feat_dict (dict[str, torch.Tensor]): A dict containing the MSA features.
        dim_dict (dict[str, int]): A dict containing the dimensions of the MSA features.
        cutoff (int): The maximum number of features to sample.
        lower_bound (int): The minimum number of features to sample.
        strategy (str): The sampling strategy to use. Can be either "random" or "sequential".

    Returns:
        dict[str, torch.Tensor]: A dict containing the sampled MSA features.
    """
    msa_len = feat_dict["msa"].size(dim=dim_dict["msa"])
    indices = sample_indices(
        n=msa_len,
        device=feat_dict["msa"].device,
        lower_bound=lower_bound,
        strategy=strategy,
    )
    if cutoff > 0:
        indices = indices[:cutoff]

    msa_feat_dict = {
        feat_name: torch.index_select(
            input=feat_dict[feat_name], dim=dim, index=indices
        )
        for feat_name, dim in dim_dict.items()
    }
    return msa_feat_dict


def expand_at_dim(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """expand a tensor at specific dim by n times

    Args:
        x (torch.Tensor): input
        dim (int): dimension to expand
        n (int): expand size

    Returns:
        torch.Tensor: expanded tensor of shape [..., n, ...]
    """
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1 :]
    return x.expand(*before_shape, n, *after_shape)


def pad_at_dim(
    x: torch.Tensor,
    dim: int,
    pad_length: Union[tuple[int], list[int]],
    value: float = 0,
) -> torch.Tensor:
    """pad to input x at dimension dim with length pad_length[0] to the left and and pad_length[1] to the right.

    Args:
        x (torch.Tensor): input
        dim (int): padding dimension
        pad_length (Union[Tuple[int], List[int]]): length to pad to the beginning and end.

    Returns:
        torch.Tensor: padded tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    pad = (pad_length[0], pad_length[1])
    if pad == (0, 0):
        return x
    k = n_dim - (dim + 1)
    if k > 0:
        pad_skip = (0, 0) * k
        pad = (*pad_skip, *pad)
    return nn.functional.pad(x, pad=pad, value=value)


def reshape_at_dim(
    x: torch.Tensor, dim: int, target_shape: Union[tuple[int], list[int]]
) -> torch.Tensor:
    """reshape dimension dim of x to target_shape

    Args:
        x (torch.Tensor): input
        dim (int): dimension to reshape
        target_shape (Union[Tuple[int], List[int]]): target_shape of dim

    Returns:
        torch.Tensor: reshaped tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    target_shape = tuple(target_shape)
    target_shape = (*x.shape[:dim], *target_shape)
    if dim + 1 < n_dim:
        target_shape = (*target_shape, *x.shape[dim + 1 :])
    return x.reshape(target_shape)


def move_final_dim_to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Move the final dimension of a tensor to a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Target dimension to move the final dimension to.

    Returns:
        torch.Tensor: Tensor with the final dimension moved to the specified dimension.
    """
    # permute_final_dims
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim
    if dim >= n_dim - 1:
        return x

    new_order = (n_dim - 1,)
    if dim > 0:
        new_order = tuple(range(dim)) + new_order
    if dim < n_dim - 1:
        new_order = new_order + tuple(range(dim, n_dim - 1))

    return x.permute(new_order)


def simple_merge_dict_list(dict_list: list[dict]) -> dict:
    """
    Merge a list of dictionaries into a single dictionary.

    Args:
        dict_list (list[dict]): List of dictionaries to merge.

    Returns:
        dict: Merged dictionary where values are concatenated arrays.
    """
    merged_dict = {}

    def add(key, value):
        merged_dict.setdefault(key, [])
        if isinstance(value, (float, int)):
            value = np.array([value])
        elif isinstance(value, torch.Tensor):
            if value.dim() == 0:
                value = np.array([value.item()])
            else:
                value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type for metric data: {type(value)}")
        merged_dict[key].append(value)

    for x in dict_list:
        for k, v in x.items():
            add(k, v)
    for k, v in merged_dict.items():
        merged_dict[k] = np.concatenate(v)
    return merged_dict
