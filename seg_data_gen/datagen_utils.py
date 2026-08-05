# Copyright (c) Neel Dey
# Project Home: https://github.com/neel-dey/anatomix/
# Modified on 2026-08-05: Based on the above open-source project for secondary development
# Modified by: Derong Yu
import numpy as np
import torch
import torch.nn.functional as F

from monai.transforms import (
    ScaleIntensityd,
    Compose,
    RandBiasFieldd,
    RandAdjustContrastd,
    RandGaussianSmoothd,
    RandGaussianSharpend,
    RandGibbsNoised,
    RandKSpaceSpikeNoised,
    RandSimulateLowResolutiond,
    ThresholdIntensityd
)

def create_sphere_kernel(radius: int, device='cuda'):
    """
    Generate a 3D sphere kernel.
    """
    diameter = 2 * radius + 1
    array_shape = (diameter, diameter, diameter)
    center = (radius, radius, radius)

    # Create an empty 3D array
    kernel = torch.zeros(array_shape, dtype=torch.float32, device=device)
    
    # Generate coordinates for all voxels in the array
    x_coords, y_coords, z_coords = torch.meshgrid(
        torch.arange(array_shape[0], device=device),
        torch.arange(array_shape[1], device=device),
        torch.arange(array_shape[2], device=device),
    )
    
    # Calculate distances for all voxels to the center
    distances = torch.sqrt(
        (x_coords - center[0]) ** 2 +
        (y_coords - center[1]) ** 2 +
        (z_coords - center[2]) ** 2
    )
    
    # Set values within the sphere's radius to 1
    kernel[distances <= radius] = 1
    
    return kernel.unsqueeze(0).unsqueeze(0)


def spherical_dilation(x: torch.Tensor, radius: int, iterations=1):
    """3D spherical dilation."""
    kernel = create_sphere_kernel(radius, x.device)
    kernel = kernel / kernel.sum()
    pad = radius  

    for _ in range(iterations):
        x_conv = F.conv3d(
            x.to(torch.float32), kernel, padding=pad, stride=1,
        )
        x_dilated = x_conv > 0
        x = x_dilated.int()

    return x


def spherical_erosion(x: torch.Tensor, radius: int, iterations=1):
    """3D spherical dilation."""
    kernel = create_sphere_kernel(radius, x.device)
    kernel = kernel / kernel.sum()
    pad = radius  

    for _ in range(iterations):
        x_conv = F.conv3d(
            x.to(torch.float32), kernel, padding=pad, stride=1,
        )
        x_eroded = x_conv > 0.9
        x = x_eroded.int()

    return x


def spherical_closing(x: torch.Tensor, radius: int):
    """3D spherical closing."""
    x = spherical_dilation(x, radius)
    x = spherical_erosion(x, radius)
    return x


def get_foreground_mask(foreground_label):
    """get the foreground mask of the volume."""
    foreground_label_pad = torch.zeros(
        (foreground_label.shape[0] + 20, 
         foreground_label.shape[1] + 20, 
         foreground_label.shape[2] + 20), dtype=foreground_label.dtype, device=foreground_label.device)
    foreground_label_pad[10:-10, 10:-10, 10:-10] = foreground_label
    
    close_foreground = foreground_label_pad.unsqueeze(0).unsqueeze(0)
    for radius in [13, 11, 7, 4, 2, 1]:
        close_foreground = spherical_closing(close_foreground, radius=radius)
    close_foreground = close_foreground.squeeze(0).squeeze(0)

    close_foreground = close_foreground[10:-10, 10:-10, 10:-10]
    return close_foreground



def get_transforms():
    """
    Generates a MONAI composed transformation set for augmenting the GMM
    sampled intensities. 

    See the comments below to walk through the transforms.

    Returns
    -------
    train_transforms : monai.transforms.Compose
        A MONAI Compose object containing the specified sequence of transforms.

    Notes
    -----
    The specific probabilities and parameter ranges for each transformation are
    based on the empirical settings used in the paper. Play around with it!
    """
    
    train_transforms = Compose(
        [
            # Rescale to [0, 1]:
            ScaleIntensityd(keys=["view"]),
            # Apply bias fields:
            RandBiasFieldd(
                keys=["view"], prob=0.98, coeff_range=(0.0, 0.075),
            ),
            # Apply K-spikes:
            RandKSpaceSpikeNoised(keys=["view"], prob=0.2),
            # Apply gamma transforms:
            RandAdjustContrastd(keys=["view"], prob=0.5, gamma=(0.5, 2.)),
            # Apply smoothing:
            RandGaussianSmoothd(
                keys=["view"],
                prob=0.5,
                sigma_x=(0.0, 0.333),
                sigma_y=(0.0, 0.333),
                sigma_z=(0.0, 0.333),
            ),
            # Apply gibbs ringing (applies a box mask to kspace. alpha=0, box
            # width=1, i.e. no masking. alpha=1, boxwidth=0, i.e. all masked):
            RandGibbsNoised(keys=["view"], prob=0.5, alpha=(0.0, 0.333)),
            # Apply sharpening:
            RandGaussianSharpend(keys=["view"], prob=0.25),
            # Simulate much bigger voxels. MONAI does it nnUNet style, as
            # opposed to TorchIO's (IMO better) per-axis anisotropic style:
            RandSimulateLowResolutiond(keys=["view"], prob=0.333),
            # Clip out negative values:
            ThresholdIntensityd(
                keys=["view"], above=True, threshold=0.,
            ),
            # Rescale to [0, 1]:
            ScaleIntensityd(keys=["view"]),
        ]
    )
    return train_transforms


def draw_perlin_volume(
    out_shape,
    scales,
    min_std=0,
    max_std=1,
    dtype=torch.float32,
    device="cpu",
):
    """
    #TODO: merge draw_perlin_volume and draw_perlin_deformation
    
    Generates a 3D tensor with Perlin noise as defined in
    https://arxiv.org/abs/2004.10282

    Parameters
    ----------
    out_shape : tuple of int
        Shape of the output tensor (e.g., (D, H, W)).
    scales : float or list of float
        List of scales at which to generate the noise. 
        A single float can also be provided.
    min_std : float, optional
        Minimum standard deviation of the Gaussian noise. Default is 0.
    max_std : float, optional
        Maximum standard deviation of the Gaussian noise. Default is 1.
    dtype : torch.dtype, optional
        Data type of the output tensor. Default is torch.float32.
    device : str or torch.device, optional
        Device on which to create the tensor. Default is 'cpu'.

    Returns
    -------
    torch.Tensor
        A tensor of shape `out_shape` with generated Perlin noise.
    """
    out_shape = np.asarray(out_shape, dtype=np.int32)
    if np.isscalar(scales):
        scales = [scales]

    out = torch.zeros(tuple(out_shape), dtype=dtype, device=device)

    for scale in scales:
        sample_shape = np.ceil(out_shape / scale).astype(np.uint8)
    
        std = (max_std - min_std) * torch.rand(
            (1,), dtype=torch.float32, device=device
        )
        std = std + min_std
        gauss = std * torch.randn(
            tuple(sample_shape), dtype=torch.float32, device=device
        )
    
        zoom = [o // s for o, s in zip(out_shape, sample_shape)]
        if scale == 1:
            out += gauss
        else:
            out += torch.nn.functional.interpolate(
                gauss[None, None, ...],
                size=out.size(),
                # scale_factor=scale,
                mode='trilinear'
            )[0, 0, ...]

    return out


def minmax(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())

def sample_gmm(means, stds, label_map, zero_bckgnd=0.8, device="cuda"):
    """
    Generate a synthetic image using a Gaussian Mixture Model (GMM).
    
    This function creates a synthetic image where each region corresponding 
    to a unique label in the 3D synthetic `label_map` is filled with values
    from a Gaussian distribution characterized by the specified means and 
    standard deviations (`stds`). 
    
    100*zero_bckgnd % of the time, fill background label with zeros.

    Parameters
    ----------
    means : list or np.ndarray
        A list or array of means for the Gaussians, one for each label.
    stds : list or np.ndarray
        A list or array of std devs for the Gaussians, one for each label.
    label_map : np.ndarray
        A 3D array where each element corresponds to a label indicating the
        region in the synthetic image.
    zero_bckgnd : float
        Probability of filling background with zeros instead of intensities.

    Returns
    -------
    torch.Tensor
        A synthetic image/torch Tensor with values generated from the Gaussian 
        distributions, with values clipped to a minimum of 0 and scaled using 
        min-max normalization.
        
    """
    labels = np.unique(label_map)
    synthimage = torch.zeros(label_map.shape, requires_grad=False, device=device)

    for i, label in enumerate(labels):
        if (i == 0) and (label == 0) and (torch.rand(1) < zero_bckgnd):
            continue
        indices = label_map==label
        synthimage[indices] = stds[i] * torch.randn(indices.sum(), device=device) + means[i]

    synthimage = torch.clip(synthimage, min=0)
    synthimage = minmax(synthimage)

    return synthimage


def transform_uniform(arr, minval, maxval):
    """
    Transform arr from a uniform distribution in [0, 1] to [minval, maxval].
    """
    assert arr.min() >= 0
    assert arr.max() <= 1
    return (maxval - minval) * arr + minval