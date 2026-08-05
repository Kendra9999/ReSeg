# Copyright (c) Neel Dey
# Project Home: https://github.com/neel-dey/anatomix/
# Modified on 2026-08-05: Based on the above open-source project for secondary development
# Modified by: Derong Yu
import torch
import numpy as np
import os
import random
import string
import argparse
import nibabel as nib

import torch.multiprocessing as mp

from glob import glob
from concurrent.futures import ProcessPoolExecutor

from datagen_utils import (
    get_transforms,
    sample_gmm,
    draw_perlin_volume,
    transform_uniform,
)


# -----------------------------------------------------------------------------
# generate images:

def process_volume(
    lab,
    means_range,
    stds_range,
    perl_scales,
    perl_max_std,
    perl_mult_factor,
    savedir,
    seed,
):
    """
    Process a single volume: load the label ensemble, generate synthetic views,
    apply transformations, and save the outputs.
    
    Parameters
    ----------
    lab : str
        Path to the label ensemble nifti file.
    means_range : tuple of int
        Range of means for the Gaussian distributions.
    stds_range : tuple of int
        Range of standard deviations for the Gaussians.
    perl_scales : tuple of int
        Scales for generating Perlin-like noise.
    perl_max_std : float
        Maximum standard deviation for Perlin-like noise.
    perl_mult_factor : float
        Multiplicative constant applied to sampled Perlin-like noise.
    savedir : str
        Directory where the output synthetic volumes/views will be saved.
    seed : int
        Random seed for the process.
    
    Returns
    -------
    None
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    print(
        'Synthesizing ensemble {} with seed {}'.format(
            os.path.basename(lab), seed,
        )
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize MONAI augmentation pipeline:
    transforms = get_transforms()

    current_label_nib = nib.load(lab)
    current_label = current_label_nib.get_fdata()
    labels = np.unique(current_label)

    # Sample random means and std devs for a volume:
    means = transform_uniform(
        torch.rand(len(labels)), means_range[0], means_range[1],
    )
    
    stds = transform_uniform(
        torch.rand(len(labels)), stds_range[0], stds_range[1],
    )

    # Sample a volume from the specified GMMs:
    synthview = sample_gmm(means, stds, current_label, device=device)

    # Sample Perlin-like noise to simulate spatial structure in texture:
    randperl_view = draw_perlin_volume(
        out_shape=current_label.shape,
        scales=perl_scales,
        max_std=perl_max_std,
        device=device,
    )

    # Pointwise multiply with Perlin-like noise and downscale intensities 
    # by `perl_mult_factor`:
    synthperl = synthview * (1 + perl_mult_factor * randperl_view)
    
    # Create data dict and send to MONAI augmentation pipeline:
    inputimgs = {
        "view": synthperl, "label": current_label,
    }
    outputs = transforms(inputimgs)

    # Save synthetic volumes as nifti files:
    # Saving as uint8 volumes to not blow up disk usage.
    fpath_dir = os.path.join(savedir, 'synthesized_images', os.path.basename(lab).split('.')[0])
    os.makedirs(fpath_dir, exist_ok=True)
    randstr = ''.join(
        random.choices(string.ascii_uppercase + string.digits, k=7)
    )
    fpath = os.path.join(fpath_dir, f"{randstr}.nii.gz")
    while os.path.isfile(fpath):
        randstr = ''.join(
            random.choices(string.ascii_uppercase + string.digits, k=7)
        )
        fpath = os.path.join(fpath_dir, f"{randstr}.nii.gz")
    
    nib.save(
        nib.Nifti1Image(
            outputs['view'].cpu().numpy(),
            affine=current_label_nib.affine,
        ),
        fpath,
    )

    torch.cuda.empty_cache()


def run(
    per_vol_num,
    savedir,
    means_range=(25, 225, 255),
    stds_range=(5, 20),
    perl_scales=(4, 8, 16, 32),
    perl_max_std=5.,
    perl_mult_factor=0.02,
    max_workers=None,
):
    """
    Generate several synthetic volumes from pre-generated label ensembles using
    the appearance model from the paper.
    
    Given a 3D label map, we sample means and standard deviations at random
    from which to sample initial intensities. These initial volumes are then extensively
    augmented to create highly variable training volumes.
    
    Takes a path to pre-generated label files,
    and synthesizes nifti files for the specified number of volumes.

    Parameters
    ----------
    per_vol_num : int
        Synthesized image number for each label map.
    savedir : str
        Directory where the output synthetic views will be saved.
    means_range : tuple of int, optional
        Range of means for the Gaussian distributions. Default [25, 255].
    stds_range : tuple of int, optional
        Range of standard deviations for the Gaussians. Default [5, 20].
    perl_scales : tuple of int, optional
        Scales for generating Perlin noise. Default (4, 8, 16, 32).
    perl_max_std : float, optional
        Maximum standard deviation for Perlin noise. Default 5.0.
    perl_mult_factor : float, optional
        Multiplication factor for Perlin noise. Default 0.02.
    max_workers : int, optional
        Maximum number of worker processes to use. 
        Default is None, which uses all available resources.

    Returns
    -------
    None
    """
    mp.set_start_method("spawn")

    # Load list of precomputed label ensembles:
    labs = sorted(glob(savedir + '/label_maps/*.nii.gz'))
    assert len(labs) > 0

    # Generate random seeds for each process
    random_seeds = np.random.choice(
        range(1, len(labs)*per_vol_num*100), size=len(labs)*per_vol_num, replace=False,
    )

    labs_rp = labs * per_vol_num

    # Generate random means range
    random_means_range = np.random.randint(
        means_range[0], means_range[1], size=len(labs)*per_vol_num,
    )

    # Process volumes in parallel:
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_volume,
                lab,
                (mean_low, means_range[2]),
                stds_range,
                perl_scales,
                perl_max_std,
                perl_mult_factor,
                savedir,
                seed
            )
            for lab, seed, mean_low in zip(labs_rp, random_seeds, random_means_range)
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--per_vol_num', type=int, default=30,
        help='Synthesized images for each label map',
    )
    parser.add_argument(
        '--savedir', type=str, default='./Data_gen/',
        help='Directory to save the synthetic data',
    )
    parser.add_argument(
        '--max_workers', type=int, default=3,
        help='Maximum number of worker processes to use',
    )

    args = parser.parse_args()

    os.makedirs(os.path.join(args.savedir, 'synthesized_images'), exist_ok=True)

    run(
        args.per_vol_num,
        savedir=args.savedir,
        max_workers=args.max_workers
    )
