# Copyright (c) Neel Dey
# Project Home: https://github.com/neel-dey/anatomix/
# Modified on 2026-08-05: Based on the above open-source project for secondary development
# Modified by: Derong Yu
import numpy as np
import nibabel as nib
import os
import argparse
import json

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from glob import glob

# -----------------------------------------------------------------------------
# Preprocessing label helpers:

def merge_labels_worker(segdir, savedir, combined_labels, tag='v1'):
    """
    Merge individual label files in a segmentation directory.

    Parameters
    ----------
    segdir : str
        The folder containing individual label nifti files.
    savedir : str
        The directory to save the combined labels.
    
    Returns
    -------
    None
    """
    if os.path.exists(os.path.join(savedir, 'labels', segdir.split('/')[-3] + f"_{tag}.nii.gz")):
        print(f"Labels already merged in {segdir}")
        return

    print(f"Merging labels in {segdir}")

    fpaths = sorted(glob(segdir + '/*.nii.gz'))
    dummy_for_metadata = nib.load(fpaths[0])
    
    # Empty arrays for aggregating labels:
    all_labels = np.zeros(dummy_for_metadata.shape)

    for idx, label_name in combined_labels.items():
        if int(idx) > 0:
            seg_file = os.path.join(segdir, f"{label_name}.nii.gz")
            segdata = nib.load(seg_file).get_fdata()
            
            all_labels[segdata == 1] = idx

    # Save merged labels:
    nib.save(
        nib.Nifti1Image(
            all_labels.astype(np.uint8),
            dummy_for_metadata.affine,
        ),
        os.path.join(savedir, 'labels', segdir.split('/')[-3] + f"_{tag}.nii.gz")
    )
        

def merge_labels(basedir, savedir, combined_labels, max_workers=None, tag='v1'):
    """
    Merge labels in all segmentation directories in the base directory.

    Parameters
    ----------
    basedir : str
        The base directory containing the segmentation directories.
    savedir : str
        The directory to save the combined labels.
    combined_labels : dict
        A dictionary mapping label numbers to their names.
    max_workers : int, optional
        The maximum number of worker processes to use.
        Default is None, i.e., use all available CPU cores.
    
    Returns
    -------
    None
    """
    
    segdirs = sorted(glob(basedir + '/**/segmentations/'))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                merge_labels_worker, 
                segdir,
                savedir,
                combined_labels,
                tag
            )
            for segdir in segdirs
        ]
        for future in futures:
            future.result()


def get_combined_labels_dict(basedir, savedir, segjson=None, tag='v1'):
    """
    Get a dictionary mapping label numbers to their names.
    Parameters
    ----------
    basedir : str
        The base directory containing the segmentation directories.
    savedir : str
        The directory to save the combined labels.
    segjson : str, optional
    """
    if segjson is None:
        segjson = os.path.join(savedir, f"seg_labels_{tag}.json")

        all_seg_files = os.listdir(os.path.join(basedir, 's0000', 'segmentations'))
        all_seg_name = sorted([seg_name.split('.',1)[0] for seg_name in all_seg_files])
        all_seg_dict = {0: "background"}
        for idx, name in enumerate(all_seg_name):
            all_seg_dict[idx+1] = name
        with open(segjson, 'w') as f:
            json.dump(all_seg_dict, f, indent=4)

    with open(segjson, 'r') as f:
        all_seg_name = json.load(f)

    return all_seg_name


# -----------------------------------------------------------------------------
# Main script:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--totalsegmentator_path_v1',
        type=str,
        default='./Totalsegmentator_dataset_v1/',
        help='Path to unzipped TotalSegmentator v1 dataset',
    )
    parser.add_argument(
        '--totalsegmentator_path_v2',
        type=str,
        default='./Totalsegmentator_dataset_v2/',
        help='Path to unzipped TotalSegmentator v2 dataset',
    )
    parser.add_argument(
        '--savedir', type=str, default='./Data_gen/',
        help='Directory to save the combined labels',
    )
    parser.add_argument(
        '--segjson', type=str, default=None, #default='./Data_gen/seg_labels.json',
        help='The combined labels dict json file',
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=None,
        help='Maximum number of worker processes to use',
    )

    args = parser.parse_args()

    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(os.path.join(args.savedir, 'labels'), exist_ok=True)
    
    # Get combined labels dict:
    combined_labels_v1 = get_combined_labels_dict(
                        args.totalsegmentator_path_v1,
                        args.savedir,
                        args.segjson,
                        tag = 'v1',
                    )
    combined_labels_v2 = get_combined_labels_dict(
                        args.totalsegmentator_path_v2,
                        args.savedir,
                        args.segjson,
                        tag = 'v2',
                    )
    
    # Merge all labels:
    mp.set_start_method("spawn")

    merge_labels(
        args.totalsegmentator_path_v1,
        args.savedir,
        combined_labels_v1,
        args.max_workers,
        tag = 'v1',
    )
    merge_labels(
        args.totalsegmentator_path_v2,
        args.savedir,
        combined_labels_v2,
        args.max_workers,
        tag = 'v2',
    )

    