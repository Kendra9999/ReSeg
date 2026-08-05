# Copyright (c) Neel Dey
# Project Home: https://github.com/neel-dey/anatomix/
# Modified on 2026-08-05: Based on the above open-source project for secondary development
# Modified by: Derong Yu
import numpy as np
import nibabel as nib
import os
import argparse
import json
import copy
import json
import torch

import torch.multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

from glob import glob

from datagen_utils import get_foreground_mask

# -----------------------------------------------------------------------------
# generate 3D label maps:
def generate_label_map(seg_path, savedir, similar_label_mapping):
    """
    Generate 3D label maps with foreground masks, from the combined labels.
    Parameters
    ----------
    seg_path : str
        Path to the combined labels.
    savedir : str
        Directory where the generated label maps will be saved.
    """
    if os.path.exists(os.path.join(savedir, 'label_maps', seg_path.split('/')[-1])):
        print ('Skip {}'.format(seg_path))
        return

    print('Generate label map for {}'.format(seg_path))

    # Load the combined labels
    all_labels_nib = nib.load(seg_path)
    all_labels = all_labels_nib.get_fdata()

    all_labels = torch.from_numpy(all_labels).cuda()
    new_labels = torch.zeros_like(all_labels)
    # Combine similar structure to one label
    for k, v in similar_label_mapping.items():
        new_labels[all_labels == k] = v

    foreground_label = torch.zeros_like(new_labels)
    foreground_label[new_labels > 0] = 1

    # Generate foreground mask
    close_foreground = get_foreground_mask(foreground_label)

    label_map = copy.deepcopy(new_labels)
    label_map[close_foreground > 0] += 1

    # Save label map:
    nib.save(
        nib.Nifti1Image(
            label_map.cpu().numpy().astype(np.uint8),
            all_labels_nib.affine,
        ),
        os.path.join(savedir, 'label_maps', seg_path.split('/')[-1])
    )

    torch.cuda.empty_cache()


def get_similar_label_mapping(combined_labels):
    """
    Get a mapping of similar labels to one label.
    Parameters
    ----------
    combined_labels : dict
        A dictionary of combined labels.
    Returns
    -------
    similar_label_mapping : dict
        A dictionary of similar labels to one label.
    """
    similar_label_mapping = {}

    invert_dict = {v: k for k, v in combined_labels.items()}

    for idx, label_name in combined_labels.items():
        if "rib" in label_name:
            similar_label_mapping[int(idx)] = int(invert_dict["rib_left_1"])
        elif "vertebrae" in label_name or "sacrum" in label_name:
            similar_label_mapping[int(idx)] = int(invert_dict["vertebrae_C1"])
        elif "lung" in label_name:
            similar_label_mapping[int(idx)] = int(invert_dict["lung_lower_lobe_left"])
        elif "gluteus" in label_name:
            similar_label_mapping[int(idx)] = int(invert_dict["gluteus_maximus_left"])
        elif "right" in label_name:
            similar_label_mapping[int(idx)] = int(invert_dict[label_name.replace("right", "left")])
        else:
            similar_label_mapping[int(idx)] = int(idx)

    return similar_label_mapping


def main(savedir, max_workers=None):
    """
    Generate 3D label maps with foreground masks, from the combined labels.
    Parameters
    ----------
    savedir : str
        Directory where the generated label maps will be saved.
    max_workers : int, optional
        Maximum number of worker processes to use.
        Default is None, i.e., use all available CPU cores.
    """
    mp.set_start_method("spawn")

    with open(os.path.join(savedir, 'seg_labels_v1.json'), 'r') as f:
        combined_labels_v1 = json.load(f)

    with open(os.path.join(savedir, 'seg_labels_v2.json'), 'r') as f:
        combined_labels_v2 = json.load(f)

    # Get a mapping of similar labels to one label
    similar_label_mapping_v1 = get_similar_label_mapping(combined_labels_v1)
    print (similar_label_mapping_v1)
    similar_label_mapping_v2 = get_similar_label_mapping(combined_labels_v2)
    print (similar_label_mapping_v2)

    # Load the combined labels
    fpaths = sorted(glob(savedir + '/labels/*.nii.gz'))

    # Generate label maps
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                generate_label_map,
                fpath,
                savedir,
                similar_label_mapping_v1 if "v1" in fpath else similar_label_mapping_v2,
            )
            for fpath in fpaths
        ]
        for future in futures:
            future.result()


# -----------------------------------------------------------------------------
# Main script:


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--savedir', type=str, default='./Data_gen/',
        help='Directory to save the synthetic data',
    )
    parser.add_argument(
        '--max_workers',
        type=int,
        default=None,
        help='Maximum number of worker processes to use',
    )

    args = parser.parse_args()

    os.makedirs(os.path.join(args.savedir, 'label_maps'), exist_ok=True)

    # Run
    main(args.savedir, args.max_workers)