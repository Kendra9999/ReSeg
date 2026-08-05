import os
import argparse
import nibabel as nib
import numpy as np
import json

abd4_labels = ["liver", "spleen", "kidney_right", "kidney_left"]


def main(data_dir):

    label_dir = os.path.join(data_dir, "labels")
    label_map_dir = os.path.join(data_dir, "label_maps")
    image_dir = os.path.join(data_dir, "synthesized_images")

    filename = os.listdir(label_dir)


    abd4_label_dir = os.path.join(data_dir, "labels_abd4")
    os.makedirs(abd4_label_dir, exist_ok=True)


    with open(os.path.join(data_dir, 'seg_labels_v1.json'), 'r') as f:
        all_labels_v1 = json.load(f)
    with open(os.path.join(data_dir, 'seg_labels_v2.json'), 'r') as f:
        all_labels_v2 = json.load(f)

    label_mapping_v1 = {}
    invert_label_dict_v1 = {v: k for k, v in all_labels_v1.items()}
    label_mapping_v2 = {}
    invert_label_dict_v2 = {v: k for k, v in all_labels_v2.items()}
    abd4_labels_dict = {0: "background"}
    for idx, label in enumerate(abd4_labels):
        label_mapping_v1[int(invert_label_dict_v1[label])] = idx + 1
        label_mapping_v2[int(invert_label_dict_v2[label])] = idx + 1
        abd4_labels_dict[idx + 1] = label
    print (label_mapping_v1)
    print (label_mapping_v2)

    with open(os.path.join(data_dir, 'seg_labels_abd4.json'), 'w') as f:
        json.dump(abd4_labels_dict, f, indent=4)



    for file in sorted(filename):
        if "v1" in file:
            label_mapping = label_mapping_v1
        elif "v2" in file:
            label_mapping = label_mapping_v2
        else:
            print (f"{file} dose not contain v1 or v2 in the name.")
            continue

        label_file = os.path.join(label_dir, file)
        label_nib = nib.load(label_file)
        label_data = label_nib.get_fdata().astype(np.uint8)

        label_idx = np.unique(label_data)
        if not np.any(np.isin(label_idx, list(label_mapping.keys()))):
            print (f"{file} dose not contain labels in the mapping.")
            continue
        
        print (f"{file} contain labels in the mapping.")

        new_label_data = np.zeros_like(label_data)
        for key, value in label_mapping.items():
            new_label_data[label_data == key] = value

        new_label_nib = nib.Nifti1Image(new_label_data.astype(np.uint8), label_nib.affine, label_nib.header)
        new_label_file = os.path.join(abd4_label_dir, file)
        nib.save(new_label_nib, new_label_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument(
        '--data_dir', type=str, default='./Data_gen/',
        help='Directory to save the synthetic data',
    )
    args = parser.parse_args()

    data_dir = args.data_dir

    main(data_dir)