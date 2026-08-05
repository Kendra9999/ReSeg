import numpy as np
from monai.transforms import Transform
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    RandGaussianNoised,
    Rand3DElasticd,
    RandGaussianSmoothd,
    RandAxisFlipd,
    RandAffined,
    ScaleIntensityd
)

class RandZeroBackgroundd(Transform):
    """
    Set intensities in the volumes spatially coinciding with the background label to 0
    """
    def __init__(self, keys: list[str], label_key: str, prob: float = 0.75):
        self.keys = keys
        self.label_key = label_key
        self.prob = min(max(prob, 0.0), 1.0)

    def __call__(self, data):
        d = dict(data)
        if np.random.uniform() < self.prob:
            label = d[self.label_key]
            for key in self.keys:
                img = d[key]
                img[label == 0] = 0
                d[key] = img
        return d


def get_online_transforms():
    """
    Generates a MONAI composed transformation set for augmenting 
        the synthetic data online. 
    Refer to anatomix paper Fig.12 for the augmentation pipeline.
    """

    train_transforms = Compose(
        [
            LoadImaged(keys=["view1", "view2", "label"]),
            EnsureChannelFirstd(keys=["view1", "view2", "label"]),
            EnsureTyped(keys=["view1", "view2", "label"]),
            # Apply Gaussian noise:
            RandGaussianNoised(
                keys=["view1"],
                prob=0.333,
                mean=0.0,
                std=0.25,
            ),
            RandGaussianNoised(
                keys=["view2"],
                prob=0.333,
                mean=0.0,
                std=0.25,
            ),
            # # Apply K-space motion:
            # Rand3DElasticd(
            #     keys=["view1", "view2", "label"],
            #     prob=0.333,
            #     mode=['bilinear', 'bilinear', 'nearest'],
            #     sigma_range=(3.0, 5.0),
            #     magnitude_range=(10, 50),
            #     translate_range=(10, 10, 10),
            #     rotate_range=(np.pi/18, np.pi/18, np.pi/18),
            # ),
            # Rand3DElasticd(
            #     keys=["view1", "view2", "label"],
            #     prob=0.333,
            #     mode=['bilinear', 'bilinear', 'nearest'],
            #     sigma_range=(3.0, 5.0),
            #     magnitude_range=(10, 50),
            #     translate_range=(10, 10, 10),
            #     rotate_range=(np.pi/18, np.pi/18, np.pi/18),
            # ),
            # Apply Gaussian blur:
            RandGaussianSmoothd(
                keys=["view1"],
                prob=0.5,
                sigma_x=(0.0, 0.333),
                sigma_y=(0.0, 0.333),
                sigma_z=(0.0, 0.333),
            ),
            RandGaussianSmoothd(
                keys=["view2"],
                prob=0.5,
                sigma_x=(0.0, 0.333),
                sigma_y=(0.0, 0.333),
                sigma_z=(0.0, 0.333),
            ),
            # Zero background:
            RandZeroBackgroundd(
                keys=["view1"],
                label_key="label",
                prob=0.75,
            ),
            RandZeroBackgroundd(
                keys=["view2"],
                label_key="label",
                prob=0.75,
            ),
            # Apply axis flip:
            RandAxisFlipd(
                keys=["view1", "view2", "label"],
                prob=0.98,
            ),
            # Apply affine deformation:
            RandAffined(
                keys=["view1", "view2", "label"],
                prob=0.98,
                mode=['bilinear', 'bilinear', 'nearest'],
                translate_range=(16, 16, 16),
                rotate_range=(np.pi/4, np.pi/4, np.pi/4),
                scale_range=(0.4, 0.4, 0.4),
            ),
            # Rescale to [0, 1]:
            ScaleIntensityd(keys=["view1", "view2"]),
        ]
    )

    return train_transforms



if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt

    # image_fn = "foreground_masked_shapes31_ZYZ45ZR.nii.gz"
    # image_fn = "unconstrained_shapes38_QXN2T0J.nii.gz"
    image_fn = "foreground_masked_enveloped_shapes34_59WSV6Z.nii.gz"
    label_dir = "/data1/ydr/Project_representation/Data_gen/Totalsegmentator_gen_v1/label_ensembles"
    image_view1_dir = "/data1/ydr/Project_representation/Data_gen/Totalsegmentator_gen_v1/synthesized_views/view1"
    image_view2_dir = "/data1/ydr/Project_representation/Data_gen/Totalsegmentator_gen_v1/synthesized_views/view2"

    data = {"view1": os.path.join(image_view1_dir, "view1_" + image_fn),
            "view2": os.path.join(image_view2_dir, "view2_" + image_fn),
            "label": os.path.join(label_dir, image_fn)}
    
    transform_data = get_online_transforms()(data)

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].set_title("view1")
    ax[0].imshow(transform_data["view1"][0, :, :, 60], cmap="gray")
    ax[1].set_title("view2")
    ax[1].imshow(transform_data["view2"][0, :, :, 60], cmap="gray")
    ax[2].set_title("label")
    ax[2].imshow(transform_data["label"][0, :, :, 60], cmap="gray")
    plt.savefig("tmp_after_transform.png")


    load_transforms = Compose(
        [
            LoadImaged(keys=["view1", "view2", "label"]),
            EnsureChannelFirstd(keys=["view1", "view2", "label"]),
            EnsureTyped(keys=["view1", "view2", "label"]),
        ]
    )
    load_data = load_transforms(data)

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].set_title("view1")
    ax[0].imshow(load_data["view1"][0, :, :, 60], cmap="gray")
    ax[1].set_title("view2")
    ax[1].imshow(load_data["view2"][0, :, :, 60], cmap="gray")
    ax[2].set_title("label")
    ax[2].imshow(load_data["label"][0, :, :, 60], cmap="gray")
    plt.savefig("tmp_before_transform.png")
