import copy
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import Transform
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    CropForegroundd,
    SpatialPadd,
    RandSpatialCropd,
    RandCropByPosNegLabeld,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandSimulateLowResolutiond,
    RandAdjustContrastd,
    RandFlipd,
    RandRotate90d,
    RandAffined,
    Rand3DElasticd,
    ScaleIntensityd,
)

class RandSimulateLowResolutiond_per_axis(Transform):
    """
    Simulate low resolution by zooming in the image along each axis.
    """
    def __init__(self, keys: list[str], prob: float = 0.5, zoom_range: tuple = (0.25, 1.0)):
        self.keys = keys
        self.prob = min(max(prob, 0.0), 1.0)
        self.zoom_range = zoom_range

    def __call__(self, data):
        d = dict(data)

        if np.random.uniform() < self.prob:
            scale = tuple(np.random.uniform(self.zoom_range[0], self.zoom_range[1], size=3).tolist())

            for key in self.keys:
                ori_img = d[key].unsqueeze(0)
                down_img = F.interpolate(ori_img, scale_factor=scale, mode='nearest')
                up_img = F.interpolate(down_img, size=ori_img.shape[2:], mode='trilinear', align_corners=False)
                d[key] = up_img.squeeze(0)

        return d


class GenerateSampleCenterMask(Transform):
    """
    Generates a mask for monai.transforms.RandCropByPosNegLabeld
    """
    def __init__(self, label_key: str, mask_key: str):
        self.label_key = label_key
        self.mask_key = mask_key

    def __call__(self, data):
        d = dict(data)
        # Get the label:
        label = d[self.label_key]
        
        try:
            foreground_mask = torch.where(label > 0)
            x_low, x_high = torch.min(foreground_mask[1]), torch.max(foreground_mask[1])
            y_low, y_high = torch.min(foreground_mask[2]), torch.max(foreground_mask[2])
            z_low, z_high = torch.min(foreground_mask[3]), torch.max(foreground_mask[3])
        
            sample_mask = copy.deepcopy(label)
            sample_mask[:, x_low:x_high+1, y_low:y_high+1, z_low:z_high+1] = 1
            
        except:
            sample_mask = copy.deepcopy(label)
            
        d[self.mask_key] = sample_mask
        return d

    

def get_transforms(patch_size=(128, 128, 128), spacing=(3., 3., 3.), sw_batch_size=1):
    """
    Generates a MONAI composed transformation set for augmenting 
        the synthetic seg data online. 
    """
    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            EnsureTyped(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")
            ),
            # Rescale to [0, 1]:
            ScaleIntensityd(keys=["image"]),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant"),
            # RandSpatialCropd(
            #     keys=["image", "label"],
            #     roi_size=patch_size,
            # ),
            GenerateSampleCenterMask(label_key="label", mask_key="sample_mask"),
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="sample_mask",
                spatial_size=patch_size,
                pos=9,
                neg=1,
                num_samples=sw_batch_size,
            ),
            # Apply Gaussian noise:
            RandGaussianNoised(
                keys=["image"],
                prob=0.333,
                mean=0.0,
                std=0.1,
            ),
            RandAdjustContrastd(keys=["image"], prob=0.333),
            # Apply Gaussian blur:
            RandGaussianSmoothd(
                keys=["image"],
                prob=0.5,
                sigma_x=(0.0, 0.333),
                sigma_y=(0.0, 0.333),
                sigma_z=(0.0, 0.333),
            ),
            # Simulate low resolution (some MRI cases):
            RandSimulateLowResolutiond_per_axis(keys=["image"], prob=0.4, zoom_range=(0.25, 1.0)),
            # Apply elastic deformation:
            Rand3DElasticd(
                keys=["image", "label"],
                prob=0.333,
                mode=['bilinear', 'nearest'],
                sigma_range=(5.0, 8.0),
                magnitude_range=(10, 50),
                translate_range=(10, 10, 10),
                rotate_range=(np.pi/18, np.pi/18, np.pi/18),
            ),
            # Apply affine deformation:
            RandAffined(
                keys=["image", "label"],
                prob=0.98,
                mode=['bilinear', 'nearest'],
                translate_range=(10, 10, 10),
                rotate_range=(np.pi/4, np.pi/4, np.pi/4),
                scale_range=(0.25, 0.25, 0.25),
                shear_range=(0.2, 0.2, 0.2),
            ),
            # Apply axis flip:
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            # Rescale to [0, 1]:
            ScaleIntensityd(keys=["image"]),
        ]
    )

    return train_transforms


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    image_path = "/mnt/sdb/drong/Project_representation/Data_gen/Totalsegmentator_gen_seg/synthesized_images/s1405_v2/F7NKRC3.nii.gz"
    label_path = "/mnt/sdb/drong/Project_representation/Data_gen/Totalsegmentator_gen_seg/labels_abd4/s1405_v2.nii.gz"
    data = {"image": image_path, "label": label_path}
    data = get_transforms()(data)
    print(data[0]["image"].shape, data[0]["label"].shape)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].set_title("image1")
    ax[0].imshow(data[0]["image"][0, :, :, 60], cmap="gray")
    ax[1].set_title("label1")
    ax[1].imshow(data[0]["label"][0, :, :, 60], cmap="gray")
    # ax[2].set_title("image2")
    # ax[2].imshow(data[1]["image"][0, :, :, 60], cmap="gray")
    # ax[3].set_title("label2")
    # ax[3].imshow(data[1]["label"][0, :, :, 60], cmap="gray")
    plt.savefig("tmp_after_transform.png")