import os

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    CropForegroundd,
    SpatialPadd,
    AdjustContrastd,
)

from torch.utils.data import Dataset

from mmdet.datasets.builder import DATASETS


@DATASETS.register_module()
class ValidationDataset(Dataset):
    def __init__(self, image_dir, label_dir, num_classes,
                 patch_size=(160, 160, 160), spacing=(2.5, 2.5, 2.5)):
        self.num_classes = num_classes
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_files = sorted(os.listdir(image_dir))
        
        self.val_files = [{"image": os.path.join(image_dir, f), 
                           "label": os.path.join(label_dir, f[:-12] + ".nii.gz")} 
                           for f in self.image_files]

        self.load_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
                Orientationd(keys=["image", "label"], axcodes="RAS"),
                Spacingd(keys=["image", "label"], pixdim=spacing, mode=("bilinear", "nearest")),
                # Rescale to [0, 1]:
                # ScaleIntensityd(keys=["image"]),
                ScaleIntensityRanged(keys=["image"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=["image", "label"], source_key="image"),
                SpatialPadd(keys=["image", "label"], spatial_size=patch_size, mode="constant"),
                # AdjustContrastd(keys=["image"], gamma=2.0),
            ]
        )
        
    def __len__(self):
        return len(self.val_files)
    
    def __getitem__(self, index):
        data = self.val_files[index]
        data["img_metas"] = {"filename": os.path.basename(self.image_files[index])}
        data = self.load_transforms(data)
        return data