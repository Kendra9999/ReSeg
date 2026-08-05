import argparse
import os
import sys
sys.path.append('..')
sys.path.append('../..')

from mmcv import Config
from monai.transforms import Transform
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    ScaleIntensityd,
    CropForegroundd,
    SpatialPadd,
    AdjustContrastd,
    Invertd,
    AsDiscreted,
    SaveImaged,
)
from torch.utils.data import Dataset, DataLoader

from utils import *

os.chdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))  # go to root dir of this project

def parse_args():
    parser = argparse.ArgumentParser(description='Zero-shot Segmentation')
    parser.add_argument('--config', type=str, 
                        default='configs/seg/zs_seg_abd4.py', 
                        help='train config file path')
    parser.add_argument('--checkpoint', type=str, 
                        default='work_dirs/works/zs_seg_abd4/20260420_232836/iter_75000.pth', 
                        help='pretrain checkpoint file path')
    parser.add_argument('--data-dir', type=str, 
                        default="/mnt/sdb/drong/Project_nnUNet/DATASET/nnUNet_raw/Dataset003_L2RAb_MRI/imagesTs/",
                        help='test data directory path')
    parser.add_argument('--save-dir', type=str, 
                        default='work_dirs/seg_results/', 
                        help='save segment results directory path')
    args = parser.parse_args()
    return args


class ClipIntensityd(Transform):
    """
    Clip intensities in the volumes to a range 
    """
    def __init__(self, keys: list[str], hist_cut_top: float = 0.995):
        self.keys = keys
        self.hist_cut_top = hist_cut_top

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            hir = torch.quantile(img, self.hist_cut_top)
            img[img > hir] = hir
            d[key] = img
        return d


class TestDataset(Dataset):
    def __init__(self, data_dir, patch_size=(128, 128, 128), spacing=(3., 3., 3.),
               output_dir='work_dirs/seg_results/'):
        self.data_dir = data_dir
        self.test_files = sorted(os.listdir(data_dir))

        self.load_transforms = Compose(
            [
                LoadImaged(keys=["img"]),
                EnsureChannelFirstd(keys=["img"]),
                EnsureTyped(keys=["img"]),
                Orientationd(keys=["img"], axcodes="RAS"),
                Spacingd(keys=["img"], pixdim=spacing, mode=("bilinear")),
                # Rescale to [0, 1]:
                ClipIntensityd(keys=["img"], hist_cut_top=0.995),
                ScaleIntensityd(keys=["img"]),
                CropForegroundd(keys=["img"], source_key="img"),
                SpatialPadd(keys=["img"], spatial_size=patch_size, mode="constant"),
                AdjustContrastd(keys=["img"], gamma=0.5),
            ]
        )

        self.post_transforms = Compose([EnsureTyped(keys=["pred"]),
                               Invertd(keys=["pred"],
                                       transform=self.load_transforms,
                                       orig_keys="img",
                                       meta_keys="pred_meta_dict",
                                       orig_meta_keys="img_meta_dict",
                                       meta_key_postfix="meta_dict",
                                       nearest_interp=True,
                                       to_tensor=True),
                               AsDiscreted(keys="pred", argmax=False, to_onehot=None),
                               SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_dir,
                                          separate_folder=False, output_postfix="",
                                          resample=False),
                               ])
        
    def __len__(self):
        return len(self.test_files)
    
    def __getitem__(self, index):
        test_file = self.test_files[index]
        test_path = os.path.join(self.data_dir, test_file)
        data = {"img": test_path, "img_metas": {"filename": os.path.basename(test_path)}}
        data = self.load_transforms(data)
        return data



def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    save_folder = os.path.join(args.save_dir, '_'.join(os.path.splitext(args.checkpoint)[0].split('/')[2:]),
                               "L2RAb_MR")
    os.makedirs(save_folder, exist_ok=True)

    # load model
    model = init(args.config, args.checkpoint)

    # load test data
    test_dataset = TestDataset(args.data_dir, 
                               patch_size=cfg.transform_kwargs.patch_size,
                               spacing=cfg.transform_kwargs.spacing,
                               output_dir=save_folder)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=12)

    for data in test_dataloader:
        batch_data = {"img": [data["img"]],
                  "img_metas": [[data["img_metas"]]]}
        
        seg_pred = get_embedding(batch_data, model)

        data["pred"] = seg_pred.cpu()
        test_dataset.post_transforms(data)

    print(f"Segmentation results saved to {save_folder}")
    for file in os.listdir(save_folder):
        if file.endswith(".nii.gz"):
            os.rename(os.path.join(save_folder, file), os.path.join(save_folder, file.replace("_0000.nii.gz", ".nii.gz")))
    


if __name__ == '__main__':
    main()