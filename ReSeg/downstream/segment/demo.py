import argparse
import os
import sys
sys.path.append('..')
sys.path.append('../..')

from mmcv import Config
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
    Invertd,
    AsDiscreted,
    SaveImaged,
)

from utils import *

os.chdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))  # go to root dir of this project

def parse_args():
    parser = argparse.ArgumentParser(description='Zero-shot Segmentation')
    parser.add_argument('--config', type=str, 
                        default='configs/seg/zs_seg_abd4.py', 
                        help='train config file path')
    parser.add_argument('--checkpoint', type=str, 
                        default='work_dirs/works/zs_seg_abd4/20250520_121635/best_Dice_iter_72000.pth', 
                        help='pretrain checkpoint file path')
    parser.add_argument('--img-file', type=str, 
                        default='/mnt/sdb/drong/Project_representation/Data_gen/Totalsegmentator_gen_seg/synthesized_images/s0189_v2/ADD2A4M.nii.gz',
                        help='image file path')
    parser.add_argument('--save-dir', type=str, 
                        default='work_dirs/seg_results/', 
                        help='save segment results directory path')
    args = parser.parse_args()
    return args


def load_image(img_file, patch_size=(128, 128, 128), spacing=(3., 3., 3.),
               output_dir='work_dirs/seg_results/'):
    load_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            EnsureTyped(keys=["img"]),
            Orientationd(keys=["img"], axcodes="RAS"),
            Spacingd(keys=["img"], pixdim=spacing, mode=("bilinear")),
            # Rescale to [0, 1]:
            ScaleIntensityd(keys=["img"]),
            CropForegroundd(keys=["img"], source_key="img"),
            SpatialPadd(keys=["img"], spatial_size=patch_size, mode="constant"),
        ]
    )

    data = {"img": img_file, "img_metas": {"filename": os.path.basename(img_file)}}
    data = load_transforms(data)
    
    batch_data = {"img": [data["img"].unsqueeze(0)],
                  "img_metas": [[data["img_metas"]]]}
    
    
    post_transforms = Compose([EnsureTyped(keys=["pred"]),
                               Invertd(keys=["pred"],
                                       transform=load_transforms,
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

    return batch_data, data, post_transforms

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    save_folder = os.path.join(args.save_dir, '_'.join(os.path.splitext(args.checkpoint)[0].split('/')[2:]))
    os.makedirs(save_folder, exist_ok=True)
    
    # load model
    model = init(args.config, args.checkpoint)

    # load image
    batch_data, data, post_transforms = load_image(args.img_file,
                                      patch_size=cfg.transform_kwargs.patch_size,
                                      spacing=cfg.transform_kwargs.spacing,
                                      output_dir=save_folder)

    seg_pred = get_embedding(batch_data, model)
    data["pred"] = seg_pred.cpu()
    post_transforms(data)


if __name__ == '__main__':
    main()