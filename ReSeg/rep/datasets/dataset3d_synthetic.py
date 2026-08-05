# Copyright (c) Medical AI Lab, Alibaba DAMO Academy

import os
import numpy as np
import logging
import csv
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.pipelines import Compose
import pickle

from .transforms.online_transforms import get_online_transforms
from .transforms.crop_transforms import get_crop_transforms


@DATASETS.register_module()
class Dataset3dSynthetic(object):
    CLASSES = None

    def __init__(
            self, data_dir, crop_transform=False, crop_transform_kwargs=None,
            test_mode=False, multisets=False, set_length=1000,):
        self.logger = logging.getLogger(__name__)

        self.data_path = data_dir
        self.loaddatalist(data_dir)
        self.num_image_pairs = len(self.filename)
        self.multisets = multisets
        self.sample = False
        if self.multisets:
            self.set_length = set_length
        self.transforms = get_online_transforms()
        self.crop_transforms = None
        if crop_transform:
            self.crop_transforms = get_crop_transforms(**crop_transform_kwargs)
        self.test_mode = test_mode
        self._set_group_flag()
    
    def __len__(self):
        if self.multisets:
            return self.set_length
        else:
            return self.num_image_pairs

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.
        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)

    def pre_pipeline(self, results):
        return results
    
    def __getitem__(self, index):
        if self.multisets:
            loc = np.random.randint(0, self.num_image_pairs)
        else:
            loc = index
        image_fn = self.filename[loc]

        assert not self.test_mode, "Not support test mode"

        label_path = os.path.join(self.label_dir, image_fn)
        image_view1_path = os.path.join(self.image_view1_dir, "view1_" + image_fn)
        image_view2_path = os.path.join(self.image_view2_dir, "view2_" + image_fn)

        data = self.prepare_train_img(label_path, image_view1_path, image_view2_path, image_fn)
        return data
    
    def prepare_train_img(self, label_path, image_view1_path, image_view2_path, image_fn):
        data = {"view1": image_view1_path, 
                "view2": image_view2_path, 
                "label": label_path}
        data = self.transforms(data)
        if self.crop_transforms is not None:
            data = self.crop_transforms(data)
        return data


    def loaddatalist(self, data_dir):
        self.label_dir = os.path.join(data_dir, "label_ensembles")
        self.image_view1_dir = os.path.join(data_dir, "synthesized_views", "view1")
        self.image_view2_dir = os.path.join(data_dir, "synthesized_views", "view2")

        self.filename = os.listdir(self.label_dir)
        self.logger.info("Use a synthetic dataset having {} image pairs".format(len(self.filename)))
