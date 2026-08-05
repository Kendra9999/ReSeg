# Copyright (c) Medical AI Lab, Alibaba DAMO Academy

import os
import random
import numpy as np
import logging
import json
from mmdet.datasets.builder import DATASETS

from .transforms.transforms import get_transforms


@DATASETS.register_module()
class Dataset3dSynSeg(object):
    CLASSES = None

    def __init__(
            self, data_dir, transform_kwargs, seg_label_dir, seg_label_dict_json,
            test_mode=False, multisets=False, set_length=1000,):
        self.logger = logging.getLogger(__name__)

        self.data_path = data_dir
        self.loaddatalist(data_dir, seg_label_dir, seg_label_dict_json)
        self.num_image_pairs = len(self.filename)
        self.multisets = multisets
        self.sample = False
        if self.multisets:
            self.set_length = set_length
        self.transforms = get_transforms(**transform_kwargs)
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
        image_name = random.choice(self.image_label_dict[image_fn])
        image_path = os.path.join(self.image_dir, image_fn.split(".")[0], image_name)

        data = self.prepare_train_img(label_path, image_path, image_fn)
        return data
    
    def prepare_train_img(self, label_path, image_path, image_fn):
        data = {"image": image_path,
                "label": label_path,
                "filename": image_fn}
        data = self.transforms(data)
        return data
    

    def loaddatalist(self, data_dir, seg_label_dir, seg_label_dict_json):
        self.label_dir = os.path.join(data_dir, seg_label_dir)
        self.filename = os.listdir(self.label_dir)

        self.image_dir = os.path.join(data_dir, "synthesized_images")
        self.image_label_dict = {}
        for file in self.filename:
            image_files_dir = os.path.join(self.image_dir, file.split(".")[0])
            image_files = os.listdir(image_files_dir)
            self.image_label_dict[file] = image_files


        with open(os.path.join(data_dir, seg_label_dict_json), 'r') as f:
            self.class_labels = json.load(f)