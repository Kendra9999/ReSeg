from collections.abc import Mapping, Sequence

import torch

def collate(batch):
    """
    Puts each data field into a dict of
       Keys: "img": a dict of keys:
                    "image": a Tensor of shape (N, C, D, H, W)
                    "label": a Tensor of shape (N, 1, D, H, W)
             "img_metas": a list of dict of keys:
                    "filename": image file name
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")
    
    r_batch = {"img": {}, "img_metas": []}

    all_images = []
    all_labels = []

    for data in batch:
        for d in data:
            all_images.append(d["image"])
            all_labels.append(d["label"])
            r_batch["img_metas"].append(d["filename"])

    r_batch["img"]["image"] = torch.stack(all_images, dim=0)
    r_batch["img"]["label"] = torch.stack(all_labels, dim=0)
    
    return r_batch