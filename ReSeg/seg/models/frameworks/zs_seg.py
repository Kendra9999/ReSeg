import copy
import torch
import torch.nn.functional as F

from mmdet.models import build_detector
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
from mmdet.models.detectors.base import BaseDetector
from mmcv.runner import load_checkpoint, wrap_fp16_model

from monai.inferers import sliding_window_inference


@DETECTORS.register_module()
class ZS_Seg(BaseDetector):
    def __init__(self,
                 pretrained_model,
                 pretrained_checkpoint,
                 seg_decoder,
                 patch_size=(128, 128, 128),
                 segloss=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(ZS_Seg, self).__init__(init_cfg)
        self.load_pretrained_model(pretrained_model, pretrained_checkpoint)
        self.seg_decoder = build_neck(seg_decoder)
        self.segloss = build_loss(segloss)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.patch_size = patch_size

    def load_pretrained_model(self, pretrained_model, pretrained_checkpoint):
        pre_model = build_detector(pretrained_model)
        wrap_fp16_model(pre_model)
        checkpoint = load_checkpoint(pre_model, pretrained_checkpoint, map_location='cpu')

        self.seg_backbone = copy.deepcopy(pre_model.backbone)

        self.backbone = pre_model.backbone
        self.neck = pre_model.neck
        self.semantic_head = pre_model.semantic_head

        for param in self.backbone.parameters():
            param.requires_grad = False
        for param in self.neck.parameters():
            param.requires_grad = False
        for param in self.semantic_head.parameters():
            param.requires_grad = False
        

    def extract_feat(self, img):
        """Directly extract segmentation results."""
        x = self.backbone(img)
        out1 = self.neck(x[:self.neck.end_level])[0]
        out2 = self.semantic_head(x[:self.semantic_head.end_level])[0]

        x_aux = torch.cat([out1, out2], dim=1)


        x_seg = self.seg_backbone(img, return_first_emb=True)
        seg_out = self.seg_decoder(x_seg, x_aux)

        return seg_out
    

    def forward_train(self,
                      img,
                      img_metas,
                      **kwargs):
        """Forward function during training.
        Args:
            img (dict): a dict of keys:
                    "image": a Tensor of shape (N, C, D, H, W)
                    "label": a Tensor of shape (N, 1, D, H, W)

            img_metas (list[dict]): a list of dict of keys:
                    "filename": image file name
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        
        image = img["image"]
        label = img["label"]

        losses = dict()

        seg_out = self.extract_feat(image)
        label_list = [F.interpolate(label, size=seg_out[i].size()[2:], mode="nearest") 
                                    for i in range(len(seg_out))]
        seg_loss = self.segloss(seg_out, label_list)
        losses["loss"] = seg_loss

        return losses
    
    def simple_test(self, img, img_metas, proposals=None, rescale=False, vis=False):
        """Test without augmentation."""
        sw_batch_size = 2
        patch_size = self.patch_size

        test_outputs = sliding_window_inference(
                img, patch_size, sw_batch_size,
                self.extract_feat, overlap=0.7,
            )
        test_outputs = test_outputs.argmax(1)

        if vis:
            out1, out2, x_aux, attn_map = self.vis_feat(img, img_metas)
            return test_outputs, out1, out2, x_aux, attn_map

        return test_outputs

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)
    
    def vis_feat(self, img, img_metas):
        sw_batch_size = 2
        patch_size = self.patch_size

        # feature map
        out1, out2, x_aux = self.get_feat_map(img_metas[0]["img2"].unsqueeze(0))

        # attention map
        attn_map = sliding_window_inference(
                img, patch_size, sw_batch_size,
                self.get_attention_map, overlap=0.7,
            )

        return out1, out2, x_aux, attn_map


    def get_feat_map(self, img):
        x = self.backbone(img)
        out1 = self.neck(x[:self.neck.end_level])[0]
        out2 = self.semantic_head(x[:self.semantic_head.end_level])[0]

        x_aux = torch.cat([out1, out2], dim=1)

        return out1, out2, x_aux

    def get_attention_map(self, img):
        x = self.backbone(img)
        out1 = self.neck(x[:self.neck.end_level])[0]
        out2 = self.semantic_head(x[:self.semantic_head.end_level])[0]

        x_aux = torch.cat([out1, out2], dim=1)


        x_seg = self.seg_backbone(img, return_first_emb=True)
        attn_map = self.seg_decoder.get_attention_map(x_seg, x_aux)

        return attn_map

        

