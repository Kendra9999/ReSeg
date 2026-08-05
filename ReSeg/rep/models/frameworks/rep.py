# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
# Project Home: https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2
# Modified on 2026-08-05: Based on the above open-source project for secondary development
# Modified by: Derong Yu
import os.path
import warnings

# import ipdb
import torch
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
from mmdet.models.detectors.base import BaseDetector
import torch.nn.functional as F
from torch import linalg as LA
import time
import pickle
import numpy as np


@DETECTORS.register_module()
class Rep(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 sem_neck=None,
                 patchloss=None,
                 superloss=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(Rep, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.backbone.init_weights()
        if neck is not None:
            self.neck = build_neck(neck)
        # self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.patchcriterion = build_loss(patchloss)
        self.supcriterion = build_loss(superloss)
        if sem_neck is not None:
            self.semantic_head = build_neck(sem_neck)
        else:
            self.semantic_head = build_neck(neck)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    def extract_feat(self, img, normalize=True):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        out1 = self.neck(x[:self.neck.end_level])[0]
        out2 = self.semantic_head(x[:self.semantic_head.end_level])[0]
        if normalize:
            out1 = F.normalize(out1, dim=1)
            out2 = F.normalize(out2, dim=1)
        out1 = out1.type(torch.half)
        out2 = out2.type(torch.half)
        return [out1, out2]
    
    def forward_train(self,
                      img,
                      img_metas,
                      **kwargs):
        """Forward function during training.
        Args:
            img (dict): a dict of keys:
                    "overlap_patches": a Tensor of shape (N, C, D, H, W)
                    "overlap_patches_labels": a Tensor of shape (N, 1, D, H, W)
                    "overlap_patches_girds": a Tensor of shape (N, 3, D, H, W)
                    "whole_images": a Tensor of shape (N, C, D, H, W)
                    "whole_images_labels": a Tensor of shape (N, 1, D, H, W)

            img_metas (list[dict]): a list of dict of keys:
                    "style": "overlap" or "whole"
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        overlap_patches = img["overlap_patches"]
        overlap_patches_labels = img["overlap_patches_labels"]
        overlap_patches_girds = img["overlap_patches_girds"]
        whole_images = img["whole_images"]
        whole_images_labels = img["whole_images_labels"]

        losses = dict()
        
        # forward on overlap patches
        overlap_feats = self.extract_feat(overlap_patches)
        appear_loss = self.loss_appearance(overlap_feats,
                                           overlap_patches_labels, overlap_patches_girds)
        losses["appear_loss"] = appear_loss

        # forward on whole images
        whole_feats = self.extract_feat(whole_images)
        seman_loss = self.loss_semantic(whole_feats, whole_images_labels)
        losses["seman_loss"] = seman_loss

        return losses
    
    def loss_appearance(self, feats, labels, meshgrids, **kwargs):

        N, C, D, H, W = meshgrids.shape
        label_half = F.interpolate(labels, size=(int(D / 2), int(H / 2), int(W / 2)), mode='nearest')
        grid_half = F.interpolate(meshgrids, size=(int(D / 2), int(H / 2), int(W / 2)), mode='trilinear', align_corners=False)
        
        label_half = label_half.type(torch.half)
        grid_half = grid_half.type(torch.half)
        
        for i in range(int(N / 6)):
            result = self.single_appear_fine_loss(feats[0][6 * i : 6 * (i + 1)], 
                                                  label_half[6 * i : 6 * (i + 1)], 
                                                  grid_half[6 * i : 6 * (i + 1)])
            if i == 0:
                fine_loss = result['loss']
            else:
                fine_loss += result['loss']
        fine_loss = fine_loss / int(N / 6)

        return fine_loss
    

    def single_appear_fine_loss(self, fine_feat, fine_label, fine_grid):
        out = dict()

        N_views = fine_feat.shape[0]  # 6

        fine_local_grid = self.meshgrid3d(fine_feat.shape[2:], device=fine_feat.device).long()  # z y x

        list_y_min = [fine_grid[i, 0].min() for i in range(N_views)]
        list_y_max = [fine_grid[i, 0].max() for i in range(N_views)]
        list_x_min = [fine_grid[i, 1].min() for i in range(N_views)]
        list_x_max = [fine_grid[i, 1].max() for i in range(N_views)]
        list_z_min = [fine_grid[i, 2].min() for i in range(N_views)]
        list_z_max = [fine_grid[i, 2].max() for i in range(N_views)]

        intersection_y = [max(list_y_min), min(list_y_max)]
        intersection_x = [max(list_x_min), min(list_x_max)]
        intersection_z = [max(list_z_min), min(list_z_max)]
        
        # defintely have overlap

        list_intersection_volume = [(fine_grid[i, 0] >= intersection_y[0]) * (
                                     fine_grid[i, 0] <= intersection_y[1]) \
                                  * (fine_grid[i, 1] >= intersection_x[0]) * (
                                     fine_grid[i, 1] <= intersection_x[1]) \
                                  * (fine_grid[i, 2] >= intersection_z[0]) * (
                                     fine_grid[i, 2] <= intersection_z[1])
                                    for i in range(N_views)]
        
        # view 0: random select anchor points
        index_overlap_0 = fine_local_grid[list_intersection_volume[0] > 0, :]
        pos_mm_overlap_0 = fine_grid[0, :, list_intersection_volume[0] > 0]

        points_select = torch.randperm(index_overlap_0.shape[0], device=fine_feat.device)
        points_select = points_select[:int(min(self.train_cfg.intra_cfg.select_anchor_number, points_select.shape[0]))]

        pos_mm_anchor_0 = pos_mm_overlap_0[:, points_select]

        x_sample_anchor_0 = fine_feat[0, :, index_overlap_0[points_select, 0], index_overlap_0[points_select, 1], index_overlap_0[points_select, 2]].permute(1, 0).contiguous()
        label_sample_anchor_0 = fine_label[0, :, index_overlap_0[points_select, 0], index_overlap_0[points_select, 1], index_overlap_0[points_select, 2]].squeeze(0)

        # find the nearest anchor points in other views
        x_sample_anchors = [x_sample_anchor_0]
        label_sample_anchors = [label_sample_anchor_0]
        for i in range(1, N_views):
            index_overlap_i = fine_local_grid[list_intersection_volume[i] > 0, :]
            pos_mm_overlap_i = fine_grid[i, :, list_intersection_volume[i] > 0]

            dist = LA.norm((pos_mm_anchor_0.view(-1, pos_mm_anchor_0.shape[1], 1) - \
                            pos_mm_overlap_i.view(-1, 1, pos_mm_overlap_i.shape[1])), dim=0)
            
            anchor_points_i = dist.min(dim=1)[1]

            x_sample_anchor_i = fine_feat[i, :, index_overlap_i[anchor_points_i, 0], index_overlap_i[anchor_points_i, 1], index_overlap_i[anchor_points_i, 2]].permute(1, 0).contiguous()
            label_sample_anchor_i = fine_label[i, :, index_overlap_i[anchor_points_i, 0], index_overlap_i[anchor_points_i, 1], index_overlap_i[anchor_points_i, 2]].squeeze(0)

            x_sample_anchors.append(x_sample_anchor_i)
            label_sample_anchors.append(label_sample_anchor_i)

        x_sample_anchors = torch.cat(x_sample_anchors, dim=0)
        label_sample_anchors = torch.cat(label_sample_anchors, dim=0)

        # random select positive and negative points
        x_sample_others = []
        label_sample_others = []
        for i in range(N_views):
            index_all = fine_local_grid.view(-1, 3)
            points_select_all = torch.randperm(index_all.shape[0], device=fine_feat.device)
            points_select_all = points_select_all[:int(min(self.train_cfg.intra_cfg.select_other_number, points_select_all.shape[0]))]

            x_sample_other_i = fine_feat[i, :, index_all[points_select_all, 0], index_all[points_select_all, 1], index_all[points_select_all, 2]].permute(1, 0).contiguous()
            label_sample_other_i = fine_label[i, :, index_all[points_select_all, 0], index_all[points_select_all, 1], index_all[points_select_all, 2]].squeeze(0)

            x_sample_others.append(x_sample_other_i)
            label_sample_others.append(label_sample_other_i)

        x_sample_others = torch.cat(x_sample_others, dim=0)
        label_sample_others = torch.cat(label_sample_others, dim=0)
        
        out['loss'] = self.patchcriterion(x_sample_anchors, label_sample_anchors, x_sample_others, label_sample_others)

        return out


        
    def loss_semantic(self, feats, labels, **kwargs):
        N = feats[0].shape[0]
        
        for i in range(int(N / 2)):
            result = self.single_loss_semantic([feats[1][2 * i], feats[1][2 * i + 1]], 
                                               [labels[2 * i], labels[2 * i + 1]])
            if i == 0:
                fine_loss = result['loss']
            else:
                fine_loss += result['loss']
        fine_loss = fine_loss / int(N / 2)
        loss = fine_loss
        return loss
    
    def single_loss_semantic(self, feat, mask):
        out = dict()

        view_1_fine = feat[0]
        view_2_fine = feat[1]
        view_1_fine = view_1_fine.view(view_1_fine.shape[0], -1).unsqueeze(1).permute(2, 1, 0)
        view_2_fine = view_2_fine.view(view_2_fine.shape[0], -1).unsqueeze(1).permute(2, 1, 0)
        
        mask_1 = mask[0].type(torch.half)
        mask_2 = mask[1].type(torch.half)
        mask_1_fine = F.interpolate(mask_1.unsqueeze(0), size=feat[0].shape[1:]).view(-1)
        mask_2_fine = F.interpolate(mask_2.unsqueeze(0), size=feat[1].shape[1:]).view(-1)
        
        fine_feats = torch.cat((view_1_fine, view_2_fine), dim=0)
        fine_labels = torch.cat((mask_1_fine, mask_2_fine), dim=0)
        fine_mean_feat, fine_mean_labels = self.get_mean_vector(fine_labels, fine_feats)
        
        loss = self.supcriterion(fine_feats, fine_labels, fine_mean_feat, fine_mean_labels)
        out['loss'] = loss

        return out
    
    def get_mean_vector(self, mask, features):
        labels = torch.unique(mask).tolist()
        labels_organ = [label for label in labels if label > 0]
        mean_vectors = []
        for label in labels_organ:
            all_ind = torch.where(mask == label)[0]
            mean_vector = features[all_ind, 0, :].mean(dim=0)
            mean_vectors.append(mean_vector)
        mean_vectors = torch.stack(mean_vectors)
        labels_organ = torch.tensor(labels_organ)
        return mean_vectors, labels_organ
    
    
    def meshgrid3d(self, shape, device):
        z_ = torch.linspace(0., shape[0] - 1, shape[0], device=device)
        y_ = torch.linspace(0., shape[1] - 1, shape[1], device=device)
        x_ = torch.linspace(0., shape[2] - 1, shape[2], device=device)
        z, y, x = torch.meshgrid(z_, y_, x_)
        return torch.stack((z, y, x), 3)


    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        x = self.extract_feat(img)
        # outs = []
        out1 = x[0]#.data.cpu().numpy()
        out2 = x[1]#.data.cpu().numpy()
        outs = [out1, out2, img.data,#.cpu().numpy(),
                img_metas[0]['filename'].split('.', 1)[0]]
        output_embedding = self.test_cfg.get('output_embedding', True)
        if not output_embedding:
            if not os.path.exists(self.test_cfg.save_path):
                os.mkdir(self.test_cfg.save_path)
            outfilename = self.test_cfg.save_path + \
                          img_metas[0]['filename'].split('.', 1)[0] + '.pkl'
            f = open(outfilename, 'wb')
            pickle.dump(outs, f)
            return [
                x[0][0, 0, 0, 0, 0].data.cpu()]  # we have saved the data into harddisk, this is just for fit the code
        else:
            return outs

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)
    
        