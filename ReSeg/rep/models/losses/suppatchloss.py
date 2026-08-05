import torch
import torch.nn.functional as F
from torch import nn
from mmdet.models.builder import LOSSES


# -------------------------------
# Loss class for a single-layer
# -------------------------------
@LOSSES.register_module()
class SupPatchNCELoss(nn.Module):
    """
    Computes the supervised patch-based contrastive loss for
    representation learning.

    This loss encourages features from the same spatial class (as defined by
    segmentation labels) to be close in the embedding space, while features
    from different classes are pushed apart. It is designed to work with
    patch-level features extracted from 2D or 3D data, and supports both 2D
    and 3D segmentation tasks.

    This implementation is yoinked and modified from:
    https://github.com/HobbitLong/SupContrast/blob/master/losses.py

    Parameters
    ----------
    opt : argparse.Namespace
        Options containing hyperparameters for the loss, including:
            - nce_T: Temperature parameter for contrastive loss.

    Attributes
    ----------
    opt : argparse.Namespace
        The options object.
    mask_dtype : torch.dtype
        Data type for masks (torch.bool).
    temperature : float
        Temperature scaling for contrastive logits.
    _cosine_similarity : torch.nn.CosineSimilarity
        Cosine similarity function along the last dimension.
    """

    def __init__(self, temperature):

        super().__init__()
        self.mask_dtype = torch.bool

        self.temperature = temperature
        self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)

    def _compute_similarity(self, x, y):
        assert x.size()[-1] == y.size()[-1], f"wrong shape {x.size()} and {y.size()}"
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        # x shape: (N, 1, C)
        # y shape: (1, M, C)
        # v shape: (N, M)
        return v

    def forward(
        self, features, labels_seg, other_features, other_labels_seg, debug=False
    ):
        """
        Compute the supervised patch-based contrastive loss.

        Parameters
        ----------
        features : torch.Tensor
            Feature tensor of shape (num_patches, mlp_op_dim), where
            num_patches is the number of sampled patches, and mlp_op_dim is the feature dimension.
        labels_seg : torch.Tensor
            Segmentation label tensor of shape (num_patches).
        other_features : torch.Tensor
            Feature tensor of shape (num_patches, mlp_op_dim), where
            num_patches is the number of sampled patches, and mlp_op_dim is the feature dimension.
        other_labels_seg : torch.Tensor
            Segmentation label tensor of shape (num_patches).
        debug : bool, optional
            If True, enables debug mode (default: False).

        Returns
        -------
        loss : torch.Tensor
            The computed supervised contrastive loss (scalar).
        """

        device = features.device
        num_patches, nc = features.size()        
        
        # finds matching segs
        mask = torch.eq(labels_seg.unsqueeze(1), labels_seg.unsqueeze(0)).float().to(device)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(num_patches).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        overlap_mask = torch.eq(labels_seg.unsqueeze(1), other_labels_seg.unsqueeze(0)).float().to(device)
        
        mask = torch.cat([mask, overlap_mask], dim=1)
        logits_mask = torch.cat([logits_mask, torch.ones_like(overlap_mask)], dim=1)
        
        # this corresponds to self.contrast_mode == 'all' in
        # https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L62
        anchor_feature = features

        contrast_feature = torch.cat([features, other_features], dim=0)

        # compute logits
        anchor_dot_contrast = torch.div(
            self._compute_similarity(anchor_feature, contrast_feature),
            self.temperature,
        )

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        # TODO: replace the 2nd term with torch.logsumexp
        # (no need for exp_logits then but replace with logits*logits_mask)
        # Random note: 2nd term is the denominator. whether to remove the positives
        # from the denominator is an open issue IMO
        # https://github.com/HobbitLong/SupContrast/issues/64#issuecomment-1182845137
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positives
        # TODO: fix an edge case
        # https://github.com/HobbitLong/SupContrast/blob/master/losses.py#L92
        uses = mask.sum(1) > 0
        mask = mask[uses, :]
        log_prob = log_prob[uses, :]
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = -mean_log_prob_pos
        loss = loss.view(1, -1).mean()

        return loss