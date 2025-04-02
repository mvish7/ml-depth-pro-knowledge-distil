"""
    implements losses for knowledge distillation training
"""
from typing import Dict

import torch
import torch.nn.functional as F
import torch.nn as nn


def calc_berhu_loss(pred: torch.Tensor,
                    gt: torch.Tensor,
                    mask: torch.Tensor = None) -> torch.Tensor:
    """
    BerHu loss with adaptive threshold based on median depth.

    Args:
        pred: Predicted depth map.
        gt: Ground truth depth map.
        mask: mask indicating valid depth pixels

    Returns:
        torch.Tensor: BerHu loss.
    """
    if mask is not None:
        pred = pred[mask]
        gt = gt[mask]

    diff = torch.abs(pred - gt)

    C = 0.2 * torch.max(diff)

    return torch.mean(torch.where(diff < C, diff, (diff * diff + C * C) / (2 * C)))


def calc_cosine_similarity(student_feat: torch.Tensor,
                           teacher_feat: torch.Tensor) -> torch.Tensor:
    """
    calculates cosine similarity loss
    Args:
        student_feat: feature map from a layer of student n/w
        teacher_feat: feature map from a layer of teacher n/w projected to match the student

    Returns:
        cosine simi loss
    """

    # Normalize features along channel dimension
    student_feat = F.normalize(student_feat, p=2, dim=1)
    teacher_feat = F.normalize(teacher_feat, p=2, dim=1)
    # Flatten spatial dimensions -- discarding flattening to preserve the spatial features. i.e. with flattening cosine
    # similarity focuses of matching features at each spatial location and thus spatial features gets destroyed.

    #student_feat = student_feat.view(student_feat.size(0),
    #                                 student_feat.size(1), -1)  # (B, C, H*W)
    #teacher_feat = teacher_feat.view(teacher_feat.size(0),
    #                                 teacher_feat.size(1), -1)  # (B, C, H*W)

    # Compute Cosine Similarity Loss
    cosine_sim = torch.sum(student_feat * teacher_feat, dim=1)  # (B, H*W)
    cosine_sim_loss = 1.0 - cosine_sim.mean()
    return cosine_sim_loss


def calc_spatial_grad_loss(student_feat: torch.Tensor,
                           teacher_feat: torch.Tensor) -> torch.Tensor:
    """
    loss to align feature gradients in teacher and student feature maps. helps with spatial feature matching
    Args:
        student_feat: feature map from a layer of student n/w
        teacher_feat: feature map from a layer of teacher n/w projected to match the student

    Returns:
        spatial gradient loss
    """

    # Spatial gradient loss
    s_grad_x = student_feat[:, :, 1:, :] - student_feat[:, :, :-1, :]
    t_grad_x = teacher_feat[:, :, 1:, :] - teacher_feat[:, :, :-1, :]
    s_grad_y = student_feat[:, :, :, 1:] - student_feat[:, :, :, :-1]
    t_grad_y = teacher_feat[:, :, :, 1:] - teacher_feat[:, :, :, :-1]

    grad_loss = F.mse_loss(s_grad_x, t_grad_x) + F.mse_loss(s_grad_y, t_grad_y)

    return grad_loss


class DepthSupervision(nn.Module):

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    @staticmethod
    def scale_invariant_loss(depth_pred: torch.Tensor,
                             depth_gt: torch.Tensor,
                             mask: torch.Tensor = None,
                             eps=1e-6) -> torch.Tensor:
        """
        Scale-Invariant Loss between predicted depth and ground truth.
        Args:
            depth_pred: (B, 1, H, W) predicted depth
            depth_gt: (B, 1, H, W) ground truth depth
            mask: (B, 1, H, W) mask for valid pixels (optional)
            eps: Small value to prevent log instability
        """
        # Apply mask if provided
        if mask is not None:
            valid_pixels = mask > 0
            if not torch.any(valid_pixels):
                return torch.tensor(0.0, device=depth_pred.device)
            depth_gt = depth_gt[valid_pixels]
            depth_pred = depth_pred[valid_pixels]

        valid_depths = (depth_pred > eps) & (depth_gt > eps)

        if not torch.any(valid_depths):
            return torch.tensor(0.0, device=depth_pred.device)

        # Only compute loss on valid pixels
        depth_pred = depth_pred[valid_depths]
        depth_gt = depth_gt[valid_depths]

        # Log difference (now safe since we've filtered to positive values)
        log_diff = torch.log(depth_pred) - torch.log(depth_gt)

        # Scale-Invariant Loss
        loss = torch.mean(log_diff ** 2) - 0.5 * (torch.mean(log_diff) ** 2)


        return loss

    def forward(self, pred: torch.Tensor, gt: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for loss calculation.

        Args:
            pred: Predicted depth map.
            gt: Ground truth depth map.
            mask: allows masking of invalid depth values from loss calculation

        Returns:
            torch.Tensor: Computed loss.
        """
        berhu_depth = calc_berhu_loss(pred, gt, mask)
        si_depth = self.scale_invariant_loss(pred, gt, mask)
        # todo: apply scale factors from configs
        total_depth_loss = self.config["berhu_scaling"] * berhu_depth + self.config["si_scaling"] * si_depth
        return total_depth_loss


class FoVSupervision(nn.Module):

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    def forward(self,  teacher_focal, student_focal):
        # movign cos_sim_loss out as it is a KD loss
        # cos_sim_loss = calc_cosine_similarity(student_feat, teacher_feat)
        loss_focal = F.l1_loss(student_focal, teacher_focal)
        return self.config["fov_l1_scaling"] * loss_focal


class FeatureDistillation(nn.Module):

    def __init__(self, config: Dict):
        super().__init__()
        self.config = config

    def forward(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor,
                full_combination: bool = True, grad_loss_only: bool = False, cs_loss_only: bool = False) -> torch.Tensor:
        if full_combination:
            return self.apply_full_combination(student_feat, teacher_feat)
        elif grad_loss_only:
            return self.apply_grad_loss(student_feat, teacher_feat)
        elif cs_loss_only:
            return self.apply_cosine_simi_loss(student_feat, teacher_feat)

    def apply_full_combination(self, student_feat: torch.Tensor,
                               teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        applies berhu, cosine similarity and gradient loss between teacher and student features.
        Args:
            teacher_feat: activations from designated layer of teacher
            student_feat: activations from designated layer of student
        """
        berhu_kd = calc_berhu_loss(student_feat, teacher_feat, mask=None)
        cs_kd = calc_cosine_similarity(student_feat, teacher_feat)
        grad_kd = calc_spatial_grad_loss(student_feat, teacher_feat)
        return (self.config["berhu_kd_scaling"] * berhu_kd + self.config["cs_kd_scaling"] * cs_kd +
         self.config["grad_kd_scaling"] * grad_kd)

    def apply_grad_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        applies gradient loss between teacher and student features.
        Args:
            teacher_feat: activations from designated layer of teacher
            student_feat: activations from designated layer of student
        """
        grad_kd = calc_spatial_grad_loss(student_feat, teacher_feat)

        return self.config["grad_kd_scaling"] * grad_kd

    def apply_cosine_simi_loss(self, student_feat: torch.Tensor, teacher_feat: torch.Tensor) -> torch.Tensor:
        """
        applies cosine similarity loss between teacher and student features.
        Args:
            teacher_feat: activations from designated layer of teacher
            student_feat: activations from designated layer of student
        """
        cs_kd = calc_cosine_similarity(student_feat, teacher_feat)

        return self.config["cs_kd_scaling"] * cs_kd



# Example Usage:
if __name__ == "__main__":
    # Create dummy depth maps
    batch_size = 4
    height, width = 64, 64
    pred = torch.rand(batch_size, 3, height, width)
    gt = torch.rand(batch_size, 3, height, width) * 10.0
    mask = torch.ones_like(gt, dtype=torch.bool)

    # Create loss instance
    # loss_fn = DepthSupervision()
    # loss_fn = FoVSupervision()
    # loss_fn(pred, gt, torch.tensor([25.3]), torch.tensor([38.6]))

    loss_fn = FeatureDistillation()
    loss_fn(gt, pred)

    # Calculate loss
    loss = loss_fn(pred, gt, mask)
    print("BerHu Loss:", loss.item())

    # Example with zero values in ground truth to test masking.
    gt[0, 10:20, 10:20] = 0
    gt[1, 30:35, 22:45] = 0
    loss = loss_fn(pred, gt)
    print("BerHu Loss with zeros in ground truth:", loss.item())
