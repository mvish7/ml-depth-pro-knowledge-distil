"""
    implements losses for knowledge distillation training
"""
import torch
import torch.nn.functional as F
import torch.nn as nn


def calc_berhu_loss(pred: torch.Tensor, gt: torch.Tensor,
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
    if mask:
        pred = pred[mask]
        gt = gt[mask]

    diff = torch.abs(pred - gt)

    # Calculate median depth for adaptive threshold
    median_depth = torch.median(gt)

    # Adaptive threshold (c)
    c = 0.2 * median_depth

    # BerHu loss calculation
    l1_mask = diff <= c
    l2_mask = diff > c

    l1_loss = diff[l1_mask]
    # making l2 loss match l1 loss when diff == c, avoids discontinuity
    l2_loss = (diff[l2_mask]**2 + c**2) / (2 * c)


    berhu_loss = torch.cat([l1_loss, l2_loss]).mean() if len(torch.cat([l1_loss, l2_loss])) > 0 \
                                                else torch.tensor(0.0, device=pred.device)

    return berhu_loss


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
    # Flatten spatial dimensions
    student_feat = student_feat.view(student_feat.size(0),
                                     student_feat.size(1), -1)  # (B, C, H*W)
    teacher_feat = teacher_feat.view(teacher_feat.size(0),
                                     teacher_feat.size(1), -1)  # (B, C, H*W)
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

    def __init__(self):
        super().__init__()

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

        # mask = gt > 0
        # Apply mask
        depth_gt = depth_gt * mask
        depth_pred = depth_pred * mask

        # Log difference
        log_diff = torch.log(depth_pred + eps) - torch.log(depth_gt + eps)
        log_diff = log_diff[mask > 0]

        # Scale-Invariant Loss
        loss = torch.mean(log_diff**2) - (torch.mean(log_diff)**2)

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
        si_depth = self.scale_invariant_loss(pred, gt)
        # todo: apply scale factors from configs
        return  berhu_depth + si_depth


class FoVSupervision(nn.Module):

    def __init__(self, teacher_channels, student_channels):
        super().__init__()
        self.projection = nn.Conv2d(teacher_channels,
                                    student_channels,
                                    kernel_size=1,
                                    bias=False)

    def forward(self, student_feat, teacher_feat, teacher_focal,
                student_focal):
        # Project teacher features to student feature dimension
        cos_sim_loss = self.calc_cosine_similarity(student_feat, teacher_feat)
        loss_focal = F.l1_loss(student_focal, teacher_focal)

        return loss

    def get_cosine_similarity(self, student_feat: torch.Tensor,
                              teacher_feat: torch.Tensor) -> torch.Tensor:
        teacher_proj = self.projection(teacher_feat)
        cs_loss = calc_cosine_similarity(student_feat, teacher_proj)
        return cs_loss


class FeatureDistillation(nn.Module):

    def __init__(self, ):
        pass

    def apply_full_combination(self, teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
        """
        applies berhu, cosine similarity and gradient loss between teacher and student features.
        Args:
            teacher_feat: activations from designated layer of teacher
            student_feat: activations from designated layer of student
        """
        berhu_kd = calc_berhu_loss(student_feat, teacher_feat, mask=None)
        cs_kd = calc_cosine_similarity(student_feat, teacher_feat)
        grad_kd = calc_spatial_grad_loss(student_feat, teacher_feat)
        # todo: apply loss scaling factors from configs
        return berhu_kd + cs_kd + grad_kd

    def apply_grad_loss(self,teacher_feat: torch.Tensor, student_feat: torch.Tensor) -> torch.Tensor:
        """
        applies gradient loss between teacher and student features.
        Args:
            teacher_feat: activations from designated layer of teacher
            student_feat: activations from designated layer of student
        """
        grad_kd = calc_spatial_grad_loss(student_feat, teacher_feat)

        return grad_kd



# Example Usage:
if __name__ == "__main__":
    # Create dummy depth maps
    batch_size = 4
    height, width = 64, 64
    pred = torch.randn(batch_size, height, width)
    gt = torch.rand(batch_size, height, width) * 10.0  # Simulate depth range

    # Create loss instance
    loss_fn = DepthSupervision()

    # Calculate loss
    loss = loss_fn(pred, gt)
    print("BerHu Loss:", loss.item())

    # Example with zero values in ground truth to test masking.
    gt[0, 10:20, 10:20] = 0
    gt[1, 30:35, 22:45] = 0
    loss = loss_fn(pred, gt)
    print("BerHu Loss with zeros in ground truth:", loss.item())
