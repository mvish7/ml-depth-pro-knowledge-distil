"""
    implements losses for knowledge distillation training
"""
import torch
import torch.nn.functional as F
import torch.nn as nn


import torch
import torch.nn as nn

class OutputSupervision(nn.Module):
    def __init__(self):
        super().__init__()

    def __call__(self):
        pass

    def berhu_loss(self, pred: torch.Tensor, gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        BerHu loss with adaptive threshold based on median depth.

        Args:
            pred: Predicted depth map.
            gt: Ground truth depth map.
            mask: mask indicating valid depth pixels

        Returns:
            torch.Tensor: BerHu loss.
        """
        # mask = (gt > 0)
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

        loss = torch.cat([l1_loss, l2_loss]).mean() if len(torch.cat([l1_loss, l2_loss])) > 0 else torch.tensor(0.0, device=pred.device)

        return loss

    def scale_invariant_loss(self, depth_pred, depth_gt, mask=None, eps=1e-6):
        """
        Scale-Invariant Loss between predicted depth (decoder lower res features) and ground truth.
        Args:
            depth_pred: (B, 1, H, W) predicted depth
            depth_gt: (B, 1, H, W) ground truth depth
            mask: (B, 1, H, W) mask for valid pixels (optional)
            eps: Small value to prevent log instability
        """

        # Downsample GT to match prediction size
        depth_gt_ds = F.interpolate(depth_gt, size=depth_pred.shape[-2:], mode='bilinear', align_corners=True)

        # Apply mask
        depth_gt_ds = depth_gt_ds * mask
        depth_pred = depth_pred * mask

        # Log difference
        log_diff = torch.log(depth_pred + eps) - torch.log(depth_gt_ds + eps)
        log_diff = log_diff[mask > 0]

        # Scale-Invariant Loss
        loss = torch.mean(log_diff ** 2) - (torch.mean(log_diff) ** 2)

        return loss

    def forward(self, pred, gt):
        """
        Forward pass for loss calculation.

        Args:
            pred (torch.Tensor): Predicted depth map.
            gt (torch.Tensor): Ground truth depth map.

        Returns:
            torch.Tensor: Computed loss.
        """
        depth_berhu = self.berhu_loss(pred, gt)
        depth_si = self.scale_invariant_loss(pred, gt)

        return


# Example Usage:
if __name__ == "__main__":
    # Create dummy depth maps
    batch_size = 4
    height, width = 64, 64
    pred = torch.randn(batch_size, height, width)
    gt = torch.rand(batch_size, height, width) * 10.0  # Simulate depth range

    # Create loss instance
    loss_fn = OutputSupervision()

    # Calculate loss
    loss = loss_fn(pred, gt)
    print("BerHu Loss:", loss.item())

    # Example with zero values in ground truth to test masking.
    gt[0,10:20,10:20]= 0
    gt[1,30:35,22:45]= 0
    loss = loss_fn(pred, gt)
    print("BerHu Loss with zeros in ground truth:", loss.item())