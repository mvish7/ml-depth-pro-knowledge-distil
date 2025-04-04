# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Field of View network architecture.

from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F


class FOVNetwork(nn.Module):
    """Field of View estimation network."""

    def __init__(
            self,
            num_features: int,
            fov_encoder: Optional[nn.Module] = None,
            student: bool = False,  # Add student flag
    ):
        """Initialize the Field of View estimation block.

        Args:
        ----
            num_features: Number of features used.
            fov_encoder: Optional encoder to bring additional network capacity.
            student: Whether this is a student model.

        """
        super().__init__()

        # Create FOV head.
        fov_head0 = [
            nn.Conv2d(num_features,
                      num_features // 2,
                      kernel_size=3,
                      stride=2,
                      padding=1),  # 128 x 24 x 24
            nn.ReLU(True),
        ]
        fov_head = [
            nn.Conv2d(num_features // 2,
                      num_features // 4,
                      kernel_size=3,
                      stride=2,
                      padding=1),  # 64 x 12 x 12
            nn.ReLU(True),
            nn.Conv2d(num_features // 4,
                      num_features // 8,
                      kernel_size=3,
                      stride=2,
                      padding=1),  # 32 x 6 x 6
            nn.ReLU(True),
            nn.Conv2d(num_features // 8, 1, kernel_size=6, stride=1,
                      padding=0),
        ]
        if fov_encoder is not None:
            self.encoder = nn.Sequential(
                fov_encoder, nn.Linear(fov_encoder.embed_dim,
                                       num_features // 2))
            self.downsample = nn.Sequential(*fov_head0)
        else:
            fov_head = fov_head0 + fov_head
        self.head = nn.Sequential(*fov_head)
        # self.penultimate_output = None
        # self.head[2].register_forward_hook(self._forward_hook)

        # 1x1 convolutional layers for projecting student features to match teacher's features
        # if student:
        #     # For penultimate_output: 12 -> 32 -> 12
        #     self.penultimate_proj = nn.Conv2d(in_channels=12,
        #                                       out_channels=32,
        #                                       kernel_size=1)
        #     self.penultimate_reproj = nn.Conv2d(in_channels=32,
        #                                         out_channels=12,
        #                                         kernel_size=1)

        self.student = student

    # def _forward_hook(self, module, input, output):
    #     self.penultimate_output = output
    #
    #     # Project penultimate features if student during training
    #     if self.student and self.training:
    #         # Project to teacher dimensions
    #         self.penultimate_projected = self.penultimate_proj(output)
    #         # Reproject back to student dimensions (will replace original output)
    #         output = self.penultimate_reproj(self.penultimate_projected)
    #
    #     return output

    def forward(self, x: torch.Tensor,
                lowres_feature: torch.Tensor) -> torch.Tensor:
        """Forward the fov network.

        Args:
        ----
            x (torch.Tensor): Input image.
            lowres_feature (torch.Tensor): Low resolution feature.

        Returns:
        -------
            The field of view tensor.

        """
        if hasattr(self, "encoder"):
            x = F.interpolate(
                x,
                size=None,
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
            )
            x = self.encoder(x)[:, 1:].permute(0, 2, 1)
            lowres_feature = self.downsample(lowres_feature)
            x = x.reshape_as(lowres_feature) + lowres_feature
        else:
            x = lowres_feature
        x = self.head(x)
        return x