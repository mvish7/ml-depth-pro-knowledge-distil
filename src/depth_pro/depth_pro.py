# Copyright (C) 2024 Apple Inc. All Rights Reserved.
# Depth Pro: Sharp Monocular Metric Depth in Less Than a Second

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Tuple, Union

import torch
from torch import nn
from torchvision.transforms import (
    Compose,
    ConvertImageDtype,
    Lambda,
    Normalize,
    ToTensor,
)

from .network.decoder import MultiresConvDecoder
from .network.encoder import DepthProEncoder
from .network.fov import FOVNetwork
from .network.vit_factory import VIT_CONFIG_DICT, ViTPreset, create_vit
from src.configs.model_configs import DepthProConfig, DEFAULT_MONODEPTH_CONFIG_DICT, SMALL_MONODEPTH_CONFIG_DICT

# @dataclass
# class DepthProConfig:
#     """Configuration for DepthPro."""
#
#     patch_encoder_preset: ViTPreset
#     image_encoder_preset: ViTPreset
#     decoder_features: int
#
#     checkpoint_uri: Optional[str] = None
#     fov_encoder_preset: Optional[ViTPreset] = None
#     use_fov_head: bool = True
#
#
# DEFAULT_MONODEPTH_CONFIG_DICT = DepthProConfig(
#     patch_encoder_preset="dinov2l16_384",
#     image_encoder_preset="dinov2l16_384",
#     checkpoint_uri="./checkpoints/depth_pro.pt",
#     decoder_features=256,
#     use_fov_head=True,
#     fov_encoder_preset="dinov2l16_384",
# )
#
# SMALL_MONODEPTH_CONFIG_DICT = DepthProConfig(
#     patch_encoder_preset="dinov2s16_384",
#     image_encoder_preset="dinov2s16_384",
#     checkpoint_uri=None,
#     decoder_features=96,
#     use_fov_head=True,
#     fov_encoder_preset="dinov2s16_384",
# )


def create_backbone_model(preset: ViTPreset) -> Tuple[nn.Module, ViTPreset]:
    """Create and load a backbone model given a config.

    Args:
    ----
        preset: A backbone preset to load pre-defind configs.

    Returns:
    -------
        A Torch module and the associated config.

    """
    if preset in VIT_CONFIG_DICT:
        config = VIT_CONFIG_DICT[preset]
        model = create_vit(preset=preset, use_pretrained=False)
    else:
        raise KeyError(f"Preset {preset} not found.")

    return model, config


def create_model_and_transforms(
    config: DepthProConfig = DEFAULT_MONODEPTH_CONFIG_DICT,
    device: torch.device = torch.device("cpu"),
    precision: torch.dtype = torch.float32,
    is_student: bool = False,
) -> Tuple[DepthPro, Compose]:
    """Create a DepthPro model and load weights from `config.checkpoint_uri`.

    Args:
    ----
        config: The configuration for the DPT model architecture.
        device: The optional Torch device to load the model onto, default runs on "cpu".
        precision: The optional precision used for the model, default is FP32.

    Returns:
    -------
        The Torch DepthPro model and associated Transform.

    """
    patch_encoder, patch_encoder_config = create_backbone_model(
        preset=config.patch_encoder_preset)
    image_encoder, _ = create_backbone_model(
        preset=config.image_encoder_preset)

    fov_encoder = None
    if config.use_fov_head and config.fov_encoder_preset is not None:
        fov_encoder, _ = create_backbone_model(
            preset=config.fov_encoder_preset)

    dims_encoder = patch_encoder_config.encoder_feature_dims
    hook_block_ids = patch_encoder_config.encoder_feature_layer_ids
    encoder = DepthProEncoder(
        dims_encoder=dims_encoder,
        patch_encoder=patch_encoder,
        image_encoder=image_encoder,
        hook_block_ids=hook_block_ids,
        decoder_features=config.decoder_features,
        student=is_student,
    )
    decoder = MultiresConvDecoder(
        dims_encoder=[config.decoder_features] + list(encoder.dims_encoder),
        dim_decoder=config.decoder_features,
        student=is_student,
    )

    model = DepthPro(
        encoder=encoder,
        decoder=decoder,
        last_dims=(32, 1),
        use_fov_head=config.use_fov_head,
        fov_encoder=fov_encoder,
        student=is_student,
    ).to(device)

    if precision is not None:
        model = model.to(precision)

    transform = Compose([
        ToTensor(),
        Lambda(lambda x: x.to(device)),
        Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ConvertImageDtype(precision),
    ])

    if config.checkpoint_uri is not None:
        state_dict = torch.load(config.checkpoint_uri, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=state_dict, strict=True)

        if len(unexpected_keys) != 0:
            raise KeyError(
                f"Found unexpected keys when loading monodepth: {unexpected_keys}"
            )

        # fc_norm is only for the classification head,
        # which we would not use. We only use the encoding.
        missing_keys = [key for key in missing_keys if "fc_norm" not in key]
        if len(missing_keys) != 0:
            raise KeyError(
                f"Keys are missing when loading monodepth: {missing_keys}")

    return model, transform


class DepthPro(nn.Module):
    """DepthPro network."""

    def __init__(
        self,
        encoder: DepthProEncoder,
        decoder: MultiresConvDecoder,
        last_dims: tuple[int, int],
        use_fov_head: bool = True,
        fov_encoder: Optional[nn.Module] = None,
        student: bool = False,
    ):
        """Initialize DepthPro.

        Args:
        ----
            encoder: The DepthProEncoder backbone.
            decoder: The MultiresConvDecoder decoder.
            last_dims: The dimension for the last convolution layers.
            use_fov_head: Whether to use the field-of-view head.
            fov_encoder: A separate encoder for the field of view.
            student: Whether the model is a student model.

        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.student = student

        # 1x1 convolutional layers for projecting student head features to match teacher's features
        if student:
            # For head intermediate features: 48 -> 128 -> 48
            self.head_feat_proj = nn.Conv2d(in_channels=48,
                                            out_channels=128,
                                            kernel_size=1)
            self.head_feat_reproj = nn.Conv2d(in_channels=128,
                                              out_channels=48,
                                              kernel_size=1)

        self.head_intermediate_op = None
        self.head_intermediate_projected = None

        dim_decoder = decoder.dim_decoder
        self.head = nn.Sequential(
            nn.Conv2d(dim_decoder,
                      dim_decoder // 2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ConvTranspose2d(
                in_channels=dim_decoder // 2,
                out_channels=dim_decoder // 2,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=True,
            ),
            nn.Conv2d(
                dim_decoder // 2,
                last_dims[0],
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Conv2d(last_dims[0],
                      last_dims[1],
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.ReLU(),
        )

        # Set the final convolution layer's bias to be 0.
        self.head[4].bias.data.fill_(0)

        # Register hook to capture intermediate features
        self.head[1].register_forward_hook(self._head_fwd_hook)

        # Set the FOV estimation head.
        if use_fov_head:
            self.fov = FOVNetwork(num_features=dim_decoder,
                                  fov_encoder=fov_encoder)

    def _head_fwd_hook(self, module, input, output):
        """Hook to capture intermediate features from head."""
        self.head_intermediate_op = output

        # Project intermediate features if student during training
        if self.student and self.training:
            # Project to teacher dimensions
            self.head_intermediate_projected = self.head_feat_proj(output)
            # Reproject back to student dimensions (will replace original output)
            output = self.head_feat_reproj(self.head_intermediate_projected)

        return output

    @property
    def img_size(self) -> int:
        """Return the internal image size of the network."""
        return self.encoder.img_size

    def forward(
            self,
            x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode by projection and fusion of multi-resolution encodings.

        Args:
        ----
            x (torch.Tensor): Input image.

        Returns:
        -------
            The canonical inverse depth map [m] and the optional estimated field of view [deg].

        """
        _, _, H, W = x.shape
        assert H == self.img_size and W == self.img_size

        if self.student:
            encodings, projected_encodings = self.encoder(x)
            (features,
             features_0), projected_features = self.decoder(encodings)
            canonical_inverse_depth = self.head(features)
            head_intermediate = self.head_intermediate_op
            head_intermediate_projected = self.head_intermediate_projected
        else:
            encodings, _ = self.encoder(x)
            (features, features_0), _ = self.decoder(encodings)
            canonical_inverse_depth = self.head(features)
            head_intermediate = self.head_intermediate_op
            head_intermediate_projected = None

        fov_deg = None
        if hasattr(self, "fov"):
            fov_deg, op_feat = self.fov.forward(x, features_0.detach())

        # Return projected features for knowledge distillation if student during training
        if self.student and self.training:
            return {
                'depth': canonical_inverse_depth,
                'fov': fov_deg,
                'projected_features': {
                    'x0': projected_encodings[0],
                    'x1': projected_encodings[1],
                    'x_global': projected_encodings[2],
                    'decoder_features': projected_features[0],
                    'decoder_lowres': projected_features[1],
                    'head_intermediate': head_intermediate_projected
                },
                'intermediate_features': {
                    'head': head_intermediate
                }
            }

        return canonical_inverse_depth, fov_deg

    @torch.no_grad()
    def infer(
        self,
        x: torch.Tensor,
        f_px: Optional[Union[float, torch.Tensor]] = None,
        interpolation_mode="bilinear",
    ) -> Mapping[str, torch.Tensor]:
        """Infer depth and fov for a given image.

        If the image is not at network resolution, it is resized to 1536x1536 and
        the estimated depth is resized to the original image resolution.
        Note: if the focal length is given, the estimated value is ignored and the provided
        focal length is use to generate the metric depth values.

        Args:
        ----
            x (torch.Tensor): Input image
            f_px (torch.Tensor): Optional focal length in pixels corresponding to `x`.
            interpolation_mode (str): Interpolation function for downsampling/upsampling. 

        Returns:
        -------
            Tensor dictionary (torch.Tensor): depth [m], focallength [pixels].

        """
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        _, _, H, W = x.shape
        resize = H != self.img_size or W != self.img_size

        if resize:
            x = nn.functional.interpolate(
                x,
                size=(self.img_size, self.img_size),
                mode=interpolation_mode,
                align_corners=False,
            )

        canonical_inverse_depth, fov_deg = self.forward(x)
        if f_px is None:
            f_px = 0.5 * W / torch.tan(
                0.5 * torch.deg2rad(fov_deg.to(torch.float)))

        inverse_depth = canonical_inverse_depth * (W / f_px)
        f_px = f_px.squeeze()

        if resize:
            inverse_depth = nn.functional.interpolate(inverse_depth,
                                                      size=(H, W),
                                                      mode=interpolation_mode,
                                                      align_corners=False)

        depth = 1.0 / torch.clamp(inverse_depth, min=1e-4, max=1e4)

        return {
            "depth": depth.squeeze(),
            "focallength_px": f_px,
        }

    def get_projection_parameters(self):
        """Get parameters of all projection and reprojection layers for
        knowledge distillation.

        Returns:
            List of parameters from projection layers in encoder, decoder
            and head
        """
        projection_params = []

        # Encoder projection parameters
        if hasattr(self.encoder, 'x0_proj'):
            projection_params.extend([{
                'params':
                self.encoder.x0_proj.parameters()
            }, {
                'params':
                self.encoder.x0_reproj.parameters()
            }, {
                'params':
                self.encoder.x1_proj.parameters()
            }, {
                'params':
                self.encoder.x1_reproj.parameters()
            }, {
                'params':
                self.encoder.x_global_proj.parameters()
            }, {
                'params':
                self.encoder.x_global_reproj.parameters()
            }])

        # Decoder projection parameters
        if hasattr(self.decoder, 'features_proj'):
            projection_params.extend([{
                'params':
                self.decoder.features_proj.parameters()
            }, {
                'params':
                self.decoder.features_reproj.parameters()
            }, {
                'params':
                self.decoder.lowres_proj.parameters()
            }, {
                'params':
                self.decoder.lowres_reproj.parameters()
            }])

        # Head projection parameters
        if hasattr(self, 'head_feat_proj'):
            projection_params.extend([{
                'params':
                self.head_feat_proj.parameters()
            }, {
                'params':
                self.head_feat_reproj.parameters()
            }])

        return projection_params

    def get_non_projection_parameters(self):
        """Get all parameters except projection and reprojection layers.
        
        Returns:
            List of parameters excluding projection layers
        """
        all_params = set(self.parameters())

        # Get all projection parameters
        proj_params = set()

        # Encoder projection parameters
        if hasattr(self.encoder, 'x0_proj'):
            proj_params.update(self.encoder.x0_proj.parameters())
            proj_params.update(self.encoder.x0_reproj.parameters())
            proj_params.update(self.encoder.x1_proj.parameters())
            proj_params.update(self.encoder.x1_reproj.parameters())
            proj_params.update(self.encoder.x_global_proj.parameters())
            proj_params.update(self.encoder.x_global_reproj.parameters())

        # Decoder projection parameters
        if hasattr(self.decoder, 'features_proj'):
            proj_params.update(self.decoder.features_proj.parameters())
            proj_params.update(self.decoder.features_reproj.parameters())
            proj_params.update(self.decoder.lowres_proj.parameters())
            proj_params.update(self.decoder.lowres_reproj.parameters())

        # Head projection parameters
        if hasattr(self, 'head_feat_proj'):
            proj_params.update(self.head_feat_proj.parameters())
            proj_params.update(self.head_feat_reproj.parameters())

        # Return parameters that are not in projection layers
        return [{'params': list(all_params - proj_params)}]
