from dataclasses import dataclass
from typing import Optional, Literal, Dict, List


ViTPreset = Literal[
    "dinov2l16_384",
    "dinov2s16_384",
]


@dataclass
class DepthProConfig:
    """Configuration for DepthPro."""

    patch_encoder_preset: ViTPreset
    image_encoder_preset: ViTPreset
    decoder_features: int

    checkpoint_uri: Optional[str] = None
    fov_encoder_preset: Optional[ViTPreset] = None
    use_fov_head: bool = True


@dataclass
class ViTConfig:
    """Configuration for ViT."""

    in_chans: int
    embed_dim: int

    img_size: int = 384
    patch_size: int = 16

    # In case we need to rescale the backbone when loading from timm.
    timm_preset: Optional[str] = None
    timm_img_size: int = 384
    timm_patch_size: int = 16

    # The following 2 parameters are only used by DPT.  See dpt_factory.py.
    encoder_feature_layer_ids: List[int] = None
    """The layers in the Beit/ViT used to constructs encoder features for DPT."""
    encoder_feature_dims: List[int] = None
    """The dimension of features of encoder layers from Beit/ViT features for DPT."""


VIT_CONFIG_DICT: Dict[ViTPreset, ViTConfig] = {
    "dinov2l16_384": ViTConfig(
        in_chans=3,
        embed_dim=1024,
        encoder_feature_layer_ids=[5, 11, 17, 23],
        encoder_feature_dims=[256, 512, 1024, 1024],
        img_size=384,
        patch_size=16,
        timm_preset="vit_large_patch14_dinov2",
        timm_img_size=518,
        timm_patch_size=14,
    ),
    "dinov2s16_384": ViTConfig(
        in_chans=3,
        embed_dim=384,
        # should be lower than nw depth
        encoder_feature_layer_ids=[3, 5, 8, 11],
        # final dim should match embed_dim of underlying vit
        encoder_feature_dims=[96, 192, 384, 384],
        img_size=384,
        patch_size=16,
        timm_preset="vit_small_patch14_dinov2",
        timm_img_size=518,
        timm_patch_size=14,
    ),
}


DEFAULT_MONODEPTH_CONFIG_DICT = DepthProConfig(
    patch_encoder_preset="dinov2l16_384",
    image_encoder_preset="dinov2l16_384",
    checkpoint_uri="./checkpoints/depth_pro.pt",
    decoder_features=256,
    use_fov_head=True,
    fov_encoder_preset="dinov2l16_384",
)

SMALL_MONODEPTH_CONFIG_DICT = DepthProConfig(
    patch_encoder_preset="dinov2s16_384",
    image_encoder_preset="dinov2s16_384",
    checkpoint_uri=None,
    decoder_features=96,
    use_fov_head=True,
    fov_encoder_preset="dinov2s16_384",
)
