"""Frame-shared 3D backbones and the corrected detector/linker contract."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .contracts import EncodedWindow, LinkOutput, NodeBatch
from .node_features import (
    physical_coordinates_um,
    spatial_sinusoidal_embedding,
    temporal_node_features,
)


def _double_conv(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class CustomUNetFeatureExtractor(nn.Module):
    """Notebook-style 3D U-Net modified to emit a feature map."""

    def __init__(self, input_channels: int = 1, feature_dim: int = 32) -> None:
        super().__init__()
        self.encoder1 = _double_conv(input_channels, 24)
        self.encoder2 = _double_conv(24, 48)
        self.encoder3 = _double_conv(48, 96)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = _double_conv(96, 192)
        self.up3 = nn.ConvTranspose3d(192, 96, 2, stride=2)
        self.decoder3 = _double_conv(192, 96)
        self.up2 = nn.ConvTranspose3d(96, 48, 2, stride=2)
        self.decoder2 = _double_conv(96, 48)
        self.up1 = nn.ConvTranspose3d(48, feature_dim, 2, stride=2)
        self.decoder1 = _double_conv(feature_dim + 24, feature_dim)

    @staticmethod
    def _resize_like(tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if tensor.shape[2:] == reference.shape[2:]:
            return tensor
        return F.interpolate(
            tensor,
            size=reference.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        skip1 = self.encoder1(image)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        hidden = self.bottleneck(self.pool(skip3))
        hidden = self._resize_like(self.up3(hidden), skip3)
        hidden = self.decoder3(torch.cat([hidden, skip3], dim=1))
        hidden = self._resize_like(self.up2(hidden), skip2)
        hidden = self.decoder2(torch.cat([hidden, skip2], dim=1))
        hidden = self._resize_like(self.up1(hidden), skip1)
        return self.decoder1(torch.cat([hidden, skip1], dim=1))


class NNUNetFeatureExtractor(nn.Module):
    """nnU-Net-configured PlainConvUNet whose logits are dense features."""

    def __init__(self, input_channels: int, feature_dim: int, config: dict) -> None:
        super().__init__()
        try:
            from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
        except ImportError:
            get_network_from_plans = None

        architecture_kwargs = {
            "n_stages": int(config["n_stages"]),
            "features_per_stage": [int(value) for value in config["features_per_stage"]],
            "conv_op": "torch.nn.modules.conv.Conv3d",
            "kernel_sizes": [[int(v) for v in values] for values in config["kernel_sizes"]],
            "strides": [[int(v) for v in values] for values in config["strides"]],
            "n_conv_per_stage": [int(value) for value in config["n_conv_per_stage"]],
            "n_conv_per_stage_decoder": [
                int(value) for value in config["n_conv_per_stage_decoder"]
            ],
            "conv_bias": True,
            "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
            "norm_op_kwargs": {"eps": 1e-5, "affine": True},
            "dropout_op": None,
            "dropout_op_kwargs": None,
            "nonlin": "torch.nn.LeakyReLU",
            "nonlin_kwargs": {"inplace": True},
        }
        if get_network_from_plans is not None:
            self.network = get_network_from_plans(
                arch_class_name=config["architecture"],
                arch_kwargs=architecture_kwargs,
                arch_kwargs_req_import=["conv_op", "norm_op", "dropout_op", "nonlin"],
                input_channels=input_channels,
                output_channels=feature_dim,
                allow_init=True,
                deep_supervision=False,
            )
        else:
            try:
                from dynamic_network_architectures.architectures.unet import PlainConvUNet
            except ImportError as exc:
                raise RuntimeError(
                    "nnU-Net is not installed; sync with --extra baseline --extra nnunet"
                ) from exc
            self.network = PlainConvUNet(
                input_channels=input_channels,
                num_classes=feature_dim,
                n_stages=architecture_kwargs["n_stages"],
                features_per_stage=architecture_kwargs["features_per_stage"],
                conv_op=nn.Conv3d,
                kernel_sizes=architecture_kwargs["kernel_sizes"],
                strides=architecture_kwargs["strides"],
                n_conv_per_stage=architecture_kwargs["n_conv_per_stage"],
                n_conv_per_stage_decoder=architecture_kwargs["n_conv_per_stage_decoder"],
                conv_bias=True,
                norm_op=nn.InstanceNorm3d,
                norm_op_kwargs=architecture_kwargs["norm_op_kwargs"],
                dropout_op=None,
                dropout_op_kwargs=None,
                nonlin=nn.LeakyReLU,
                nonlin_kwargs=architecture_kwargs["nonlin_kwargs"],
                deep_supervision=False,
            )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.network(image)
        if not isinstance(features, torch.Tensor):
            raise TypeError("nnU-Net backbone must return one full-resolution feature tensor")
        return features


class TemporalAttention(nn.Module):
    """Legacy EXP-0006 per-voxel temporal attention."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, time, channels = features.shape[:3]
        spatial = features.shape[3:]
        locations = math.prod(spatial)
        sequence = (
            features.reshape(batch, time, channels, locations)
            .permute(0, 3, 1, 2)
            .reshape(batch * locations, time, channels)
        )
        normalised = self.norm(sequence)
        attended, _ = self.attention(normalised, normalised, normalised, need_weights=False)
        attended = (
            attended.reshape(batch, locations, time, channels)
            .permute(0, 2, 3, 1)
            .reshape(batch, time, channels, *spatial)
        )
        return features + attended


class TemporalNNUNetFeatureExtractor(NNUNetFeatureExtractor):
    """Legacy PlainConvUNet with temporal attention at selected stages."""

    def __init__(self, input_channels: int, feature_dim: int, config: dict) -> None:
        super().__init__(input_channels, feature_dim, config)
        temporal = config["temporal"]
        heads = int(temporal["heads"])
        features_per_stage = [int(value) for value in config["features_per_stage"]]
        self.temporal_blocks = nn.ModuleDict(
            {
                str(stage): TemporalAttention(features_per_stage[stage], heads)
                for stage in temporal["stages"]
            }
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 6:
            raise ValueError(f"expected (B,T,C,Z,Y,X), got {tuple(image.shape)}")
        batch, time = image.shape[:2]
        hidden = image.reshape(batch * time, *image.shape[2:])
        skips = []
        for stage_index, stage in enumerate(self.network.encoder.stages):
            hidden = stage(hidden)
            key = str(stage_index)
            if key in self.temporal_blocks:
                temporal = hidden.reshape(batch, time, *hidden.shape[1:])
                temporal = self.temporal_blocks[key](temporal)
                hidden = temporal.reshape(batch * time, *temporal.shape[2:])
            skips.append(hidden)
        features = self.network.decoder(skips)
        if not isinstance(features, torch.Tensor):
            raise TypeError("temporal nnU-Net must return one feature tensor")
        return features.reshape(batch, time, *features.shape[1:])


class RelativeTimeTemporalAttention(nn.Module):
    """Per-voxel attention with learned relative time and a ReZero gate."""

    def __init__(self, channels: int, heads: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.relative_time_embedding = nn.Sequential(
            nn.Linear(1, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.residual_gate = nn.Parameter(torch.zeros(channels))

    def forward(
        self,
        features: torch.Tensor,
        relative_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if features.ndim != 6:
            raise ValueError("features must have shape (B,T,C,Z,Y,X)")
        batch, time, channels = features.shape[:3]
        spatial = features.shape[3:]
        locations = math.prod(spatial)
        if relative_time is None:
            relative_time = torch.arange(
                time, device=features.device, dtype=torch.float32
            ).view(1, time).expand(batch, -1)
        else:
            relative_time = torch.as_tensor(
                relative_time, device=features.device, dtype=torch.float32
            )
            if relative_time.ndim == 1 and relative_time.numel() == time:
                relative_time = relative_time.view(1, time).expand(batch, -1)
            elif relative_time.shape != (batch, time):
                raise ValueError("relative_time must have shape (T,) or (B,T)")
        relative_time = relative_time - relative_time[:, :1]
        sequence = (
            features.reshape(batch, time, channels, locations)
            .permute(0, 3, 1, 2)
            .reshape(batch * locations, time, channels)
        )
        time_embedding = self.relative_time_embedding(relative_time.unsqueeze(-1))
        time_embedding = (
            time_embedding.unsqueeze(1)
            .expand(batch, locations, time, channels)
            .reshape(batch * locations, time, channels)
            .to(sequence.dtype)
        )
        attended_input = self.norm(sequence + time_embedding)
        attended, _ = self.attention(
            attended_input, attended_input, attended_input, need_weights=False
        )
        attended = attended * self.residual_gate.to(attended.dtype)
        attended = (
            attended.reshape(batch, locations, time, channels)
            .permute(0, 2, 3, 1)
            .reshape(batch, time, channels, *spatial)
        )
        return features + attended


class CorrectedTemporalNNUNetFeatureExtractor(NNUNetFeatureExtractor):
    """PlainConvUNet with corrected relative-time fusion at selected stages."""

    def __init__(self, input_channels: int, feature_dim: int, config: dict) -> None:
        super().__init__(input_channels, feature_dim, config)
        temporal = config["temporal_fusion"]
        heads = int(temporal["heads"])
        features_per_stage = [int(value) for value in config["features_per_stage"]]
        self.temporal_blocks = nn.ModuleDict(
            {
                str(stage): RelativeTimeTemporalAttention(features_per_stage[stage], heads)
                for stage in temporal["stages"]
            }
        )

    def forward(
        self,
        image: torch.Tensor,
        relative_time: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if image.ndim != 6:
            raise ValueError(f"expected (B,T,C,Z,Y,X), got {tuple(image.shape)}")
        batch, time = image.shape[:2]
        hidden = image.reshape(batch * time, *image.shape[2:])
        skips = []
        for stage_index, stage in enumerate(self.network.encoder.stages):
            hidden = stage(hidden)
            key = str(stage_index)
            if key in self.temporal_blocks:
                temporal = hidden.reshape(batch, time, *hidden.shape[1:])
                temporal = self.temporal_blocks[key](temporal, relative_time)
                hidden = temporal.reshape(batch * time, *temporal.shape[2:])
            skips.append(hidden)
        features = self.network.decoder(skips)
        if not isinstance(features, torch.Tensor):
            raise TypeError("corrected temporal nnU-Net must return one feature tensor")
        return features.reshape(batch, time, *features.shape[1:])


class FrameSharedBackbone(nn.Module):
    """Apply one spatial 3D network with shared weights over time."""

    def __init__(self, spatial_network: nn.Module) -> None:
        super().__init__()
        self.spatial_network = spatial_network

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 6:
            raise ValueError(f"expected (B,T,C,Z,Y,X), got {tuple(image.shape)}")
        batch, time = image.shape[:2]
        features = self.spatial_network(image.reshape(batch * time, *image.shape[2:]))
        return features.reshape(batch, time, *features.shape[1:])


def build_backbone(config: dict) -> nn.Module:
    """Build a legacy or corrected spatial/temporal backbone."""
    name = config["name"]
    feature_dim = int(config["feature_dim"])
    input_channels = int(config.get("input_channels", 1))
    contract = config.get("contract", config.get("model_contract"))
    if contract == "corrected_v2":
        if name != "nnunet":
            raise ValueError("the corrected_v2 contract currently requires name='nnunet'")
        temporal_fusion = config["temporal_fusion"]
        mode = temporal_fusion["mode"]
        if mode == "identity":
            spatial = NNUNetFeatureExtractor(input_channels, feature_dim, config["nnunet"])
            return FrameSharedBackbone(spatial)
        if mode == "per_voxel_mha":
            corrected_config = {
                **config["nnunet"],
                "temporal_fusion": temporal_fusion,
            }
            return CorrectedTemporalNNUNetFeatureExtractor(
                input_channels, feature_dim, corrected_config
            )
        raise ValueError(f"unsupported corrected temporal fusion: {mode}")
    if name == "custom_unet":
        spatial = CustomUNetFeatureExtractor(input_channels, feature_dim)
    elif name == "nnunet":
        spatial = NNUNetFeatureExtractor(input_channels, feature_dim, config["nnunet"])
    elif name == "nnunet_temporal":
        temporal_config = {**config["nnunet"], "temporal": config["temporal"]}
        return TemporalNNUNetFeatureExtractor(input_channels, feature_dim, temporal_config)
    else:
        raise ValueError(f"unsupported backbone: {name}")
    return FrameSharedBackbone(spatial)


def build_joint_model(config: dict, host_training_module) -> nn.Module:
    """Attach the common organizer detection head and node transformer."""
    backbone = build_backbone(config)
    feature_dim = int(config["feature_dim"])
    contract = config.get("contract", config.get("model_contract"))
    if contract == "corrected_v2":
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(config.get("common_head_seed", 0)))
            return CorrectedTrackingModel(
                host_training_module,
                backbone,
                feature_dim,
                spatial_embedding_dim=int(
                    config.get("spatial_embedding_dim", host_training_module._POS_EMBED_DIM)
                ),
                link_candidate_radius_um=float(config["link_candidate_radius_um"]),
                link_candidate_top_k=int(config["link_candidate_top_k"]),
            )
    if config["name"] == "nnunet_temporal":
        return ContextualTemporalNodeModel(
            host_training_module,
            backbone,
            feature_dim,
            4 * host_training_module._POS_EMBED_DIM,
        )
    return host_training_module.UNetNodeTransformer(
        unet=backbone,
        unet_out_channels=feature_dim,
        pos_feat_dim=4 * host_training_module._POS_EMBED_DIM,
    )


class ContextualTemporalNodeModel(nn.Module):
    """Legacy EXP-0006 temporal nnU-Net contextual model."""

    def __init__(
        self,
        host_training_module,
        backbone: nn.Module,
        feature_dim: int,
        position_dim: int,
    ) -> None:
        super().__init__()
        self.unet = backbone
        self.feature_dim = feature_dim
        self.detect_head = nn.Conv3d(feature_dim, 1, kernel_size=1)
        self.division_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.GELU(), nn.Linear(feature_dim, 1)
        )
        self.transformer = host_training_module.SimpleNodeTransformer(
            feat_dim=feature_dim + position_dim + 3,
            hidden_dim=128,
            n_heads=4,
            n_blocks=4,
            dropout=0.3,
        )

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        features = self.unet(images.unsqueeze(2))
        logits = [self.detect_head(features[:, index]) for index in range(features.shape[1])]
        return features, logits

    @staticmethod
    def _index_features(
        feature_maps: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, channels = feature_maps.shape[:2]
        spatial = feature_maps.shape[2:]
        output = torch.zeros(
            batch,
            coords.shape[1],
            channels,
            device=feature_maps.device,
            dtype=feature_maps.dtype,
        )
        for sample in range(batch):
            count = int(mask[sample].sum().item())
            if count == 0:
                continue
            selected = coords[sample, :count].long()
            z = selected[:, 0].clamp(0, spatial[0] - 1)
            y = selected[:, 1].clamp(0, spatial[1] - 1)
            x = selected[:, 2].clamp(0, spatial[2] - 1)
            output[sample, :count] = feature_maps[sample, :, z, y, x].T
        return output

    def predict_edges_contextual(
        self,
        appearance_source: torch.Tensor,
        appearance_target: torch.Tensor,
        detection_logits_source: torch.Tensor,
        detection_logits_target: torch.Tensor,
        coords_source: torch.Tensor,
        coords_target: torch.Tensor,
        coords_source_um: torch.Tensor,
        coords_target_um: torch.Tensor,
        position_source: torch.Tensor,
        position_target: torch.Tensor,
        mask_source: torch.Tensor,
        mask_target: torch.Tensor,
        delta_t: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        confidence_source = self._index_features(
            detection_logits_source.sigmoid(), coords_source, mask_source
        )
        confidence_target = self._index_features(
            detection_logits_target.sigmoid(), coords_target, mask_target
        )
        division_logits_source = self.division_head(appearance_source)
        division_logits_target = self.division_head(appearance_target)
        feature_source = torch.cat(
            [
                appearance_source,
                position_source,
                confidence_source,
                division_logits_source.sigmoid(),
                torch.zeros_like(confidence_source),
            ],
            dim=-1,
        )
        feature_target = torch.cat(
            [
                appearance_target,
                position_target,
                confidence_target,
                division_logits_target.sigmoid(),
                torch.full_like(confidence_target, float(delta_t)),
            ],
            dim=-1,
        )
        edge_logits = self.transformer(
            feature_source,
            feature_target,
            coords_source_um,
            coords_target_um,
            mask_source,
            mask_target,
        )
        return edge_logits, division_logits_source[..., 0]


class CorrectedTrackingModel(nn.Module):
    """Common corrected-v2 detector/linker used by spatial and temporal arms."""

    def __init__(
        self,
        host_training_module,
        backbone: nn.Module,
        feature_dim: int,
        spatial_embedding_dim: int = 8,
        link_candidate_radius_um: float | None = None,
        link_candidate_top_k: int | None = None,
    ) -> None:
        super().__init__()
        if spatial_embedding_dim <= 0 or spatial_embedding_dim % 2:
            raise ValueError("spatial_embedding_dim must be a positive even integer")
        if link_candidate_radius_um is not None and (
            not math.isfinite(link_candidate_radius_um) or link_candidate_radius_um <= 0
        ):
            raise ValueError("link_candidate_radius_um must be finite and positive")
        if link_candidate_top_k is not None and link_candidate_top_k <= 0:
            raise ValueError("link_candidate_top_k must be positive")
        self.unet = backbone
        self.feature_dim = int(feature_dim)
        self.spatial_embedding_dim = int(spatial_embedding_dim)
        self.link_candidate_radius_um = link_candidate_radius_um
        self.link_candidate_top_k = link_candidate_top_k
        self.detect_head = nn.Conv3d(self.feature_dim, 1, kernel_size=1)
        self.division_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
        )
        node_feature_dim = self.feature_dim + 3 * self.spatial_embedding_dim + 4
        self.transformer = host_training_module.SimpleNodeTransformer(
            feat_dim=node_feature_dim,
            hidden_dim=128,
            n_heads=4,
            n_blocks=4,
            dropout=0.3,
        )
        self.null_parent_head = nn.Sequential(
            nn.Linear(node_feature_dim, 128), nn.GELU(), nn.Linear(128, 1)
        )

    def encode_window(self, images: torch.Tensor) -> EncodedWindow:
        if images.ndim == 5:
            window = images.unsqueeze(2)
        elif images.ndim == 6 and images.shape[2] == 1:
            window = images
        else:
            raise ValueError("images must have shape (B,T,Z,Y,X) or (B,T,1,Z,Y,X)")
        features = self.unet(window)
        if features.ndim != 6:
            raise ValueError("backbone must return features with shape (B,T,C,Z,Y,X)")
        batch, time = features.shape[:2]
        flat_features = features.reshape(batch * time, *features.shape[2:])
        detection_logits = self.detect_head(flat_features).reshape(
            batch, time, 1, *features.shape[-3:]
        )
        return EncodedWindow(features=features, detection_logits=detection_logits)

    def encode(self, images: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        encoded = self.encode_window(images)
        return encoded.features, [
            encoded.detection_logits[:, frame]
            for frame in range(encoded.detection_logits.shape[1])
        ]

    @staticmethod
    def _index_features(
        feature_maps: torch.Tensor,
        coords: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if feature_maps.ndim != 5:
            raise ValueError("feature_maps must have shape (B,C,Z,Y,X)")
        if coords.ndim != 3 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape (B,N,3)")
        if mask.shape != coords.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape (B,N)")
        spatial = feature_maps.shape[-3:]
        coordinates = coords.to(torch.float32)
        normalised = []
        for axis, size in zip((2, 1, 0), reversed(spatial), strict=True):
            values = coordinates[..., axis].clamp(0, size - 1)
            values = (
                torch.zeros_like(values)
                if size == 1
                else 2.0 * values / float(size - 1) - 1.0
            )
            normalised.append(values)
        grid = torch.stack(normalised, dim=-1).unsqueeze(2).unsqueeze(2)
        sampled = F.grid_sample(
            feature_maps,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled[:, :, :, 0, 0].transpose(1, 2)
        return sampled * mask.unsqueeze(-1).to(sampled.dtype)

    def build_nodes(
        self,
        encoded: EncodedWindow,
        coords: torch.Tensor,
        masks: torch.Tensor,
        image_shape: Sequence[int] | torch.Tensor,
        voxel_size: Sequence[float] | torch.Tensor,
        frame_indices: Sequence[int] | torch.Tensor | None = None,
        delta_t: float | torch.Tensor = 1.0,
    ) -> tuple[NodeBatch, ...]:
        if coords.ndim != 4 or coords.shape[-1] != 3:
            raise ValueError("coords must have shape (B,T,N,3)")
        if masks.shape != coords.shape[:3] or masks.dtype != torch.bool:
            raise ValueError("masks must be boolean with shape (B,T,N)")
        if coords.shape[:2] != encoded.features.shape[:2]:
            raise ValueError("coords and encoded window must share batch/time axes")
        if coords.device != encoded.features.device or masks.device != encoded.features.device:
            raise ValueError("encoded outputs, coords, and masks must be on the same device")
        batch, frames, nodes = coords.shape[:3]
        spatial_position = spatial_sinusoidal_embedding(
            coords, image_shape, self.spatial_embedding_dim
        ).to(encoded.features.dtype)
        physical_coords = physical_coordinates_um(coords, voxel_size)
        frame_role, elapsed = temporal_node_features(
            batch_size=batch,
            frames=frames,
            nodes=nodes,
            device=coords.device,
            dtype=encoded.features.dtype,
            frame_indices=frame_indices,
            delta_t=delta_t,
        )
        result = []
        for frame in range(frames):
            valid = masks[:, frame]
            appearance = self._index_features(
                encoded.features[:, frame], coords[:, frame], valid
            )
            sampled_detection_logits = self._index_features(
                encoded.detection_logits[:, frame], coords[:, frame], valid
            )
            division_logits = self.division_head(appearance)
            valid_feature = valid.unsqueeze(-1).to(appearance.dtype)
            division_logits = division_logits * valid_feature
            result.append(
                NodeBatch(
                    appearance=appearance,
                    grid_coords=coords[:, frame],
                    physical_coords_um=physical_coords[:, frame],
                    spatial_position=spatial_position[:, frame] * valid_feature,
                    valid_mask=valid,
                    detection_probability=(
                        sampled_detection_logits.sigmoid().detach() * valid_feature
                    ),
                    division_probability=(division_logits.sigmoid().detach() * valid_feature),
                    division_logits=division_logits,
                    frame_role=frame_role[:, frame] * valid_feature,
                    delta_t=elapsed[:, frame] * valid_feature,
                )
            )
        return tuple(result)

    @staticmethod
    def _linker_features(nodes: NodeBatch) -> torch.Tensor:
        return torch.cat(
            [
                nodes.appearance,
                nodes.spatial_position,
                nodes.detection_probability,
                nodes.division_probability,
                nodes.frame_role,
                nodes.delta_t,
            ],
            dim=-1,
        )

    def _link_candidate_mask(self, source: NodeBatch, target: NodeBatch) -> torch.Tensor:
        candidate_mask = source.valid_mask.unsqueeze(-1) & target.valid_mask.unsqueeze(1)
        if candidate_mask.shape[1] == 0 or candidate_mask.shape[2] == 0:
            return candidate_mask
        distances = torch.cdist(
            source.physical_coords_um.to(torch.float32),
            target.physical_coords_um.to(torch.float32),
        )
        if self.link_candidate_radius_um is not None:
            candidate_mask &= distances <= self.link_candidate_radius_um
        if self.link_candidate_top_k is not None:
            count = min(self.link_candidate_top_k, candidate_mask.shape[1])
            ranked_distances = distances.masked_fill(~candidate_mask, torch.inf)
            nearest_indices = ranked_distances.topk(
                count, dim=1, largest=False, sorted=False
            ).indices
            nearest_mask = torch.zeros_like(candidate_mask)
            nearest_mask.scatter_(1, nearest_indices, True)
            candidate_mask &= nearest_mask
        return candidate_mask

    def link_pair(self, source: NodeBatch, target: NodeBatch) -> LinkOutput:
        source_features = self._linker_features(source)
        target_features = self._linker_features(target)
        null_parent_logits = self.null_parent_head(target_features).squeeze(-1)
        candidate_mask = self._link_candidate_mask(source, target)
        if source_features.shape[1] == 0 or target_features.shape[1] == 0:
            edge_logits = source_features.new_empty(
                source_features.shape[0],
                source_features.shape[1],
                target_features.shape[1],
            )
        else:
            safe_source_mask = source.valid_mask.clone()
            safe_target_mask = target.valid_mask.clone()
            safe_source_mask[~safe_source_mask.any(dim=1), 0] = True
            safe_target_mask[~safe_target_mask.any(dim=1), 0] = True
            edge_logits = self.transformer(
                source_features,
                target_features,
                source.physical_coords_um,
                target.physical_coords_um,
                safe_source_mask,
                safe_target_mask,
            )
            edge_logits = edge_logits.masked_fill(~candidate_mask, -1.0e4)
        return LinkOutput(
            edge_logits=edge_logits,
            null_parent_logits=null_parent_logits,
            division_logits=source.division_logits.squeeze(-1),
            candidate_mask=candidate_mask,
        )


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def validate_spatial_divisibility(
    shape: Sequence[int], strides: Sequence[Sequence[int]]
) -> None:
    cumulative = [1, 1, 1]
    for stride in strides[1:]:
        cumulative = [
            current * int(value)
            for current, value in zip(cumulative, stride, strict=True)
        ]
    if any(int(size) < factor for size, factor in zip(shape, cumulative, strict=True)):
        raise ValueError(f"spatial shape {tuple(shape)} is too small for strides {list(strides)}")
