from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import torch
import torch.nn as nn
import torchvision.models as tvm


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def normalize_imagenet(x: torch.Tensor) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(x.device, x.dtype)
    std = IMAGENET_STD.to(x.device, x.dtype)
    return (x - mean) / std


class VGG16SpatialBackbone(nn.Module):
    """VGG16 convolutional trunk returning the 7x7x512 spatial map for 224x224 input."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = tvm.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        base = tvm.vgg16(weights=weights)
        self.features = base.features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(normalize_imagenet(x))


class ChannelSelector(nn.Module):
    def __init__(self, selected_channels: Optional[Sequence[int]] = None):
        super().__init__()
        if selected_channels is None:
            selected_channels = list(range(512))
        idx = torch.as_tensor(sorted(set(map(int, selected_channels))), dtype=torch.long)
        if len(idx) == 0:
            raise ValueError("At least one VGG feature channel must be selected.")
        if idx.min() < 0 or idx.max() >= 512:
            raise ValueError("Selected VGG channels must be in [0, 511].")
        self.register_buffer("indices", idx)

    @property
    def out_dim(self) -> int:
        return int(self.indices.numel())

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        return torch.index_select(fmap, 1, self.indices)


class SpatialTokenEncoder(nn.Module):
    """Converts BxCxHxW to Bx(HW)xC in row-major spatial order."""

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:
        return fmap.flatten(2).transpose(1, 2).contiguous()


class AttentionRNNHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        attention_dim: int = 256,
        attention_heads: int = 4,
        recurrent_hidden: int = 192,
        recurrent_layers: int = 1,
        recurrent_type: str = "rnn",
        bidirectional: bool = True,
        dropout: float = 0.3,
        classifier_hidden: int = 128,
    ):
        super().__init__()
        if attention_dim % attention_heads != 0:
            raise ValueError("attention_dim must be divisible by attention_heads")

        self.proj = nn.Linear(in_dim, attention_dim)
        self.norm1 = nn.LayerNorm(attention_dim)
        self.attn = nn.MultiheadAttention(
            attention_dim,
            attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(attention_dim)

        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}.get(recurrent_type.lower())
        if rnn_cls is None:
            raise ValueError("recurrent_type must be one of rnn/gru/lstm")

        self.rnn = rnn_cls(
            input_size=attention_dim,
            hidden_size=recurrent_hidden,
            num_layers=recurrent_layers,
            dropout=dropout if recurrent_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = recurrent_hidden * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden, 2),
        )

    def forward(
        self, tokens: torch.Tensor, return_attention: bool = False
    ):
        z = self.proj(tokens)
        z = self.norm1(z)
        attn_out, attn_weights = self.attn(
            z, z, z, need_weights=return_attention, average_attn_weights=False
        )
        z = self.norm2(z + attn_out)

        seq_out, hidden = self.rnn(z)
        # Pool all spatial states rather than treating the final raster token as uniquely privileged.
        pooled = seq_out.mean(dim=1)
        logits = self.classifier(pooled)
        if return_attention:
            return logits, attn_weights
        return logits


class AAOMARN(nn.Module):
    def __init__(
        self,
        selected_channels: Optional[Sequence[int]] = None,
        pretrained_vgg16: bool = True,
        attention_dim: int = 256,
        attention_heads: int = 4,
        recurrent_hidden: int = 192,
        recurrent_layers: int = 1,
        recurrent_type: str = "rnn",
        bidirectional: bool = True,
        dropout: float = 0.3,
        classifier_hidden: int = 128,
    ):
        super().__init__()
        self.backbone = VGG16SpatialBackbone(pretrained_vgg16)
        self.selector = ChannelSelector(selected_channels)
        self.tokenizer = SpatialTokenEncoder()
        self.head = AttentionRNNHead(
            in_dim=self.selector.out_dim,
            attention_dim=attention_dim,
            attention_heads=attention_heads,
            recurrent_hidden=recurrent_hidden,
            recurrent_layers=recurrent_layers,
            recurrent_type=recurrent_type,
            bidirectional=bidirectional,
            dropout=dropout,
            classifier_hidden=classifier_hidden,
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        fmap = self.backbone(x)
        fmap = self.selector(fmap)
        tokens = self.tokenizer(fmap)
        return self.head(tokens, return_attention=return_attention)


class VGG16BinaryBaseline(nn.Module):
    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()
        self.backbone = VGG16SpatialBackbone(pretrained)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(512, 2))

    def forward(self, x: torch.Tensor):
        return self.fc(self.pool(self.backbone(x)))


def model_from_config(cfg: dict, selected_channels=None) -> AAOMARN:
    m = cfg["model"]
    return AAOMARN(
        selected_channels=selected_channels,
        pretrained_vgg16=bool(m.get("pretrained_vgg16", True)),
        attention_dim=int(m["attention_dim"]),
        attention_heads=int(m["attention_heads"]),
        recurrent_hidden=int(m["recurrent_hidden"]),
        recurrent_layers=int(m.get("recurrent_layers", 1)),
        recurrent_type=str(m.get("recurrent_type", "rnn")),
        bidirectional=bool(m.get("bidirectional", True)),
        dropout=float(m["dropout"]),
        classifier_hidden=int(m.get("classifier_hidden", 128)),
    )
