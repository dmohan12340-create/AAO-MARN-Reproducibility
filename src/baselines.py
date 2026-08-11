from __future__ import annotations

import torch.nn as nn
import torchvision.models as tvm


def _replace_classifier(model, attr: str, in_features: int):
    setattr(model, attr, nn.Linear(in_features, 2))
    return model


def build_baseline(name: str, pretrained: bool = True) -> nn.Module:
    name = name.lower()
    if name == "mobilenet_v2":
        weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.mobilenet_v2(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
        return m

    if name == "efficientnet_b0":
        weights = tvm.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.efficientnet_b0(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
        return m

    if name == "convnext_tiny":
        weights = tvm.ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.convnext_tiny(weights=weights)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 2)
        return m

    if name == "vit_b_16":
        weights = tvm.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.vit_b_16(weights=weights)
        m.heads.head = nn.Linear(m.heads.head.in_features, 2)
        return m

    if name in {"swin_t", "swin_transformer"}:
        weights = tvm.Swin_T_Weights.IMAGENET1K_V1 if pretrained else None
        m = tvm.swin_t(weights=weights)
        m.head = nn.Linear(m.head.in_features, 2)
        return m

    # timm covers EfficientViT variants when installed.
    if name.startswith("efficientvit"):
        import timm
        return timm.create_model(name, pretrained=pretrained, num_classes=2)

    raise ValueError(f"Unknown baseline: {name}")
