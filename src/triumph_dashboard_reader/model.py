"""Inference architecture preserved from dashboard_joint_v1 revision 7a3fd991.

Training losses and training orchestration are intentionally not distributed.
"""

from __future__ import annotations

import importlib.util
import os

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

ARCHITECTURE_ID = (
    "dashboard_v1_mobilenetv3small_stride4_lcd4_rpmroi4_spatial_bbox4_mask_v3"
)
POINT_COUNTS = {"localizer": 8, "speed": 12, "rpm": 8}


def require_colab_device(device="cuda", *, inference_only=False):
    """Resolve an explicit Colab device; never fall back from CUDA to CPU."""
    if device not in ("cuda", "cpu"):
        raise ValueError("device must be 'cuda' or 'cpu'")
    try:
        in_colab = importlib.util.find_spec("google.colab") is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        in_colab = False
    if not (in_colab or os.environ.get("COLAB_RELEASE_TAG")):
        raise RuntimeError(
            "Neural model construction/training/inference is restricted to Google Colab"
        )
    if device == "cpu":
        if not inference_only:
            raise RuntimeError(
                "CPU is permitted only for explicit Colab inference; training requires CUDA"
            )
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A Colab CUDA runtime is required; no CPU/local model fallback"
        )
    return torch.device("cuda")


def require_colab_cuda():
    """Unchanged training policy: a Colab CUDA runtime is required."""
    return require_colab_device("cuda")


def _validate_execution(images, input_hw, device, inference_only, training):
    if (
        images.device.type != device
        or images.ndim != 4
        or tuple(images.shape[1:]) != (3, *input_hw)
    ):
        raise ValueError(
            f"Expected normalized Colab {device.upper()} N,3,H,W inputs at declared shape"
        )
    if device == "cpu" and not inference_only:
        raise RuntimeError("CPU training is not permitted")
    if inference_only and (
        training or torch.is_grad_enabled() or not torch.is_inference_mode_enabled()
    ):
        raise RuntimeError(
            "Inference-only models require eval() and torch.inference_mode()"
        )


def block(inputs, outputs):
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.GroupNorm(8, outputs),
        nn.SiLU(inplace=True),
    )


class InstrumentModel(nn.Module):
    def __init__(
        self, task, input_hw, pretrained=True, *, device="cuda", inference_only=False
    ):
        require_colab_device(device, inference_only=inference_only)
        super().__init__()
        if task not in POINT_COUNTS:
            raise ValueError("Unknown task")
        self.task, self.input_hw = task, tuple(input_hw)
        # Plain policy attributes add no parameters/buffers or checkpoint keys.
        self._execution_device, self._inference_only = device, bool(inference_only)
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        self.features = mobilenet_v3_small(weights=weights).features
        self.top = block(576, 96)
        self.decode16 = block(96 + 48, 64)
        self.decode8 = block(64 + 24, 48)
        self.decode4 = block(48 + 16, 32)
        self.heatmap_head = nn.Conv2d(32, POINT_COUNTS[task], 1)
        self.visibility_head = nn.Linear(576, POINT_COUNTS[task])
        if task == "localizer":
            self.mask_head = nn.Conv2d(32, 1, 1)
            self.dashboard_heatmap_head = nn.Conv2d(32, 4, 1)
            self.presence_head = nn.Linear(576, 1)

    def forward(self, images):
        _validate_execution(
            images,
            self.input_hw,
            self._execution_device,
            self._inference_only,
            self.training,
        )
        value, skips = images, {}
        for index, layer in enumerate(self.features):
            value = layer(value)
            if index in (1, 3, 8):
                skips[index] = value
        pooled = value.mean(dim=(-1, -2))
        value = self.top(value)
        for index, decoder in (
            (8, self.decode16),
            (3, self.decode8),
            (1, self.decode4),
        ):
            value = F.interpolate(
                value,
                size=skips[index].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            value = decoder(torch.cat([value, skips[index]], dim=1))
        result = {
            "heatmap_logits": self.heatmap_head(value),
            "visibility_logits": self.visibility_head(pooled),
        }
        if self.task == "localizer":
            # Retain spatial location: four dashboard bounding-box corner maps
            # share the dense decoder, rather than regressing position from GAP.
            dashboard_logits = self.dashboard_heatmap_head(value)
            h, w = dashboard_logits.shape[-2:]
            yy, xx = torch.meshgrid(
                torch.arange(h, device=images.device),
                torch.arange(w, device=images.device),
                indexing="ij",
            )
            probability = dashboard_logits.float().flatten(2).softmax(-1)
            # Same pixel-center transform as landmark heatmaps, then normalize
            # by input width/height to match the exported box target convention.
            xs = ((xx.float() + 0.5) * self.input_hw[1] / w - 0.5) / self.input_hw[1]
            ys = ((yy.float() + 0.5) * self.input_hw[0] / h - 0.5) / self.input_hw[0]
            corners = torch.stack(
                [
                    (probability * xs.flatten()).sum(-1),
                    (probability * ys.flatten()).sum(-1),
                ],
                dim=-1,
            )
            result.update(
                lcd_logits=F.interpolate(
                    self.mask_head(value),
                    size=self.input_hw,
                    mode="bilinear",
                    align_corners=False,
                ),
                dashboard_heatmap_logits=dashboard_logits,
                dashboard_box=torch.cat(
                    [corners.amin(dim=1), corners.amax(dim=1)], dim=1
                ),
                presence_logits=self.presence_head(pooled).squeeze(1),
            )
        return result
