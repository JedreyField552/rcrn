"""Mixed-precision context.

Ref: Methods, "Training and inference" (the manuscript does not print a precision
setting; the value is an engineering default exposed in configuration).
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
from torch import nn

PRECISIONS: tuple[str, ...] = ("fp32", "fp16", "bf16")


def autocast_context(precision: str, device: torch.device) -> Any:
    """Return the autocast context for a precision name, or a no-op for fp32."""

    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision}")
    if precision == "fp32" or device.type != "cuda":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


def build_grad_scaler(precision: str, enabled: bool = True) -> Any:
    """Gradient scaler for fp16, disabled for fp32 and bf16 and without an accelerator."""

    if precision == "fp16":
        active = enabled and torch.cuda.is_available()
        return torch.amp.GradScaler("cuda", enabled=active)
    return None


def support_matrix() -> dict[str, str]:
    """Which precisions keep parameters in full precision."""

    return {"fp32": "parameters_fp32", "bf16": "parameters_fp32", "fp16": "loss_scaled"}


def parameters_dtype(model: nn.Module) -> str:
    """The dtype of the first parameter, for a checkpoint record."""

    for parameter in model.parameters():
        return str(parameter.dtype)
    return "none"
