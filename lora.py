"""
LoRA (Low-Rank Adaptation) implementation for Echo-TTS fine-tuning.

This module provides LoRA adapters that can be applied to the EchoDiT model
for efficient few-shot training while preserving voice cloning capabilities.
"""

import math
import re
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    LoRA adapter that wraps an existing nn.Linear layer.

    Implements low-rank decomposition: W' = W + BA where B is (out, rank) and A is (rank, in).
    The original weights W are frozen, only A and B are trained.
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original = original_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original_layer.in_features
        out_features = original_layer.out_features

        # LoRA matrices - A projects down, B projects up
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize A with Kaiming, B with zeros (standard LoRA init)
        # This ensures the adapter starts as identity (no change to original)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze original weights
        self.original.weight.requires_grad = False
        if self.original.bias is not None:
            self.original.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward pass (frozen)
        result = self.original(x)

        # LoRA delta: x @ A^T @ B^T * scaling
        lora_out = self.dropout(x)
        lora_out = F.linear(lora_out, self.lora_A.to(x.device, x.dtype))  # x @ A^T
        lora_out = F.linear(lora_out, self.lora_B.to(x.device, x.dtype))  # (x @ A^T) @ B^T

        return result + lora_out * self.scaling

    def merge_weights(self) -> nn.Linear:
        """Merge LoRA weights into original layer for efficient inference."""
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None
        )

        # W' = W + scaling * B @ A
        delta = (self.lora_B @ self.lora_A) * self.scaling
        merged.weight.data = self.original.weight.data + delta.to(self.original.weight.dtype)

        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data

        return merged


def _matches_pattern(name: str, patterns: List[str]) -> bool:
    """Check if module name matches any of the glob-style patterns."""
    for pattern in patterns:
        # Convert glob pattern to regex
        regex = pattern.replace(".", r"\.").replace("*", r"[^.]+")
        if re.match(f"^{regex}$", name):
            return True
    return False


def _set_module_by_name(model: nn.Module, name: str, new_module: nn.Module) -> None:
    """Replace a module in the model by its dot-separated name."""
    parts = name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> Tuple[nn.Module, Dict[str, LoRALinear]]:
    """
    Apply LoRA adapters to specified modules in the model.

    Args:
        model: The EchoDiT model to adapt
        rank: LoRA rank (lower = fewer params, less expressive)
        alpha: LoRA scaling factor (typically equal to rank)
        dropout: Dropout probability for regularization
        target_modules: List of module name patterns to target.
                       Uses glob-style patterns with * for wildcards.
                       If None, uses default targets for style adaptation.

    Returns:
        Tuple of (modified model, dict of LoRA modules by name)

    Example:
        model, lora_modules = apply_lora_to_model(
            model,
            rank=16,
            target_modules=[
                "blocks.*.attention.wq",
                "blocks.*.attention.wk",
                "blocks.*.attention.wv",
                "blocks.*.attention.wo",
            ]
        )
    """
    if target_modules is None:
        # Default: target main decoder attention for style adaptation
        # Preserve speaker path (wk_speaker, wv_speaker) for voice cloning
        target_modules = [
            # Main decoder self-attention
            "blocks.*.attention.wq",
            "blocks.*.attention.wk",
            "blocks.*.attention.wv",
            "blocks.*.attention.wo",
            # Text cross-attention (affects text-to-audio mapping)
            "blocks.*.attention.wk_text",
            "blocks.*.attention.wv_text",
            # Latent cross-attention (for controllable rhythm/timing)
            "blocks.*.attention.wk_latent",
            "blocks.*.attention.wv_latent",
            # MLP layers (affects feature transformation)
            "blocks.*.mlp.w1",
            "blocks.*.mlp.w2",
            "blocks.*.mlp.w3",
        ]

    # Freeze all base model parameters first
    for param in model.parameters():
        param.requires_grad = False

    lora_modules: Dict[str, LoRALinear] = {}

    # Find and wrap matching Linear layers
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and _matches_pattern(name, target_modules):
            lora_layer = LoRALinear(
                original_layer=module,
                rank=rank,
                alpha=alpha,
                dropout=dropout,
            )
            _set_module_by_name(model, name, lora_layer)
            lora_modules[name] = lora_layer

    return model, lora_modules


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get all trainable LoRA parameters from the model."""
    params = []
    for module in model.modules():
        if isinstance(module, LoRALinear):
            params.append(module.lora_A)
            params.append(module.lora_B)
    return params


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in the model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def save_lora_checkpoint(
    model: nn.Module,
    path: str,
    config: Optional[Dict] = None,
) -> None:
    """
    Save only the LoRA weights to a checkpoint file.

    Args:
        model: Model with LoRA adapters applied
        path: Path to save the checkpoint
        config: Optional config dict to save alongside weights
    """
    lora_state_dict = {}

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            lora_state_dict[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state_dict[f"{name}.lora_B"] = module.lora_B.data.cpu()

    checkpoint = {
        "lora_state_dict": lora_state_dict,
        "config": config or {},
    }

    torch.save(checkpoint, path)


def load_lora_checkpoint(
    model: nn.Module,
    path: str,
    device: str = "cuda",
) -> Dict:
    """
    Load LoRA weights from a checkpoint into an existing model with LoRA applied.

    Args:
        model: Model with LoRA adapters already applied
        path: Path to the checkpoint file
        device: Device to load weights to

    Returns:
        The config dict from the checkpoint
    """
    checkpoint = torch.load(path, map_location=device)
    lora_state_dict = checkpoint["lora_state_dict"]

    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            a_key = f"{name}.lora_A"
            b_key = f"{name}.lora_B"

            if a_key in lora_state_dict:
                module.lora_A.data = lora_state_dict[a_key].to(device)
            if b_key in lora_state_dict:
                module.lora_B.data = lora_state_dict[b_key].to(device)

    return checkpoint.get("config", {})


def merge_lora_weights(model: nn.Module) -> nn.Module:
    """
    Merge all LoRA weights into the base model for efficient inference.

    This removes the LoRA overhead by baking the adaptations into the original weights.
    The model can no longer be fine-tuned after merging.

    Args:
        model: Model with LoRA adapters applied

    Returns:
        Model with LoRA weights merged into base weights
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, LoRALinear):
            merged = module.merge_weights()
            _set_module_by_name(model, name, merged)

    return model
