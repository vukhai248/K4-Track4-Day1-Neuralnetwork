"""optimizer.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Được dùng torch.optim.* và torch.nn.utils.clip_grad_norm_ (xem README mục 5).
File này gom việc chọn bộ tối ưu và cắt gradient để `train.py` gọn và mọi thí nghiệm công bằng.

Công thức cần hiểu (slide Chương 4):
    SGD            : w <- w - lr * g
    SGD + momentum : v <- mu * v + g ;  w <- w - lr * v          (dạng PyTorch)
    Adam           : m <- b1 m + (1-b1) g ; v <- b2 v + (1-b2) g^2 ; w <- w - lr * m_hat / (sqrt(v_hat) + eps)
    AdamW          : như Adam nhưng suy giảm trọng số tách riêng: w <- w - lr * wd * w - lr * m_hat / (sqrt(v_hat) + eps)
"""
from __future__ import annotations

import torch

OPTIMIZERS = ("sgd", "sgd_momentum", "adam", "adamw")


def build_optimizer(name: str, params, lr: float, weight_decay: float = 0.0,
                    momentum: float = 0.9, betas=(0.9, 0.999), eps: float = 1e-8):
    """Trả về một torch.optim.Optimizer."""
    name_clean = name.lower()
    if name_clean not in OPTIMIZERS:
        raise ValueError(f"Bộ tối ưu '{name}' không hợp lệ. Chọn từ: {OPTIMIZERS}")

    if name_clean == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    elif name_clean == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif name_clean == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif name_clean == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)


def build_scheduler(optimizer, name: str | None, total_steps: int, **kwargs):
    """(Tuỳ chọn) Bộ lập lịch tốc độ học, ví dụ cosine hoặc linear."""
    if name is None:
        return None

    s_name = name.lower()
    if s_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=kwargs.get("eta_min", 0.0)
        )
    elif s_name == "linear":
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=kwargs.get("end_factor", 0.01),
            total_iters=total_steps,
        )
    elif s_name == "step":
        step_size = kwargs.get("step_size", max(1, total_steps // 3))
        gamma = kwargs.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        raise ValueError(f"Không hỗ trợ scheduler: '{name}'")


def clip_gradients(params, max_norm: float | None) -> float:
    """Cắt gradient theo chuẩn L2 toàn cục, và TRẢ VỀ chuẩn gradient TRƯỚC KHI cắt."""
    if max_norm is None:
        norm = torch.nn.utils.clip_grad_norm_(params, float("inf"))
    else:
        norm = torch.nn.utils.clip_grad_norm_(params, float(max_norm))

    return float(norm.item() if isinstance(norm, torch.Tensor) else norm)
