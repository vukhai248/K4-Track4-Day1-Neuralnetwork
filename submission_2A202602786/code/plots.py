"""plots.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Ảnh biểu đồ là sản phẩm nộp (xem README mục 6): mỗi thí nghiệm một ảnh figures/<exp_id>.png.
Khi notebook chạy trong code/, lưu vào "../figures/" (ví dụ path = f"../figures/{exp_id}.png").
"""
from __future__ import annotations

import matplotlib.pyplot as plt


from pathlib import Path


def plot_run(result: dict, path: str) -> None:
    """Vẽ MỘT thí nghiệm thành một ảnh PNG có ít nhất 3 ô:
         (1) train_loss và val_loss theo epoch (cùng một trục)
         (2) val_acc và val_macro_f1 theo epoch
         (3) grad_norm theo epoch (đo TRƯỚC khi clip)
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    history = result["history"]
    cfg = result["cfg"]
    summary = result.get("summary", {})

    epochs = history.get("epoch", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    val_acc = history.get("val_acc", [])
    val_f1 = history.get("val_macro_f1", [])
    grad_norm = history.get("grad_norm", [])
    best_epoch = summary.get("best_epoch", None)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Ô 1: Train & Val Loss
    axes[0].plot(epochs, train_loss, label="Train Loss", color="royalblue", lw=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", color="crimson", lw=2)
    if best_epoch is not None:
        axes[0].axvline(best_epoch, color="gray", linestyle="--", alpha=0.7, label=f"Best Ep {best_epoch}")
    axes[0].set_title("Train & Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Ô 2: Accuracy & Macro-F1
    axes[1].plot(epochs, val_acc, label="Val Acc", color="forestgreen", lw=2)
    if val_f1:
        axes[1].plot(epochs, val_f1, label="Val Macro-F1", color="darkorange", lw=2)
    if best_epoch is not None:
        axes[1].axvline(best_epoch, color="gray", linestyle="--", alpha=0.7)
    best_f1_val = summary.get("val_macro_f1", 0.0)
    axes[1].set_title(f"Accuracy & Macro-F1 (Best F1: {best_f1_val:.4f})")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Ô 3: Gradient Norm trước khi clip
    if grad_norm:
        axes[2].plot(epochs, grad_norm, label="Grad Norm (pre-clip)", color="purple", lw=1.8)
    axes[2].set_title("Grad Norm (pre-clip)")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Norm L2")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    cfg_str = f"opt={cfg.get('optimizer')}, lr={cfg.get('lr')}, batch={cfg.get('batch')}, init={cfg.get('init')}, drop={cfg.get('dropout')}, clip={cfg.get('clip_norm')}"
    fig.suptitle(f"[{cfg.get('exp_id')}] {cfg.get('description', '')} ({cfg_str})", fontsize=11, fontweight="bold")

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(results: list[dict], metric: str, path: str, title: str = "") -> None:
    """Vẽ chồng một chỉ số của nhiều thí nghiệm trên cùng một trục."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.2))

    for res in results:
        history = res.get("history", {})
        exp_id = res.get("cfg", {}).get("exp_id", "exp")
        epochs = history.get("epoch", [])
        vals = history.get(metric, [])
        if vals:
            ax.plot(epochs, vals, marker="o", markersize=3, label=exp_id, lw=1.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric)
    if not title:
        title = f"So sánh {metric} giữa các thí nghiệm"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
