"""train.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Gồm: đặt seed, đánh giá, vòng huấn luyện `run_experiment(cfg, data)`, dự đoán và ghi file nộp.
Mọi thí nghiệm chỉ là *đổi dict cfg* rồi gọi lại run_experiment (xem GUIDE, Part 2).

Mọi chỉ số (loss, accuracy, macro-F1) dùng cùng định nghĩa với scripts/evaluate.py.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, EXPECTED_PARAMS, count_params
from optimizer import build_optimizer, clip_gradients

# Cấu hình mặc định = BASELINE (M-base). `lr` do bạn tự chọn bằng val rồi điền vào.
DEFAULT_CFG = dict(
    exp_id="base-s1", group="baseline", description="Baseline M-base",
    loss="ce",                 # "ce" | "mse"
    optimizer="sgd_momentum",  # "sgd" | "sgd_momentum" | "adam" | "adamw"
    lr=None,                   # TODO: chọn bằng val, không dùng eval
    weight_decay=0.0, momentum=0.9,
    batch=512, epochs=20,
    hidden=(256, 128), dropout=0.0, init="he",
    clip_norm=None,            # None = không clip; hoặc số, ví dụ 1.0
    precision="fp32",          # "fp32" | "fp16" | "bf16"
    seed=1,
)


import random
from pathlib import Path
import pandas as pd


def set_seed(seed: int) -> None:
    """Đặt seed cho random, numpy, torch (và torch.cuda nếu có)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    """macro-F1 = trung bình cộng F1 của 7 lớp; F1_c = 2PR/(P+R), bằng 0 nếu P+R = 0."""
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    prec = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
    rec = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp), where=(prec + rec) > 0)
    return float(f1.mean())


@torch.no_grad()
def predict(model, X, batch_size: int = 8192) -> torch.Tensor:
    """Trả về nhãn dự đoán int64 (N,) = argmax của logits."""
    was_training = model.training
    model.eval()
    preds_list = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        logits = model(xb)
        preds = torch.argmax(logits, dim=1)
        preds_list.append(preds)
    if was_training:
        model.train()
    return torch.cat(preds_list, dim=0) if preds_list else torch.empty(0, dtype=torch.int64, device=X.device)


@torch.no_grad()
def evaluate(model, X, y, loss_name: str = "ce", batch_size: int = 8192) -> dict:
    """Trả về dict(loss, acc, macro_f1) ở chế độ eval() (dropout tắt) và no_grad."""
    was_training = model.training
    model.eval()
    total_loss = 0.0
    n_samples = len(X)
    all_preds = []
    all_y = []

    for i in range(0, n_samples, batch_size):
        xb = X[i:i + batch_size]
        yb = y[i:i + batch_size]
        logits = model(xb)
        if loss_name.lower() == "ce":
            loss = F.cross_entropy(logits, yb, reduction="sum")
        else:
            y_oh = F.one_hot(yb, num_classes=logits.size(-1)).float()
            loss = F.mse_loss(logits, y_oh, reduction="sum")

        total_loss += float(loss.item())
        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu())
        all_y.append(yb.cpu())

    if was_training:
        model.train()

    avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
    cat_preds = torch.cat(all_preds).numpy() if all_preds else np.array([])
    cat_y = torch.cat(all_y).numpy() if all_y else np.array([])
    acc = float((cat_preds == cat_y).mean()) if len(cat_y) > 0 else 0.0

    cm = np.zeros((7, 7), dtype=np.int64)
    np.add.at(cm, (cat_y, cat_preds), 1)
    macro_f1 = macro_f1_from_confusion(cm)

    return {"loss": avg_loss, "acc": acc, "macro_f1": macro_f1}


def compute_loss(logits, y, loss_name: str):
    """"ce" : cross-entropy; "mse": MSE với nhãn one-hot."""
    if loss_name.lower() == "ce":
        return F.cross_entropy(logits, y)
    elif loss_name.lower() == "mse":
        y_one_hot = F.one_hot(y, num_classes=logits.size(-1)).float()
        return F.mse_loss(logits, y_one_hot)
    else:
        raise ValueError(f"Không hỗ trợ loss: '{loss_name}'. Chọn 'ce' hoặc 'mse'")


def run_experiment(cfg: dict, data: dict) -> dict:
    """Huấn luyện một cấu hình và trả về lịch sử + tóm tắt."""
    seed = cfg.get("seed", 1)
    set_seed(seed)

    device = data["X_tr"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    dropout = float(cfg.get("dropout", 0.0))
    init_name = cfg.get("init", "he")
    model = MLP(hidden=hidden, dropout=dropout, init=init_name, in_features=54, num_classes=7).to(device)

    if hidden in EXPECTED_PARAMS:
        assert count_params(model) == EXPECTED_PARAMS[hidden], (
            f"Lệch tham số: {count_params(model)} != {EXPECTED_PARAMS[hidden]}"
        )

    opt_name = cfg.get("optimizer", "sgd_momentum")
    lr = float(cfg.get("lr", 0.05))
    wd = float(cfg.get("weight_decay", 0.0))
    momentum = float(cfg.get("momentum", 0.9))
    optimizer = build_optimizer(opt_name, model.parameters(), lr=lr, weight_decay=wd, momentum=momentum)

    precision = cfg.get("precision", "fp32").lower()
    use_amp = (precision == "fp16" and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # Step 0 loss trên val (trước cập nhật đầu tiên)
    loss_name = cfg.get("loss", "ce")
    step0_loss = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)["loss"]

    # Tập con cố định 50,000 mẫu của train để đo train_loss nhanh chóng
    n_tr_eval = min(50_000, len(data["X_tr"]))
    X_tr_eval = data["X_tr"][:n_tr_eval]
    y_tr_eval = data["y_tr"][:n_tr_eval]

    epochs = int(cfg.get("epochs", 20))
    batch_size = int(cfg.get("batch", 512))
    clip_norm = cfg.get("clip_norm", None)

    history = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "grad_norm": [],
        "epoch_time_s": [],
    }

    best_val_loss = float("inf")
    best_epoch = 1
    best_state = None
    diverged = False

    generator = torch.Generator(device=device).manual_seed(seed)

    for epoch in range(1, epochs + 1):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()

        model.train()
        epoch_grad_norms = []

        for xb, yb in iterate_batches(data["X_tr"], data["y_tr"], batch_size, generator=generator, shuffle=True):
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(xb)
                    loss = compute_loss(logits, yb, loss_name)

                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break

                scaler.scale(loss).backward()
                if clip_norm is not None:
                    scaler.unscale_(optimizer)
                gn = clip_gradients(model.parameters(), clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(xb)
                loss = compute_loss(logits, yb, loss_name)

                if torch.isnan(loss) or torch.isinf(loss):
                    diverged = True
                    break

                loss.backward()
                gn = clip_gradients(model.parameters(), clip_norm)
                optimizer.step()

            epoch_grad_norms.append(gn)

        if diverged:
            print(f"[{cfg.get('exp_id')}] DIVERGED tại epoch {epoch}!")
            break

        if device.type == "cuda":
            torch.cuda.synchronize()
        epoch_time = time.time() - t0

        tr_eval = evaluate(model, X_tr_eval, y_tr_eval, loss_name=loss_name)
        val_eval = evaluate(model, data["X_val"], data["y_val"], loss_name=loss_name)
        avg_gn = float(np.mean(epoch_grad_norms)) if epoch_grad_norms else 0.0

        history["epoch"].append(epoch)
        history["train_loss"].append(tr_eval["loss"])
        history["val_loss"].append(val_eval["loss"])
        history["val_acc"].append(val_eval["acc"])
        history["val_macro_f1"].append(val_eval["macro_f1"])
        history["grad_norm"].append(avg_gn)
        history["epoch_time_s"].append(epoch_time)

        if val_eval["loss"] < best_val_loss:
            best_val_loss = val_eval["loss"]
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Lấy metrics tại best_epoch
    if history["epoch"]:
        best_idx = best_epoch - 1
        best_val_acc = history["val_acc"][best_idx]
        best_val_macro_f1 = history["val_macro_f1"][best_idx]
        final_tr_loss = history["train_loss"][-1]
        final_val_loss = history["val_loss"][-1]
        avg_time = float(np.mean(history["epoch_time_s"]))
    else:
        best_val_acc, best_val_macro_f1, final_tr_loss, final_val_loss, avg_time = 0.0, 0.0, 0.0, 0.0, 0.0

    peak_mem_MB = 0.0
    if device.type == "cuda":
        peak_mem_MB = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    summary = {
        "step0_loss": float(step0_loss),
        "best_val_loss": float(best_val_loss),
        "best_epoch": int(best_epoch),
        "final_train_loss": float(final_tr_loss),
        "final_val_loss": float(final_val_loss),
        "val_acc": float(best_val_acc),
        "val_macro_f1": float(best_val_macro_f1),
        "time_per_epoch_s": float(avg_time),
        "peak_mem_MB": float(peak_mem_MB),
        "diverged": bool(diverged),
    }

    return {
        "cfg": cfg,
        "history": history,
        "summary": summary,
        "best_state": best_state,
    }


def write_predictions(row_id, preds, path: str) -> None:
    """Ghi file nộp cho scripts/evaluate.py: CSV có tiêu đề `row_id,pred`."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"row_id": row_id, "pred": preds})
    df.to_csv(path, index=False)
    print(f"Đã ghi dự đoán eval vào {path} ({len(df):,} dòng)")


def final_eval(cfg: dict, result: dict, data: dict, pred_path: str) -> None:
    """Dùng MỘT LẦN cho cấu hình cuối cùng (và baseline): nạp best_state, dự đoán eval, ghi predictions."""
    device = data["X_eval"].device
    hidden = tuple(cfg.get("hidden", (256, 128)))
    model = MLP(hidden=hidden, in_features=54, num_classes=7).to(device)
    model.load_state_dict(result["best_state"])
    model.eval()

    preds = predict(model, data["X_eval"])
    preds_np = preds.cpu().numpy()
    write_predictions(data["eval_row_id"], preds_np, pred_path)
