"""data.py — PSEUDO-CODE. Bạn phải tự hoàn thiện mọi hàm có `raise NotImplementedError`.

Nhiệm vụ: nạp tập train/eval đã chia sẵn, tách validation từ train, chuẩn hoá, đưa lên thiết bị.

Điều kiện trước: đã chạy `python scripts/split_data.py` (tạo data/processed/train.npz, eval.npz).

Quy ước dữ liệu (xem README mục 2 và 3):
    X : float32, shape (N, 54)   — 10 cột đầu là số liên tục, 44 cột sau là nhị phân (one-hot)
    y : int64,   shape (N,)      — nhãn 0..6
Tập eval CHỈ dùng để chấm điểm cuối. Không dùng nó để chọn cấu hình, chuẩn hoá hay dừng sớm.
"""
from __future__ import annotations

import numpy as np
import torch

from pathlib import Path
from sklearn.model_selection import train_test_split

N_NUMERIC = 10  # số cột liên tục cần chuẩn hoá (cột 0..9)


def load_split(processed_dir: str = "data/processed"):
    """Nạp train và eval từ file .npz.

    Trả về: X_train_full, y_train_full, X_eval, y_eval, eval_row_id
    """
    p_dir = Path(processed_dir)
    train_path = p_dir / "train.npz"
    eval_path = p_dir / "eval.npz"

    if not train_path.exists() or not eval_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy file processed tại '{p_dir}'. Hãy chạy 'python scripts/split_data.py' trước."
        )

    tr = np.load(train_path)
    X_train_full = tr["X"].astype(np.float32)
    y_train_full = tr["y"].astype(np.int64)

    ev = np.load(eval_path)
    X_eval = ev["X"].astype(np.float32)
    y_eval = ev["y"].astype(np.int64)
    eval_row_id = ev["row_id"].astype(np.int64)

    assert X_train_full.shape == (464809, 54), f"Shape train lệch: {X_train_full.shape}"
    assert y_train_full.shape == (464809,), f"Shape y_train lệch: {y_train_full.shape}"
    assert X_eval.shape == (116203, 54), f"Shape eval lệch: {X_eval.shape}"
    assert y_eval.shape == (116203,), f"Shape y_eval lệch: {y_eval.shape}"
    assert len(eval_row_id) == 116203, f"Số row_id eval lệch: {len(eval_row_id)}"

    return X_train_full, y_train_full, X_eval, y_eval, eval_row_id


def make_val_split(X, y, val_fraction: float = 0.2, seed: int = 42):
    """Tách validation TỪ train (không đụng eval). Phân tầng theo nhãn.

    Trả về: X_tr, y_tr, X_val, y_val
    """
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=val_fraction, stratify=y, random_state=seed
    )
    return X_tr, y_tr, X_val, y_val


def fit_standardizer(X_tr):
    """Tính mean và std của N_NUMERIC cột đầu CHỈ trên tập train (sau khi tách val).

    Trả về: mean (shape (10,)), std (shape (10,))
    """
    mean = np.mean(X_tr[:, :N_NUMERIC], axis=0)
    std = np.std(X_tr[:, :N_NUMERIC], axis=0)
    std = np.where(std == 0.0, 1.0, std)
    return mean, std


def apply_standardizer(X, mean, std):
    """Trả về bản sao của X, trong đó 10 cột đầu được (x - mean) / std; 44 cột nhị phân giữ nguyên."""
    X_scaled = X.copy()
    X_scaled[:, :N_NUMERIC] = (X_scaled[:, :N_NUMERIC] - mean) / std
    return X_scaled


def prepare_data(device: str, val_fraction: float = 0.2, seed: int = 42,
                 processed_dir: str = "data/processed") -> dict:
    """Gộp các bước trên và đưa TOÀN BỘ dữ liệu lên `device` một lần (không dùng DataLoader).

    Trả về dict gồm các tensor trên device:
        X_tr, y_tr, X_val, y_val, X_eval, y_eval        (y là int64)
    và các mảng numpy: eval_row_id
    """
    X_train_full, y_train_full, X_eval, y_eval, eval_row_id = load_split(processed_dir)
    X_tr, y_tr, X_val, y_val = make_val_split(X_train_full, y_train_full, val_fraction=val_fraction, seed=seed)

    mean, std = fit_standardizer(X_tr)
    X_tr = apply_standardizer(X_tr, mean, std)
    X_val = apply_standardizer(X_val, mean, std)
    X_eval = apply_standardizer(X_eval, mean, std)

    dev = torch.device(device)
    X_tr_t = torch.as_tensor(X_tr, dtype=torch.float32, device=dev)
    y_tr_t = torch.as_tensor(y_tr, dtype=torch.int64, device=dev)
    X_val_t = torch.as_tensor(X_val, dtype=torch.float32, device=dev)
    y_val_t = torch.as_tensor(y_val, dtype=torch.int64, device=dev)
    X_eval_t = torch.as_tensor(X_eval, dtype=torch.float32, device=dev)
    y_eval_t = torch.as_tensor(y_eval, dtype=torch.int64, device=dev)

    maj_class = int(np.bincount(y_tr).argmax())
    majority_acc = float((y_val == maj_class).mean())
    print(f"Data ready on {device}: Train={len(X_tr):,}, Val={len(X_val):,}, Eval={len(X_eval):,}")
    print(f"Chiến lược đoán lớp đa số (lớp {maj_class}) trên val: accuracy = {majority_acc:.4f}")

    return {
        "X_tr": X_tr_t,
        "y_tr": y_tr_t,
        "X_val": X_val_t,
        "y_val": y_val_t,
        "X_eval": X_eval_t,
        "y_eval": y_eval_t,
        "eval_row_id": eval_row_id,
        "mean": mean,
        "std": std,
        "majority_acc": majority_acc,
    }


def iterate_batches(X, y, batch_size: int, generator: torch.Generator | None = None, shuffle: bool = True):
    """Generator trả về từng cặp (xb, yb), thay cho DataLoader."""
    n = len(X)
    if shuffle:
        perm = torch.randperm(n, generator=generator, device=X.device)
    else:
        perm = torch.arange(n, device=X.device)

    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        yield X[idx], y[idx]
