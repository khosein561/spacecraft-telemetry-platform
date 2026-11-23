"""
LSTM forecasting detector.

WHY THIS FILE EXISTS: the idea (Hundman et al.) is to learn what NORMAL looks like by training a
network to predict the next value from the previous L values. On normal data the prediction error
is small. When behaviour changes in a way the network has not seen (a context shift, a frozen
sensor, a changed oscillation) the error jumps, and thresholding the error finds the anomaly.

Choices, and why:
  * PyTorch rather than Keras/TensorFlow: current TensorFlow builds do not reliably support new
    NVIDIA GPU generations; PyTorch does, and the model is ~30 lines either way.
  * One model PER CHANNEL, trained on that channel's training split only (never on test data).
  * One-step-ahead forecast of one value: small, fast, easy to reason about. (The paper predicts
    several steps; that is listed as future work.)
  * Errors are EWMA-smoothed and thresholded dynamically (threshold.py) with no use of labels.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .baseline import DetectionResult
from .threshold import segmented_threshold, smooth_errors


class LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):  # x: (batch, length, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def make_windows(values, length: int):
    """Return X (n-length, length, 1) and y (n-length,): predict values[i+length] from values[i:i+length]."""
    v = np.asarray(values, dtype=np.float32)
    if len(v) <= length:
        raise ValueError(f"need more than {length} points, got {len(v)}")
    idx = np.arange(length)[None, :] + np.arange(len(v) - length)[:, None]
    return v[idx][..., None], v[length:]


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainReport:
    epochs: int
    train_loss: float
    val_loss: float
    device: str


def fit_forecaster(train_values, length: int = 48, hidden: int = 32, epochs: int = 8, lr: float = 3e-3,
                   batch_size: int = 128, seed: int = 0, val_fraction: float = 0.15, device: str | None = None):
    """Train on the training split. The last `val_fraction` is held out; the weights with the best
    validation loss are kept (so a bad late epoch can't ruin the model). Returns (model, TrainReport)."""
    device = device or pick_device()
    torch.manual_seed(seed)
    np.random.seed(seed)
    X, y = make_windows(train_values, length)
    split = int(len(X) * (1 - val_fraction))
    Xtr, ytr = torch.from_numpy(X[:split]).to(device), torch.from_numpy(y[:split]).to(device)
    Xva, yva = torch.from_numpy(X[split:]).to(device), torch.from_numpy(y[split:]).to(device)

    model = LSTMForecaster(hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    best_val, best_state, last_train = float("inf"), None, float("nan")
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(Xtr), generator=gen).to(device)
        total = 0.0
        for i in range(0, len(order), batch_size):
            b = order[i : i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[b]), ytr[b])
            loss.backward()
            opt.step()
            total += loss.item() * len(b)
        last_train = total / len(Xtr)
        model.eval()
        with torch.no_grad():
            val = loss_fn(model(Xva), yva).item() if len(Xva) else last_train
        if val < best_val:
            best_val, best_state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, TrainReport(epochs, last_train, best_val, device)


@torch.no_grad()
def prediction_errors(model, values, length: int, batch_size: int = 1024) -> np.ndarray:
    """|actual - predicted| for each point of `values` using only `values` itself as history.

    The first `length` points have no history yet, so their error is NaN (callers decide how to
    treat them). Train and test are separate recordings in SMAP/MSL, so the end of the training data
    is NOT valid history for the start of the test data: using it would create a fake jump.
    """
    v = np.asarray(values, dtype=np.float32)
    X, y = make_windows(v, length)
    device = next(model.parameters()).device
    preds = []
    for i in range(0, len(X), batch_size):
        preds.append(model(torch.from_numpy(X[i : i + batch_size]).to(device)).cpu().numpy())
    out = np.full(len(v), np.nan)
    out[length:] = np.abs(np.concatenate(preds) - y)
    return out


class LSTMDetector:
    name = "lstm"

    def __init__(self, length: int = 32, hidden: int = 32, epochs: int = 16, seed: int = 0, segment: int = 2000,
                 smoothing_span: int = 15, min_pct_decrease: float = 0.13, device: str | None = None):
        self.length, self.hidden, self.epochs, self.seed = length, hidden, epochs, seed
        self.segment, self.smoothing_span, self.min_pct_decrease, self.device = segment, smoothing_span, min_pct_decrease, device
        self.model = None
        self.report = None

    def fit(self, train_values) -> TrainReport:
        self.model, self.report = fit_forecaster(train_values, self.length, self.hidden, self.epochs, seed=self.seed, device=self.device)
        return self.report

    def detect(self, values) -> DetectionResult:
        if self.model is None:
            raise RuntimeError("call fit() first")
        raw = prediction_errors(self.model, values, self.length)
        raw[: self.length] = np.nanmedian(raw)  # unscored warm-up points get a neutral (median) error
        smooth = smooth_errors(raw, self.smoothing_span)
        thresholds, seqs = segmented_threshold(smooth, self.segment, min_pct_decrease=self.min_pct_decrease)
        return DetectionResult(seqs, smooth, thresholds)
