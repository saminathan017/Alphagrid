"""
models/transformer_model.py
Transformer encoder for time-series price direction prediction.
Uses multi-head self-attention with positional encoding.
"""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from typing import Optional
from loguru import logger
from core.config import settings


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class FinancialTransformer(nn.Module):
    """
    Transformer encoder for financial time-series.

    Architecture:
        Input projection → Positional Encoding →
        N × (Multi-Head Attention + FFN + LayerNorm) →
        Global Average Pool → Classification Head
    """

    def __init__(
        self,
        input_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_encoder_layers: int = 6,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        num_classes: int = 1,
    ) -> None:
        super().__init__()

        # Input projection to d_model
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout=dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,     # Pre-LN for training stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_encoder_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1     = nn.Linear(d_model, d_model // 2)
        self.fc2     = nn.Linear(d_model // 2, num_classes)
        self.norm    = nn.LayerNorm(d_model // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, seq_len, input_size)
        returns: (batch_size, 1) — sigmoid [0,1]
        """
        # Project to model dimension
        x = self.input_proj(x)          # (B, T, d_model)
        x = self.pos_encoding(x)

        # Transformer encoding
        x = self.transformer_encoder(x) # (B, T, d_model)

        # Global average pooling over sequence
        x = x.mean(dim=1)               # (B, d_model)

        # Classification
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.norm(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return torch.sigmoid(x)


class WarmupCosineScheduler:
    """Linear warmup + cosine annealing LR scheduler."""

    def __init__(self, optimizer, warmup_steps: int, total_steps: int) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self._step = 0
        self._base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self) -> None:
        self._step += 1
        if self._step <= self.warmup_steps:
            factor = self._step / max(1, self.warmup_steps)
        else:
            progress = (self._step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            factor = 0.5 * (1 + math.cos(math.pi * progress))

        for g, base_lr in zip(self.optimizer.param_groups, self._base_lrs):
            g["lr"] = base_lr * factor


class TransformerModel:
    """Training, inference, and persistence wrapper for FinancialTransformer."""

    def __init__(self) -> None:
        cfg = settings.get("models", {}).get("transformer", {})
        # Defaults — overridden by _adapt_to_data() at train time
        self.d_model          = cfg.get("d_model", 256)
        self.n_heads          = cfg.get("n_heads", 8)
        self.n_encoder_layers = cfg.get("n_encoder_layers", 6)
        self.dim_feedforward  = cfg.get("dim_feedforward", 512)
        self.dropout          = cfg.get("dropout", 0.1)
        self.seq_len          = cfg.get("sequence_length", 60)
        self.batch_size       = cfg.get("batch_size", 32)
        self.epochs           = cfg.get("epochs", 100)
        self.lr               = cfg.get("learning_rate", 0.0001)
        self.warmup_steps     = cfg.get("warmup_steps", 4000)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        self.model: Optional[FinancialTransformer] = None
        self.input_size: Optional[int] = None
        self._calibrator = None    # isotonic calibrator (Fix 4)
        logger.info(f"TransformerModel using device: {self.device}")

    # ── Fix 1: Data-adaptive architecture ────────────────────────────────────
    def _adapt_to_data(self, n_train: int) -> None:
        """Scale Transformer capacity to training set size."""
        if n_train < 400:
            self.d_model          = 64
            self.n_heads          = 2
            self.n_encoder_layers = 2
            self.dim_feedforward  = 128
            self.dropout          = 0.40
            logger.info(f"Transformer tiny config  (n={n_train}): d=64, layers=2, heads=2")
        elif n_train < 800:
            self.d_model          = 96
            self.n_heads          = 4
            self.n_encoder_layers = 3
            self.dim_feedforward  = 192
            self.dropout          = 0.30
            logger.info(f"Transformer small config (n={n_train}): d=96, layers=3, heads=4")
        elif n_train < 1800:
            self.d_model          = 128
            self.n_heads          = 4
            self.n_encoder_layers = 4
            self.dim_feedforward  = 256
            self.dropout          = 0.20
            logger.info(f"Transformer medium config(n={n_train}): d=128, layers=4, heads=4")
        else:
            logger.info(f"Transformer full config  (n={n_train}): d=256, layers=6, heads=8")

    def _build_model(self, input_size: int) -> FinancialTransformer:
        model = FinancialTransformer(
            input_size=input_size,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_encoder_layers=self.n_encoder_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
        ).to(self.device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Transformer built: {n_params:,} trainable parameters")
        return model

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict[str, list[float]]:
        self.input_size = X_train.shape[2]
        self._adapt_to_data(len(X_train))     # Fix 1: scale capacity to data size
        self.model = self._build_model(self.input_size)

        train_ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1),
        )
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)

        val_loader = None
        if X_val is not None:
            val_ds = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val).unsqueeze(1),
            )
            val_loader = DataLoader(val_ds, batch_size=self.batch_size)

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-4, betas=(0.9, 0.98)
        )
        total_steps = self.epochs * len(train_loader)
        scheduler = WarmupCosineScheduler(optimizer, self.warmup_steps, total_steps)
        criterion = nn.BCELoss()

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": [], "val_accuracy": []}
        best_val_loss = float("inf")
        patience_counter = 0
        patience = 15

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            train_losses = []
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                output = self.model(X_batch)
                loss = criterion(output, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            avg_train = np.mean(train_losses)
            history["train_loss"].append(avg_train)

            if val_loader:
                val_loss, val_acc = self._evaluate(val_loader, criterion)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_acc)

                if epoch % 10 == 0:
                    logger.info(
                        f"Epoch {epoch:3d}/{self.epochs} | train={avg_train:.4f} | "
                        f"val={val_loss:.4f} | acc={val_acc:.3f}"
                    )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save("models/transformer_best.pt")
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break

        return history

    def _evaluate(self, loader: DataLoader, criterion: nn.Module) -> tuple[float, float]:
        self.model.eval()
        losses, correct, total = [], 0, 0
        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                out     = self.model(X_batch)
                losses.append(criterion(out, y_batch).item())
                preds   = (out > 0.5).float()
                correct += (preds == y_batch).sum().item()
                total   += y_batch.size(0)
        return np.mean(losses), correct / (total + 1e-9)

    # ── Fix 4: Isotonic calibration ──────────────────────────────────────────
    def calibrate(self, probs_val: np.ndarray, y_val: np.ndarray) -> None:
        """Fit isotonic regression on validation set. Call after train(), before test predict()."""
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(out_of_bounds="clip")
            ir.fit(probs_val.flatten(), y_val.flatten().astype(float))
            self._calibrator = ir
            cal_probs = ir.predict(probs_val.flatten())
            logger.info(
                f"Transformer calibrated | avg confidence: "
                f"{float(np.abs(probs_val - 0.5).mean()):.4f} → "
                f"{float(np.abs(cal_probs - 0.5).mean()):.4f}"
            )
        except Exception as e:
            logger.warning(f"Transformer calibration failed: {e}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not initialized.")
        self.model.eval()
        with torch.no_grad():
            t = torch.FloatTensor(X).to(self.device)
            probs = self.model(t).cpu().numpy().flatten()
        # Apply isotonic calibration if fitted (Fix 4)
        if self._calibrator is not None:
            probs = np.clip(self._calibrator.predict(probs), 1e-6, 1 - 1e-6)
        return probs

    def predict_single(self, X: np.ndarray) -> tuple[str, float]:
        if X.ndim == 2:
            X = X[np.newaxis, :]
        prob = float(self.predict(X)[0])
        if prob > 0.60:
            return "UP", prob
        elif prob < 0.40:
            return "DOWN", 1 - prob
        return "FLAT", 0.5

    def save(self, path: str = "models/transformer_trained.pt") -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "input_size": self.input_size,
            "config": {
                "d_model": self.d_model, "n_heads": self.n_heads,
                "n_encoder_layers": self.n_encoder_layers,
                "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout,
            }
        }, path)
        logger.info(f"Transformer saved: {path}")
        return path

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.input_size = ckpt["input_size"]
        cfg = ckpt["config"]
        self.model = FinancialTransformer(
            input_size=self.input_size,
            d_model=cfg["d_model"], n_heads=cfg["n_heads"],
            n_encoder_layers=cfg["n_encoder_layers"],
            dim_feedforward=cfg["dim_feedforward"],
            dropout=cfg["dropout"],
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        logger.info(f"Transformer loaded from {path}")
