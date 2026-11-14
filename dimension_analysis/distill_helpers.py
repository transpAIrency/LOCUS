import csv
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup


def normalize_mu(records: list[dict], method: str) -> list[dict]:
    mus = np.array([r["mu"] for r in records])
    if method == "0-1":
        lo, hi = mus.min(), mus.max()
        normed = (mus - lo) / (hi - lo)
    elif method == "z-score":
        normed = (mus - mus.mean()) / mus.std()
    else:
        raise ValueError(f"Unknown normalize method: {method!r}. Use '0-1' or 'z-score'.")
    return [{**r, "mu": float(normed[i])} for i, r in enumerate(records)]


def load_teacher_data(leaderboard_path: str) -> list[dict]:
    """Return list of {id, mu, sigma, text} from leaderboard CSV."""
    if not os.path.exists(leaderboard_path):
        raise FileNotFoundError(
            f"Leaderboard CSV not found: {leaderboard_path}\n"
            "Run pairwise_compare.py first or pass --leaderboard."
        )
    csv.field_size_limit(10_000_000)
    records = []
    with open(leaderboard_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rec = {
                "id": row["id"],
                "mu": float(row["Skill (mu)"]),
                "sigma": float(row["Uncertainty (sigma)"]),
                "text": row.get("text", ""),
            }
            records.append(rec)
    missing = sum(1 for r in records if not r["text"])
    if missing:
        print(f"  Warning: {missing} rows have no text and will be dropped.")
    return [r for r in records if r["text"]]


def stratified_split(records: list[dict], test_size: int, val_fraction: float, seed: int):
    """Returns (train, val, test) lists stratified on μ.

    Small-n fallback (< 400): skips test holdout, uses 60/20 train/val split.
    """
    n = len(records)
    rng = np.random.default_rng(seed)

    if n < 400:
        print(
            f"  Warning: only {n} rated laws (< 400). Skipping test holdout; "
            "using 60/20 train/val split instead."
        )
        idx = rng.permutation(n)
        n_val = max(1, int(n * 0.20))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        return [records[i] for i in train_idx], [records[i] for i in val_idx], []

    mus = np.array([r["mu"] for r in records])
    strata = pd.qcut(mus, q=5, labels=False)

    test_indices = []
    per_stratum = test_size // 5
    for s in range(5):
        s_idx = np.where(strata == s)[0]
        chosen = rng.choice(s_idx, size=min(per_stratum, len(s_idx)), replace=False)
        test_indices.extend(chosen.tolist())

    test_mask = np.zeros(n, dtype=bool)
    test_mask[test_indices] = True
    remaining_idx = np.where(~test_mask)[0]

    remaining_records = [records[i] for i in remaining_idx]
    remaining_strata = strata[remaining_idx]
    n_remaining = len(remaining_records)
    n_val = max(1, int(n_remaining * val_fraction))

    val_indices = []
    val_per_stratum = n_val // 5
    for s in range(5):
        s_idx = np.where(remaining_strata == s)[0]
        chosen = rng.choice(s_idx, size=min(val_per_stratum, len(s_idx)), replace=False)
        val_indices.extend(chosen.tolist())

    val_mask = np.zeros(n_remaining, dtype=bool)
    val_mask[val_indices] = True

    train = [remaining_records[i] for i in range(n_remaining) if not val_mask[i]]
    val = [remaining_records[i] for i in range(n_remaining) if val_mask[i]]
    test = [records[i] for i in test_indices]

    boundaries = pd.qcut(mus, q=5, retbins=True)[1]
    print(f"  Stratum μ-boundaries: {[round(b, 3) for b in boundaries]}")
    print(f"  Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")

    return train, val, test


class LawDataset(Dataset):
    def __init__(self, records: list[dict], tokenizer, max_length: int):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        enc = self.tokenizer(
            rec["text"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "mu": torch.tensor(rec["mu"], dtype=torch.float32),
        }


class BertRegressor(nn.Module):
    def __init__(self, model_name: str, pooling: str):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        hidden = self.bert.config.hidden_size
        self.head = nn.Linear(hidden, 1)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last = out.last_hidden_state  # (B, L, H)
        if self.pooling == "cls":
            rep = last[:, 0, :]
        elif self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).float()
            rep = (last * mask).sum(1) / mask.sum(1)
        elif self.pooling == "windowed":
            raise NotImplementedError(
                "--pooling windowed is not yet implemented."
            )
        else:
            raise ValueError(f"Unknown pooling: {self.pooling!r}")
        return self.head(rep).squeeze(-1)  # (B,)


def train_bert_regressor(
    train_records: list[dict],
    val_records: list[dict],
    model_name: str,
    pooling: str,
    max_length: int,
    batch_size: int,
    epochs: int,
    lr: float,
    warmup_fraction: float,
    weight_decay: float,
    patience: int,
    device: str,
    output_dir: str,
) -> tuple[BertRegressor, AutoTokenizer, dict]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    train_loader = DataLoader(LawDataset(train_records, tokenizer, max_length), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(LawDataset(val_records, tokenizer, max_length), batch_size=batch_size)

    model = BertRegressor(model_name, pooling).to(device)
    model.bert.gradient_checkpointing_enable()
    use_amp = device == "cuda"
    scaler = GradScaler() if use_amp else None
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * warmup_fraction), total_steps)

    checkpoint_dir = os.path.join(output_dir, "bert_regressor")
    os.makedirs(checkpoint_dir, exist_ok=True)
    head_ckpt = os.path.join(checkpoint_dir, "head.pt")

    best_val_mse = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mu = batch["mu"].to(device)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16, enabled=use_amp):
                loss = F.mse_loss(model(ids, mask), mu)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                mu = batch["mu"].to(device)
                with autocast(dtype=torch.bfloat16, enabled=use_amp):
                    val_losses.append(F.mse_loss(model(ids, mask), mu).item())

        train_mse = np.mean(train_losses)
        val_mse = np.mean(val_losses)
        history.append({"epoch": epoch, "train_mse": float(train_mse), "val_mse": float(val_mse)})
        print(f"  Epoch {epoch}/{epochs}  train_mse={train_mse:.4f}  val_mse={val_mse:.4f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            no_improve = 0
            model.bert.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            torch.save(model.head.state_dict(), head_ckpt)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (no val improvement for {patience} epochs).")
                break

    model.bert = AutoModel.from_pretrained(checkpoint_dir).to(device)
    model.head.load_state_dict(torch.load(head_ckpt, map_location=device))
    return model, tokenizer, {"training_history": history, "best_val_mse": float(best_val_mse)}


def predict_bert(model: BertRegressor, tokenizer, records: list[dict], max_length: int, batch_size: int, device: str) -> np.ndarray:
    loader = DataLoader(LawDataset(records, tokenizer, max_length), batch_size=batch_size)
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            preds.append(model(ids, mask).cpu().numpy())
    return np.concatenate(preds)


def extract_embeddings(model_name: str, pooling: str, records: list[dict], max_length: int, batch_size: int, device: str) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    bert = AutoModel.from_pretrained(model_name).to(device)
    bert.eval()
    loader = DataLoader(LawDataset(records, tokenizer, max_length), batch_size=batch_size)
    all_embs = []
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            last = bert(input_ids=ids, attention_mask=mask).last_hidden_state
            if pooling == "cls":
                emb = last[:, 0, :]
            else:
                m = mask.unsqueeze(-1).float()
                emb = (last * m).sum(1) / m.sum(1)
            all_embs.append(emb.cpu().numpy())
    return np.concatenate(all_embs)


def train_ridge(train_records: list[dict], train_embeddings: np.ndarray, alpha: float, output_dir: str) -> Pipeline:
    mu_train = np.array([r["mu"] for r in train_records])
    pipe = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=alpha))])
    pipe.fit(train_embeddings, mu_train)
    joblib.dump(pipe, os.path.join(output_dir, "ridge_baseline.pkl"))
    return pipe


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> dict:
    rho, pval = spearmanr(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    print(f"  [{name}] Spearman ρ={rho:.4f} (p={pval:.2e})  R²={r2:.4f}  RMSE={rmse:.4f}")
    return {"spearman_r": float(rho), "spearman_p": float(pval), "r2": float(r2), "rmse": rmse}


def save_test_predictions(test_records: list[dict], bert_preds, ridge_preds, output_dir: str) -> None:
    path = os.path.join(output_dir, "test_predictions.csv")
    fieldnames = ["law_id", "mu_teacher", "sigma_teacher"]
    if bert_preds is not None:
        fieldnames.append("mu_bert")
    if ridge_preds is not None:
        fieldnames.append("mu_ridge")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, rec in enumerate(test_records):
            row = {"law_id": rec["id"], "mu_teacher": rec["mu"], "sigma_teacher": rec["sigma"]}
            if bert_preds is not None:
                row["mu_bert"] = float(bert_preds[i])
            if ridge_preds is not None:
                row["mu_ridge"] = float(ridge_preds[i])
            writer.writerow(row)
    print(f"  Saved {path}")


def save_plots(test_records: list[dict], bert_preds, ridge_preds, output_dir: str) -> None:
    mu_teacher = np.array([r["mu"] for r in test_records])
    sigma_teacher = np.array([r["sigma"] for r in test_records])
    ref_lo = mu_teacher.min() - 1
    ref_hi = mu_teacher.max() + 1

    def _calibration_plot(mu_pred, name, path):
        fig, ax = plt.subplots(figsize=(6, 6))
        sc = ax.scatter(mu_teacher, mu_pred, c=sigma_teacher, cmap="viridis", alpha=0.7, s=30)
        ax.plot([ref_lo, ref_hi], [ref_lo, ref_hi], "k--", linewidth=0.8, label="y = x")
        ax.set_xlabel("Teacher μ")
        ax.set_ylabel(f"Predicted μ ({name})")
        ax.set_title(f"Calibration: {name}")
        plt.colorbar(sc, ax=ax, label="Teacher σ")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")

    if bert_preds is not None:
        _calibration_plot(bert_preds, "BERT", os.path.join(output_dir, "calibration_bert.png"))
    if ridge_preds is not None:
        _calibration_plot(ridge_preds, "Ridge", os.path.join(output_dir, "calibration_ridge.png"))

    panels = []
    if bert_preds is not None:
        panels.append((bert_preds - mu_teacher, "BERT"))
    if ridge_preds is not None:
        panels.append((ridge_preds - mu_teacher, "Ridge"))

    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 5), squeeze=False)
        for ax, (residuals, name) in zip(axes[0], panels):
            ax.scatter(sigma_teacher, residuals, alpha=0.6, s=30, color="steelblue")
            ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
            ax.set_xlabel("Teacher σ")
            ax.set_ylabel("Residual (predicted − teacher μ)")
            ax.set_title(f"Residuals vs. teacher σ: {name}")
        fig.tight_layout()
        path = os.path.join(output_dir, "residuals.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"  Saved {path}")
