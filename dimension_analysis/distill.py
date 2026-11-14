import argparse
import json
import os
import random

import numpy as np
import torch

from distill_helpers import (
    evaluate,
    extract_embeddings,
    load_teacher_data,
    normalize_mu,
    predict_bert,
    save_plots,
    save_test_predictions,
    stratified_split,
    train_bert_regressor,
    train_ridge,
)


def _auto_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Distill TrueSkill teacher ratings into a BERT regressor.")

    parser.add_argument("--leaderboard", default="leaderboard.csv",
                        help="Path to leaderboard CSV with text column (default: leaderboard.csv)")
    parser.add_argument("--test-size", type=int, default=200,
                        help="Test set size, stratified by TrueSkill Uncertainty (default: 200)")
    parser.add_argument("--val-fraction", type=float, default=0.05,
                        help="Fraction of non-test data used for validation (default: 0.05)")
    parser.add_argument("--seed", type=int, default=2026,
                        help="Random seed (default: 2026)")
    parser.add_argument("--bert-model", default="nlpaueb/legal-bert-base-uncased",
                        help="HuggingFace model name or local path (default: nlpaueb/legal-bert-base-uncased)")
    parser.add_argument("--pooling", choices=["cls", "mean", "windowed"], default="cls",
                        help="Representation pooling strategy (default: cls)")
    parser.add_argument("--max-length", type=int, default=512,
                        help="Tokenizer max_length (default: 512)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="BERT fine-tuning batch size (default: 16)")
    parser.add_argument("--epochs", type=int, default=4,
                        help="Max BERT training epochs (default: 4)")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="AdamW learning rate (default: 2e-5)")
    parser.add_argument("--warmup-fraction", type=float, default=0.1,
                        help="Fraction of total steps for linear LR warmup (default: 0.1)")
    parser.add_argument("--weight-decay", type=float, default=0.01,
                        help="AdamW weight decay (default: 0.01)")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience in epochs (default: 2)")
    parser.add_argument("--ridge-alpha", type=float, default=1.0,
                        help="Ridge regularization strength (default: 1.0)")
    parser.add_argument("--embed-batch-size", type=int, default=32,
                        help="Batch size for frozen BERT embedding extraction (default: 32)")
    parser.add_argument("--device", default=None,
                        help="Device: cpu, cuda, or mps (default: auto-detect)")
    parser.add_argument("--output-dir", default=os.path.join("runs", "distill"),
                        help="Output directory for all artifacts (default: runs/distill)")
    parser.add_argument("--skip-bert", action="store_true",
                        help="Skip BERT fine-tuning; run Ridge baseline only")
    parser.add_argument("--skip-ridge", action="store_true",
                        help="Skip Ridge baseline; run BERT fine-tuning only")
    parser.add_argument("--no-plots", action="store_true",
                        help="Skip matplotlib plot generation")
    parser.add_argument("--normalize", default=None, choices=["0-1", "z-score"],
                        help="Normalize mu values before training: '0-1' or 'z-score' (default: None)")

    args = parser.parse_args()
    device = args.device or _auto_device()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=== Distillation pipeline ===")
    print(f"  Device     : {device}")
    print(f"  BERT model : {args.bert_model}")
    print(f"  Pooling    : {args.pooling}")
    print(f"  Seed       : {args.seed}")

    print("\n[1/4] Loading teacher data...")
    records = load_teacher_data(args.leaderboard)
    print(f"  Loaded {len(records)} rated laws.")

    if args.normalize:
        records = normalize_mu(records, args.normalize)
        print(f"  Normalized mu ({args.normalize}).")

    print("[2/4] Splitting data...")
    train_recs, val_recs, test_recs = stratified_split(records, args.test_size, args.val_fraction, args.seed)

    bert_preds = None
    ridge_preds = None
    bert_train_info = {}

    if not args.skip_bert:
        print(f"\n[3a/4] Training BERT regressor ({args.epochs} epochs max, patience={args.patience})...")
        model, tokenizer, bert_train_info = train_bert_regressor(
            train_records=train_recs,
            val_records=val_recs,
            model_name=args.bert_model,
            pooling=args.pooling,
            max_length=args.max_length,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            warmup_fraction=args.warmup_fraction,
            weight_decay=args.weight_decay,
            patience=args.patience,
            device=device,
            output_dir=args.output_dir,
        )
        if test_recs:
            print("  Predicting on test set (BERT)...")
            bert_preds = predict_bert(model, tokenizer, test_recs, args.max_length, args.batch_size, device)

    if not args.skip_ridge:
        print(f"\n[3b/4] Training Ridge baseline (frozen {args.bert_model})...")
        print("  Extracting embeddings for train set...")
        train_embs = extract_embeddings(args.bert_model, args.pooling, train_recs, args.max_length, args.embed_batch_size, device)
        ridge_pipe = train_ridge(train_recs, train_embs, args.ridge_alpha, args.output_dir)
        print("  Ridge fitted.")
        if test_recs:
            print("  Extracting embeddings for test set...")
            test_embs = extract_embeddings(args.bert_model, args.pooling, test_recs, args.max_length, args.embed_batch_size, device)
            ridge_preds = ridge_pipe.predict(test_embs)

    print("\n[4/4] Evaluation on test set...")
    metrics = {}
    if not test_recs:
        print("  No test set (small-n fallback). Skipping evaluation.")
    else:
        mu_true = np.array([r["mu"] for r in test_recs])
        if bert_preds is not None:
            metrics["bert"] = evaluate(mu_true, bert_preds, "BERT")
            metrics["bert"].update(bert_train_info)
        if ridge_preds is not None:
            metrics["ridge"] = evaluate(mu_true, ridge_preds, "Ridge")
        if not args.no_plots:
            save_plots(test_recs, bert_preds, ridge_preds, args.output_dir)
        save_test_predictions(test_recs, bert_preds, ridge_preds, args.output_dir)

    metrics_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")
    print("Done.")


if __name__ == "__main__":
    main()
