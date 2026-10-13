"""
Multi-Stock Nifty Decision Transformer Training Script.

Trains 15 models across 3 random stock batches × 5 lookback windows:
  Lookbacks : 30, 60, 90, 120, 150 days
  Batches   : B1, B2, B3 (each with 10 randomly sampled Nifty stocks)
  Features  : 10 stationary features (5 OHLCV + 5 Nifty extras)
  Horizon   : 5 steps — evaluated at H=1 (next-day) and H=5 (5-day)
  Split     : 80% train / 20% test (chronological)
  Costs     : 15 bps total (10 fee + 5 slippage)

Output:
  checkpoints/nifty_b{B}_T{L}d.pt   — 15 checkpoint files
  checkpoints/nifty_multi_stock_results.json
"""

import sys
import json
import time
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch

from ml.modules.module1_input import StockDataProcessor
from ml.pipeline import DependencyAwareTradingPipeline, PipelineConfig
from ml.training import train_pipeline_from_config
from ml.evaluation import evaluate_predictions, compute_classification_metrics, compute_financial_metrics

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

NIFTY_DATA_DIR  = Path("ml/data/nifty")
CHECKPOINT_DIR  = Path("checkpoints")
RESULTS_PATH    = CHECKPOINT_DIR / "nifty_multi_stock_results.json"

NUM_BATCHES      = 3
STOCKS_PER_BATCH = 10
LOOKBACKS        = [30, 60, 90, 120, 150]
HORIZON          = 5          # model trained with 5-step head
STRIDE           = 1
TRAIN_RATIO      = 0.80
COST_BPS         = 10.0
SLIPPAGE_BPS     = 5.0

# Batch seeds for reproducibility
BATCH_SEEDS = {1: 42, 2: 137, 3: 999}

# Files to explicitly exclude (metadata, aggregates, stubs)
EXCLUDE_NAMES = {"1d_data.csv", "stock_metadata.csv", "NIFTY50_all.csv", "INFRATEL.csv"}
MIN_CSV_BYTES = 50_000   # skip anything smaller than ~50 KB (stub / empty files)

# Architecture grid: (num_transformer_layers, num_attention_heads, ffn_dim, encoder_hidden_dim, epochs)
ARCH_GRID = {
    30:  (2, 4, 128, 64,  20),
    60:  (2, 4, 256, 64,  15),
    90:  (3, 4, 256, 128, 12),
    120: (4, 8, 512, 128, 10),
    150: (4, 8, 512, 256, 8),
}


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def get_all_stock_csvs() -> list:
    """Return all per-stock CSVs, excluding metadata/aggregate/stub files."""
    return [
        p for p in sorted(NIFTY_DATA_DIR.glob("*.csv"))
        if p.name not in EXCLUDE_NAMES
        and p.stat().st_size >= MIN_CSV_BYTES
    ]


def load_batch(csv_paths: list) -> pd.DataFrame:
    """Load and concatenate multiple Nifty stock CSVs into one DataFrame."""
    frames = []
    for path in csv_paths:
        try:
            df = pd.read_csv(path)
            # Normalise column name for asset identifier
            if "Symbol" not in df.columns and "Ticker" not in df.columns:
                df["Symbol"] = path.stem
            frames.append(df)
        except Exception as e:
            print(f"  ⚠  Skipping {path.name}: {e}")
    if not frames:
        raise ValueError("No valid stock CSVs loaded.")
    combined = pd.concat(frames, ignore_index=True)
    return combined


def evaluate_horizon(pipeline, X_test, M_test, Y_test, horizon_idx: int, device: str) -> dict:
    """
    Run evaluation on a single horizon slice.

    horizon_idx=0 → H=1 (next-day)
    horizon_idx=4 → H=5 (5-day)

    Slices Y_test[:, :, horizon_idx] and wraps it back into shape
    (batch, N, 1) so evaluate_predictions works without modification.
    """
    pipeline.eval()
    all_preds = []
    all_targets = []

    dataset = torch.utils.data.TensorDataset(X_test, M_test, Y_test)
    loader  = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)

    with torch.no_grad():
        for X_b, M_b, Y_b in loader:
            X_b = X_b.to(device)
            M_b = M_b.to(device)
            out = pipeline(X_b, M_b)

            # Extract self-forecasts: forecasts is (batch_size * N, T, N, H)
            forecasts = out["output"].forecasts.cpu().numpy()
            bs, N_assets, T_steps, _ = X_b.shape
            H_steps = Y_b.shape[-1]
            forecasts = forecasts.reshape(bs, N_assets, T_steps, N_assets, H_steps)
            last_fc   = forecasts[:, :, -1, :, :]        # (batch, N, N, H)

            if N_assets == 1:
                diag_fc = last_fc[:, 0, 0, :].reshape(-1, 1, H_steps)
            else:
                diag_fc = np.zeros((bs, N_assets, H_steps))
                for b in range(bs):
                    for i in range(N_assets):
                        diag_fc[b, i, :] = last_fc[b, i, i, :]

            pred_h = diag_fc[:, :, horizon_idx].reshape(-1)   # (batch*N,)
            true_h = Y_b.numpy()[:, :, horizon_idx].reshape(-1)

            all_preds.append(pred_h)
            all_targets.append(true_h)

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # Classification
    class_m = compute_classification_metrics(preds, targets)

    # Regression
    mse  = float(np.mean((preds - targets) ** 2))
    mae  = float(np.mean(np.abs(preds - targets)))
    rmse = float(np.sqrt(mse))
    ss_res = np.sum((targets - preds) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2   = float(1.0 - ss_res / (ss_tot + 1e-8))
    ic   = float(np.corrcoef(preds, targets)[0, 1]) if np.std(preds) > 1e-8 and np.std(targets) > 1e-8 else 0.0

    # Backtest
    positions  = np.sign(preds)
    pos_diff   = np.diff(positions, prepend=0)
    gross_ret  = positions * targets
    net_m      = compute_financial_metrics(gross_ret, weights_diff=pos_diff,
                                           transaction_cost_bps=COST_BPS,
                                           slippage_bps=SLIPPAGE_BPS)
    gross_m    = compute_financial_metrics(gross_ret, weights_diff=None)
    bnh_m      = compute_financial_metrics(targets,   weights_diff=None)

    return {
        # Classification
        "directional_accuracy":  round(class_m["directional_accuracy"] * 100, 2),
        "balanced_accuracy":     round(class_m["balanced_accuracy"] * 100, 2),
        "precision_up":          round(class_m["precision_up"] * 100, 2),
        "recall_up":             round(class_m["recall_up"] * 100, 2),
        "precision_down":        round(class_m["precision_down"] * 100, 2),
        "recall_down":           round(class_m["recall_down"] * 100, 2),
        "f1_score":              round(class_m["f1_score"], 4),
        "confusion_matrix":      class_m["confusion_matrix"],
        # Regression
        "mse":   round(mse, 8),
        "mae":   round(mae, 6),
        "rmse":  round(rmse, 6),
        "r2":    round(r2, 4),
        "ic":    round(ic, 4),
        # Net strategy
        "net_cumret":     round(net_m["cumulative_return"] * 100, 2),
        "net_annret":     round(net_m["annualized_return"] * 100, 2),
        "net_sharpe":     round(net_m["sharpe_ratio"], 3),
        "net_sortino":    round(net_m["sortino_ratio"], 3),
        "net_maxdd":      round(net_m["max_drawdown"] * 100, 2),
        "win_rate":       round(net_m["win_rate"] * 100, 2),
        "profit_factor":  round(net_m["profit_factor"], 3),
        "num_trades":     int(net_m["num_trades"]),
        "tx_cost_pct":    round(net_m["total_transaction_costs"] * 100, 3),
        # Gross strategy
        "gross_cumret":   round(gross_m["cumulative_return"] * 100, 2),
        "gross_sharpe":   round(gross_m["sharpe_ratio"], 3),
        # Buy & Hold benchmark
        "bnh_cumret":     round(bnh_m["cumulative_return"] * 100, 2),
        "bnh_annret":     round(bnh_m["annualized_return"] * 100, 2),
        "bnh_sharpe":     round(bnh_m["sharpe_ratio"], 3),
        "bnh_maxdd":      round(bnh_m["max_drawdown"] * 100, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Core training routine
# ──────────────────────────────────────────────────────────────────────────────

def train_one(batch_id: int, lookback: int, csv_paths: list, device: str) -> dict:
    """Train one (batch, lookback) model and return full eval dict."""
    n_layers, n_heads, ffn_dim, enc_dim, epochs = ARCH_GRID[lookback]
    ckpt_name = f"nifty_b{batch_id}_T{lookback}d.pt"
    stock_names = [p.stem for p in csv_paths]

    print(f"\n{'─'*70}")
    print(f"  Batch {batch_id} | Lookback {lookback}d | Stocks: {', '.join(stock_names)}")
    print(f"  Arch: {n_layers}L × {n_heads}H | FFN={ffn_dim} | Epochs={epochs}")
    print(f"{'─'*70}")

    # 1. Load data
    combined_df = load_batch(csv_paths)
    print(f"  ✓ Loaded {len(combined_df):,} rows from {len(csv_paths)} stocks")

    # 2. Process → MarketTensor (with all Nifty features)
    processor = StockDataProcessor(lookback_window=lookback, device=torch.device(device))
    market_tensor = processor.process(combined_df, ohlcv_only=False)
    N, T_full, F_raw = market_tensor.shape
    print(f"  ✓ MarketTensor: N={N} assets, T={T_full} days, F={F_raw} raw features")

    # 3. Create sliding sequences with 10-feature stationary block
    t0 = time.time()
    X, M, Y = processor.create_sequences(
        market_tensor,
        lookback=lookback,
        horizon=HORIZON,
        stride=STRIDE,
        stationary_features=True,
    )
    n_seq    = len(X)
    n_train  = int(n_seq * TRAIN_RATIO)
    n_test   = n_seq - n_train
    F_stat   = X.shape[-1]   # 10 if Nifty extras present, else 5

    print(f"  ✓ Sequences: {n_seq:,} total | Train={n_train:,} (80%) | Test={n_test:,} (20%)")
    print(f"  ✓ Feature dim after stationary transform: F={F_stat}")

    if n_test < 30:
        print(f"  ⚠  Only {n_test} test sequences — skipping (lookback too long for data).")
        return {"error": f"insufficient test sequences ({n_test})", "lookback": lookback, "batch": batch_id}

    X_train, M_train, Y_train = X[:n_train], M[:n_train], Y[:n_train]
    X_test,  M_test,  Y_test  = X[n_train:], M[n_train:], Y[n_train:]

    # 4. Configure pipeline
    config = PipelineConfig(
        lookback_window=lookback,
        forecast_horizon=HORIZON,
        num_sectors=max(2, N // 3),
        encoder_hidden_dim=enc_dim,
        regime_embedding_dim=enc_dim * 2,
        transformer_embedding_dim=enc_dim * 2,
        num_transformer_layers=n_layers,
        num_attention_heads=n_heads,
        transformer_ffn_dim=ffn_dim,
        transformer_dropout=0.1,
        encoder_dropout=0.1,
        regime_dropout=0.1,
        output_mode="forecasting",
        batch_size=32,
        num_epochs=epochs,
        learning_rate=1e-3,
        weight_decay=1e-5,
        early_stopping_patience=5,
        gradient_clip=1.0,
        device=device,
    )

    # 5. Train
    t_train = time.time()
    pipeline, history = train_pipeline_from_config(
        X_train, M_train, Y_train, config, save_name=ckpt_name
    )
    train_time = round(time.time() - t_train, 1)
    print(f"  ✓ Training done in {train_time}s")

    # 6. Evaluate H=1 and H=5
    print("  → Evaluating H=1 (next-day)...")
    h1_metrics = evaluate_horizon(pipeline, X_test, M_test, Y_test, horizon_idx=0, device=device)
    print(f"     Dir.Acc={h1_metrics['directional_accuracy']}%  |  Net Sharpe={h1_metrics['net_sharpe']}  |  "
          f"TN={h1_metrics['confusion_matrix']['true_negative']}")

    print("  → Evaluating H=5 (5-day)...")
    h5_metrics = evaluate_horizon(pipeline, X_test, M_test, Y_test, horizon_idx=4, device=device)
    print(f"     Dir.Acc={h5_metrics['directional_accuracy']}%  |  Net Sharpe={h5_metrics['net_sharpe']}  |  "
          f"TN={h5_metrics['confusion_matrix']['true_negative']}")

    # Train-set evaluation (H=1 only, for quick sanity check)
    print("  → Train-set sanity check H=1...")
    h1_train = evaluate_horizon(pipeline, X_train, M_train, Y_train, horizon_idx=0, device=device)
    print(f"     Train Dir.Acc={h1_train['directional_accuracy']}%")

    return {
        "batch_id":       batch_id,
        "lookback":       lookback,
        "stocks":         stock_names,
        "n_sequences":    n_seq,
        "n_train":        n_train,
        "n_test":         n_test,
        "feature_dim":    F_stat,
        "training_time_s": train_time,
        "checkpoint":     str(CHECKPOINT_DIR / ckpt_name),
        "h1_test":        h1_metrics,
        "h5_test":        h5_metrics,
        "h1_train":       h1_train,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def print_summary_tables(all_results: list):
    """Print averaged metrics across batches per lookback for H=1 and H=5."""

    def avg_across_batches(results: list, horizon_key: str, metric: str) -> float:
        vals = [r[horizon_key][metric] for r in results if horizon_key in r and "error" not in r]
        return round(float(np.mean(vals)), 2) if vals else float("nan")

    print("\n" + "=" * 90)
    print("FINAL RESULTS — AVERAGED ACROSS ALL BATCHES")
    print("=" * 90)

    for h_key, h_label in [("h1_test", "H=1  Next-Day"), ("h5_test", "H=5  5-Day")]:
        print(f"\n### {h_label} — Per-Lookback Average ({NUM_BATCHES} batches)")
        header = f"{'Lookback':>10} {'DirAcc%':>9} {'BalAcc%':>9} {'IC':>7} {'R²':>7} {'NetRet%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'BnHRet%':>9} {'TN':>6}"
        print(header)
        print("─" * len(header))
        for lb in LOOKBACKS:
            batch_results = [r for r in all_results if r.get("lookback") == lb]
            if not batch_results:
                continue
            row = (
                f"{lb:>9}d"
                f"  {avg_across_batches(batch_results, h_key, 'directional_accuracy'):>8}"
                f"  {avg_across_batches(batch_results, h_key, 'balanced_accuracy'):>8}"
                f"  {avg_across_batches(batch_results, h_key, 'ic'):>6}"
                f"  {avg_across_batches(batch_results, h_key, 'r2'):>6}"
                f"  {avg_across_batches(batch_results, h_key, 'net_cumret'):>8}"
                f"  {avg_across_batches(batch_results, h_key, 'net_sharpe'):>7}"
                f"  {avg_across_batches(batch_results, h_key, 'net_maxdd'):>7}"
                f"  {avg_across_batches(batch_results, h_key, 'bnh_cumret'):>8}"
            )
            # TN across batches
            tns = [r[h_key]["confusion_matrix"]["true_negative"] for r in batch_results if h_key in r and "error" not in r]
            avg_tn = int(round(np.mean(tns))) if tns else 0
            print(row + f"  {avg_tn:>5}")

    # Per-batch best lookback table (H=1 directional accuracy)
    print(f"\n### Best Lookback Per Batch (H=1 Directional Accuracy)")
    print(f"{'Batch':>7} {'Stocks (sample)':>30} {'Best LB':>8} {'DirAcc%':>9} {'Sharpe':>8} {'TN':>6}")
    print("─" * 75)
    for b in range(1, NUM_BATCHES + 1):
        batch_res = [r for r in all_results if r.get("batch_id") == b and "error" not in r]
        if not batch_res:
            continue
        best = max(batch_res, key=lambda r: r["h1_test"]["directional_accuracy"])
        stocks_str = ", ".join(best["stocks"][:4]) + (", …" if len(best["stocks"]) > 4 else "")
        tn = best["h1_test"]["confusion_matrix"]["true_negative"]
        print(f"  B{b}    {stocks_str:<30}  {best['lookback']:>4}d  "
              f"  {best['h1_test']['directional_accuracy']:>8}%  "
              f"  {best['h1_test']['net_sharpe']:>7}  {tn:>5}")

    print()


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 90)
    print("NIFTY MULTI-STOCK DECISION TRANSFORMER — MULTI-LOOKBACK TRAINING")
    print(f"  {NUM_BATCHES} batches × {len(LOOKBACKS)} lookbacks = {NUM_BATCHES * len(LOOKBACKS)} models")
    print(f"  Features: 10 stationary (5 OHLCV + VWAP deviation + Turnover + Trades + Deliverable ratio + %Deliv)")
    print(f"  Horizon: {HORIZON} steps → evaluated at H=1 and H=5")
    print(f"  Trading costs: {COST_BPS + SLIPPAGE_BPS:.0f} bps total")
    print("=" * 90)

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")

    all_stock_csvs = get_all_stock_csvs()
    print(f"Available Nifty stocks: {len(all_stock_csvs)}")

    all_results = []
    total_start = time.time()

    for batch_id in range(1, NUM_BATCHES + 1):
        seed = BATCH_SEEDS[batch_id]
        random.seed(seed)
        selected_csvs = random.sample(all_stock_csvs, min(STOCKS_PER_BATCH, len(all_stock_csvs)))

        print(f"\n{'═'*90}")
        print(f"  BATCH {batch_id} / {NUM_BATCHES}  (seed={seed})")
        print(f"  Stocks: {', '.join(p.stem for p in selected_csvs)}")
        print(f"{'═'*90}")

        for lookback in LOOKBACKS:
            result = train_one(batch_id, lookback, selected_csvs, device)
            all_results.append(result)

            # Checkpoint incremental results after every model
            with open(RESULTS_PATH, "w") as f:
                json.dump({
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "config": {
                        "num_batches": NUM_BATCHES,
                        "stocks_per_batch": STOCKS_PER_BATCH,
                        "lookbacks": LOOKBACKS,
                        "horizon": HORIZON,
                        "cost_bps": COST_BPS + SLIPPAGE_BPS,
                    },
                    "results": all_results,
                }, f, indent=2, default=str)

    total_time = round(time.time() - total_start, 1)
    print(f"\n✓ All {len(all_results)} runs completed in {total_time}s")
    print(f"✓ Results saved → {RESULTS_PATH}")

    print_summary_tables(all_results)


if __name__ == "__main__":
    main()
