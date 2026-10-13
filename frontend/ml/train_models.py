"""
Production Training and Evaluation Script for Dow 30 and Nifty 50.

Features & Recalibrations:
1. Leak-Free Pipeline: Zero ground-truth targets passed to Decision Transformer RTG.
2. Stationary Scale-Invariant Inputs: 1-day log returns, high-low spread, close-open ratio, shadows, rolling volume.
3. Strict Chronological Split: 80% Train, 20% Out-Of-Sample Test.
4. Mathematically Rigorous Backtesting: Realized 1-day forward step returns compounded without overlapping horizon errors.
5. Realistic Trading Costs: 10 bps transaction fee + 5 bps slippage per trade turnover.
6. Full Classification & Regression Suite: Accuracy, Precision, Recall, F1, Confusion Matrix, MSE, RMSE, MAE, Pearson IC, R^2.
7. Benchmark Comparisons: Buy & Hold Benchmark and Naive Momentum Baseline.
"""

import sys
import json
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import torch

from ml.modules.module1_input import StockDataProcessor
from ml.pipeline import DependencyAwareTradingPipeline, PipelineConfig
from ml.training import train_pipeline_from_config
from ml.inference import run_inference


def train_and_evaluate_asset(
    name: str,
    data_path: Path,
    checkpoint_name: str,
    epochs: int = 15,
    batch_size: int = 64,
    lookback: int = 60,
    horizon: int = 5,
    stride: int = 1,
    lr: float = 0.001,
) -> dict:
    """Train pipeline on asset data and return comprehensive evaluation metrics."""
    print("=" * 80)
    print(f"TRAINING PIPELINE FOR {name.upper()} (1-day OHLCV Data - Leak-Free)")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load data
    print(f"\n[1/5] Loading data from {data_path}...")
    df = pd.read_csv(data_path, sep="\t")
    print(f"  ✓ Loaded {len(df):,} bars from {df['DateTime'].iloc[0]} to {df['DateTime'].iloc[-1]}")
    
    # 2. Process into MarketTensor (purely OHLCV)
    print("\n[2/5] Aligning data into MarketTensor (OHLCV only)...")
    processor = StockDataProcessor(lookback_window=lookback, device=torch.device(device))
    market_tensor = processor.process(df, ohlcv_only=True)
    print(f"  ✓ MarketTensor shape: {market_tensor.shape} (N={market_tensor.num_assets}, T={market_tensor.lookback_window}, F={market_tensor.num_features})")
    
    # 3. Create sliding window sequences with stationary features and zero lookahead
    print(f"\n[3/5] Creating sliding sequences (lookback={lookback}, horizon={horizon}, stride={stride}, stationary=True)...")
    X, M, Y = processor.create_sequences(
        market_tensor,
        lookback=lookback,
        horizon=horizon,
        stride=stride,
        stationary_features=True,
    )
    total_seqs = len(X)
    n_train = int(total_seqs * 0.8)
    n_test = total_seqs - n_train
    print(f"  ✓ Total sequences: {total_seqs:,}")
    print(f"  ✓ Train set (80%): {n_train:,} sequences")
    print(f"  ✓ Test set  (20%): {n_test:,} sequences")
    
    # 4. Configure pipeline
    config = PipelineConfig(
        lookback_window=lookback,
        forecast_horizon=horizon,
        num_sectors=2,
        encoder_hidden_dim=64,
        regime_embedding_dim=64,
        transformer_embedding_dim=64,
        num_transformer_layers=2,
        num_attention_heads=4,
        transformer_ffn_dim=128,
        transformer_dropout=0.1,
        output_mode="forecasting",
        batch_size=batch_size,
        num_epochs=epochs,
        learning_rate=lr,
        early_stopping_patience=4,
        device=device,
    )
    
    # 5. Train pipeline without target leakage
    print(f"\n[4/5] Training pipeline for up to {epochs} epochs (80/20 train/test split)...")
    t0 = time.time()
    pipeline, history = train_pipeline_from_config(
        X, M, Y, config, save_name=checkpoint_name
    )
    t1 = time.time()
    print(f"  ✓ Training completed in {t1 - t0:.1f}s")
    
    # 6. Verify checkpoint and run inference
    print(f"\n[5/5] Running inference verification on recent market window...")
    inf_results = run_inference(pipeline, market_tensor, target_rtg=0.0)
    recent_regime = inf_results["regimes"][0, -1]
    recent_probs = inf_results["regime_probs"][0, -1]
    print(f"  ✓ Inference successful!")
    print(f"    - Current detected regime: R{recent_regime + 1}")
    print(f"    - Regime probabilities: {[round(float(p), 4) for p in recent_probs]}")
    if "forecasts" in inf_results:
        f_sample = inf_results["forecasts"][0, -1, 0]
        print(f"    - 5-step return forecasts: {[round(float(f), 4) for f in f_sample]}")
        
    train_m = history["train_metrics"]
    test_m = history["test_metrics"]
    
    def format_metrics(m: dict) -> dict:
        return {
            # Classification
            "directional_accuracy_pct": round(m.get("directional_accuracy", 0.0) * 100, 2),
            "precision_up_pct": round(m.get("precision_up", 0.0) * 100, 2),
            "recall_up_pct": round(m.get("recall_up", 0.0) * 100, 2),
            "precision_down_pct": round(m.get("precision_down", 0.0) * 100, 2),
            "recall_down_pct": round(m.get("recall_down", 0.0) * 100, 2),
            "f1_score": round(m.get("f1_score", 0.0), 4),
            "balanced_accuracy_pct": round(m.get("balanced_accuracy", 0.0) * 100, 2),
            "confusion_matrix": m.get("confusion_matrix", {}),
            # Regression
            "mse_1d": float(f"{m.get('mse', 0.0):.6f}"),
            "mae_1d": float(f"{m.get('mae', 0.0):.6f}"),
            "rmse_1d": float(f"{m.get('rmse', 0.0):.6f}"),
            "r2_score": float(f"{m.get('r2_score', 0.0):.4f}"),
            "information_coefficient": float(f"{m.get('information_coefficient', 0.0):.4f}"),
            # Trading Strategy Net (after 15 bps fee+slippage)
            "net_cumulative_return_pct": round(m.get("cumulative_return", 0.0) * 100, 2),
            "net_annualized_return_pct": round(m.get("annualized_return", 0.0) * 100, 2),
            "annualized_volatility_pct": round(m.get("volatility", 0.0) * 100, 2),
            "net_sharpe_ratio": round(m.get("sharpe_ratio", 0.0), 2),
            "net_sortino_ratio": round(m.get("sortino_ratio", 0.0), 2),
            "net_max_drawdown_pct": round(m.get("max_drawdown", 0.0) * 100, 2),
            "max_drawdown_duration_days": int(m.get("max_drawdown_duration", 0)),
            # Trading Strategy Gross
            "gross_cumulative_return_pct": round(m.get("gross_cumulative_return", 0.0) * 100, 2),
            "gross_annualized_return_pct": round(m.get("gross_annualized_return", 0.0) * 100, 2),
            "gross_sharpe_ratio": round(m.get("gross_sharpe_ratio", 0.0), 2),
            "gross_max_drawdown_pct": round(m.get("gross_max_drawdown", 0.0) * 100, 2),
            # Trade Diagnostics
            "win_rate_pct": round(m.get("win_rate", 0.0) * 100, 2),
            "profit_factor": round(m.get("profit_factor", 0.0), 2),
            "num_trades": int(m.get("num_trades", 0)),
            "win_loss_ratio": round(m.get("win_loss_ratio", 0.0), 2),
            "total_transaction_costs_pct": round(m.get("total_transaction_costs", 0.0) * 100, 3),
            "turnover_avg_daily": round(m.get("turnover", 0.0), 4),
            # Benchmarks
            "benchmark_buy_and_hold": {
                "cumulative_return_pct": round(m.get("benchmark_buy_and_hold", {}).get("cumulative_return", 0.0) * 100, 2),
                "annualized_return_pct": round(m.get("benchmark_buy_and_hold", {}).get("annualized_return", 0.0) * 100, 2),
                "sharpe_ratio": round(m.get("benchmark_buy_and_hold", {}).get("sharpe_ratio", 0.0), 2),
                "max_drawdown_pct": round(m.get("benchmark_buy_and_hold", {}).get("max_drawdown", 0.0) * 100, 2),
            },
            "benchmark_momentum": {
                "cumulative_return_pct": round(m.get("benchmark_momentum", {}).get("cumulative_return", 0.0) * 100, 2),
                "sharpe_ratio": round(m.get("benchmark_momentum", {}).get("sharpe_ratio", 0.0), 2),
                "max_drawdown_pct": round(m.get("benchmark_momentum", {}).get("max_drawdown", 0.0) * 100, 2),
            },
            "loss": float(f"{m.get('loss', 0.0):.6f}"),
        }
        
    return {
        "name": name,
        "total_bars": len(df),
        "total_sequences": total_seqs,
        "train_sequences": n_train,
        "test_sequences": n_test,
        "training_time_seconds": round(t1 - t0, 1),
        "checkpoint_path": str(Path("checkpoints") / checkpoint_name),
        "train_metrics": format_metrics(train_m),
        "test_metrics": format_metrics(test_m),
    }


def main():
    print("=" * 80)
    print("DEPENDENCY-AWARE TRADING SYSTEM: RECALIBRATED PRODUCTION TRAINING")
    print("=" * 80)
    
    checkpoints_dir = Path("checkpoints")
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Train Dow 30
    dow_results = train_and_evaluate_asset(
        name="Dow 30",
        data_path=Path("ml/data/dow30/1d_data_clean.csv"),
        checkpoint_name="dow30_1d_model.pt",
        epochs=12,
        batch_size=64,
        lookback=60,
        horizon=5,
        stride=1,
        lr=0.001,
    )
    
    # 2. Train Nifty 50
    nifty_results = train_and_evaluate_asset(
        name="Nifty 50",
        data_path=Path("ml/data/nifty/1d_data.csv"),
        checkpoint_name="nifty_1d_model.pt",
        epochs=10,
        batch_size=64,
        lookback=60,
        horizon=5,
        stride=1,
        lr=0.001,
    )
    
    # 3. Save results summary
    summary = {
        "dow30": dow_results,
        "nifty": nifty_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_path = checkpoints_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Saved full results summary to {summary_path}")
    
    # 4. Print clean markdown summary tables
    print("\n" + "=" * 80)
    print("RIGOROUS PERFORMANCE AND ACCURACY METRICS (80% TRAIN / 20% TEST)")
    print("=" * 80)
    
    for res in [dow_results, nifty_results]:
        print(f"\n### {res['name']} Performance Summary")
        print(f"- **Data**: 1-day OHLCV stationary ({res['total_bars']:,} bars, {res['total_sequences']:,} sequences)")
        print(f"- **Split**: Train={res['train_sequences']:,} (80%) | Test={res['test_sequences']:,} (20%)")
        print(f"- **Checkpoint**: `{res['checkpoint_path']}`")
        print(f"- **Training Time**: {res['training_time_seconds']}s")
        print()
        
        tm = res["train_metrics"]
        em = res["test_metrics"]
        
        print("#### 1. Predictive Performance (Classification & Regression)")
        print("| Metric | Train Set (80%) | Test Set (20%) |")
        print("| :--- | :--- | :--- |")
        print(f"| **1-Day Directional Accuracy** | **{tm['directional_accuracy_pct']}%** | **{em['directional_accuracy_pct']}%** |")
        print(f"| Precision (UP / DOWN) | {tm['precision_up_pct']}% / {tm['precision_down_pct']}% | {em['precision_up_pct']}% / {em['precision_down_pct']}% |")
        print(f"| Recall (UP / DOWN) | {tm['recall_up_pct']}% / {tm['recall_down_pct']}% | {em['recall_up_pct']}% / {em['recall_down_pct']}% |")
        print(f"| F1-Score | {tm['f1_score']} | {em['f1_score']} |")
        print(f"| Balanced Accuracy | {tm['balanced_accuracy_pct']}% | {em['balanced_accuracy_pct']}% |")
        print(f"| Confusion Matrix (TP, FP, TN, FN) | {tm['confusion_matrix']} | {em['confusion_matrix']} |")
        print(f"| Mean Squared Error (MSE 1-Day) | {tm['mse_1d']} | {em['mse_1d']} |")
        print(f"| Mean Absolute Error (MAE 1-Day) | {tm['mae_1d']} | {em['mae_1d']} |")
        print(f"| R² Score (1-Day Return) | {tm['r2_score']} | {em['r2_score']} |")
        print(f"| Information Coefficient (IC) | {tm['information_coefficient']} | {em['information_coefficient']} |")
        print()
        
        print("#### 2. Trading Strategy Backtest (Realized 1-Day Step with 15 bps Costs)")
        print("| Strategy Metric | Strategy Net (Test) | Strategy Gross (Test) | Buy & Hold Benchmark |")
        print("| :--- | :--- | :--- | :--- |")
        bnh = em["benchmark_buy_and_hold"]
        print(f"| Cumulative Return | **{em['net_cumulative_return_pct']}%** | {em['gross_cumulative_return_pct']}% | {bnh['cumulative_return_pct']}% |")
        print(f"| Annualized Return | {em['net_annualized_return_pct']}% | {em['gross_annualized_return_pct']}% | {bnh['annualized_return_pct']}% |")
        print(f"| Annualized Sharpe Ratio | **{em['net_sharpe_ratio']}** | {em['gross_sharpe_ratio']} | {bnh['sharpe_ratio']} |")
        print(f"| Sortino Ratio | {em['net_sortino_ratio']} | - | - |")
        print(f"| Maximum Drawdown | {em['net_max_drawdown_pct']}% | {em['gross_max_drawdown_pct']}% | {bnh['max_drawdown_pct']}% |")
        print(f"| Drawdown Duration | {em['max_drawdown_duration_days']} days | - | - |")
        print(f"| Win Rate | {em['win_rate_pct']}% | - | - |")
        print(f"| Profit Factor | {em['profit_factor']} | - | - |")
        print(f"| Number of Trades | {em['num_trades']} | - | - |")
        print(f"| Total Transaction Costs Paid | {em['total_transaction_costs_pct']}% | 0.0% | 0.0% |")


if __name__ == "__main__":
    main()
