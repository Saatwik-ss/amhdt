import sys
import time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pandas as pd
import numpy as np
import torch

from ml.modules.module1_input import StockDataProcessor
from ml.pipeline import DependencyAwareTradingPipeline, PipelineConfig
from ml.training import train_pipeline_from_config
from ml.evaluation import evaluate_predictions, compute_classification_metrics, compute_financial_metrics

# Load Batch 1 stocks
stocks = ['TATASTEEL', 'BPCL', 'ASIANPAINT', 'HDFCBANK', 'HCLTECH', 'GRASIM', 'BRITANNIA', 'BHARTIARTL', 'POWERGRID', 'BAJFINANCE']
csv_paths = [Path(f'ml/data/nifty/{s}.csv') for s in stocks]
frames = [pd.read_csv(p) for p in csv_paths if p.exists()]
df = pd.concat(frames, ignore_index=True)

lookback = 30
horizon = 5

processor = StockDataProcessor(lookback_window=lookback, device=torch.device('cpu'))
mt = processor.process(df, ohlcv_only=False)
N, T_total, F_raw = mt.shape
print(f"MarketTensor: N={N}, T={T_total}, F={F_raw}")

X, M, Y = processor.create_sequences(mt, lookback=lookback, horizon=horizon, stride=1, stationary_features=True)
n_seq = len(X)
n_train = int(n_seq * 0.8)

X_train, M_train, Y_train = X[:n_train], M[:n_train], Y[:n_train]
X_test, M_test, Y_test = X[n_train:], M[n_train:], Y[n_train:]

print(f"Sequences: {n_seq}, Train: {n_train}, Test: {len(X_test)}")

config = PipelineConfig(
    lookback_window=lookback,
    forecast_horizon=horizon,
    num_sectors=max(2, N // 3),
    encoder_hidden_dim=64,
    regime_embedding_dim=128,
    transformer_embedding_dim=128,
    num_transformer_layers=2,
    num_attention_heads=4,
    transformer_ffn_dim=128,
    transformer_dropout=0.1,
    encoder_dropout=0.1,
    regime_dropout=0.1,
    output_mode="forecasting",
    batch_size=32,
    num_epochs=12,
    learning_rate=1e-3,
    weight_decay=1e-5,
    early_stopping_patience=4,
    gradient_clip=1.0,
    device='cpu',
)

print("\n--- Training Model with Decomposed Temporal IC + Residual Skip ---")
t0 = time.time()
pipeline, history = train_pipeline_from_config(
    X_train, M_train, Y_train, config, save_name="nifty_b1_T30d.pt"
)
print(f"Training completed in {time.time() - t0:.1f}s")

# Check temporal variance across 100 test sequences
pipeline.eval()
print("\n--- Testing Temporal Variation Across 100 Out-of-Sample Sequences ---")
test_slice_len = min(100, len(X_test))
X_slice = X_test[:test_slice_len]
M_slice = M_test[:test_slice_len]
Y_slice = Y_test[:test_slice_len]

with torch.no_grad():
    rtg = torch.zeros(test_slice_len, N, lookback, 1)
    out = pipeline(X_slice, M_slice, rtg)
    # forecasts shape: (test_slice_len * N, lookback, N, horizon)
    fc = out['output'].forecasts.view(test_slice_len, N, lookback, N, horizon)
    
    # Extract self predictions at last step
    diag_preds = torch.zeros(test_slice_len, N, horizon)
    for b in range(test_slice_len):
        for n in range(N):
            diag_preds[b, n, :] = fc[b, n, -1, n, :]

print(f"{'Ticker':12s} | {'Time Std (H=1)':>14s} | {'Min Pred %':>10s} | {'Max Pred %':>10s} | {'UPs':>5s} | {'DOWNs':>5s}")
print("-" * 65)

for n in range(N):
    ticker = mt.asset_ids[n]
    h1_preds = diag_preds[:, n, 0].numpy() * 100.0 # percentage
    std_val = float(h1_preds.std())
    min_val = float(h1_preds.min())
    max_val = float(h1_preds.max())
    n_up = int((h1_preds > 0).sum())
    n_down = int((h1_preds <= 0).sum())
    print(f"{ticker:12s} | {std_val:14.6f} | {min_val:9.2f}% | {max_val:9.2f}% | {n_up:5d} | {n_down:5d}")

print("\nDone!")
