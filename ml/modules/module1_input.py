"""
Module 1: Stock Data Input

Collect and align multi-asset data into cohesive tensors.
Input: OHLCV, Micro/Macro indicators, Sentiment/News, Fundamentals
Output: X ∈ R^(N×T×F), M ∈ {0,1}^(N×T)

Handles survivorship bias, IPOs, delistings via validity masks.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import pandas as pd
from pathlib import Path


@dataclass
class MarketTensor:
    """Clean structure for aligned market data with metadata."""
    
    features: torch.Tensor          # Shape: (N, T, F) - asset, time, feature
    validity_mask: torch.Tensor     # Shape: (N, T) - binary mask for IPO/delist
    asset_ids: List[str]            # Length: N - ticker symbols
    timestamps: np.ndarray          # Length: T - trading dates
    feature_names: List[str]        # Length: F - feature names
    feature_means: torch.Tensor     # Shape: (F,) - for normalization
    feature_stds: torch.Tensor      # Shape: (F,) - for normalization
    raw_prices: Optional[torch.Tensor] = None  # Shape: (N, T) - unnormalized Close prices
    raw_features: Optional[torch.Tensor] = None  # Shape: (N, T, F) - unnormalized raw features
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Return shape as (N, T, F)."""
        return self.features.shape
    
    @property
    def num_assets(self) -> int:
        """Number of assets."""
        return self.features.shape[0]
    
    @property
    def lookback_window(self) -> int:
        """Lookback window length."""
        return self.features.shape[1]
    
    @property
    def num_features(self) -> int:
        """Number of features."""
        return self.features.shape[2]
    
    def to_device(self, device: torch.device) -> "MarketTensor":
        """Move tensors to device."""
        return MarketTensor(
            features=self.features.to(device),
            validity_mask=self.validity_mask.to(device),
            asset_ids=self.asset_ids,
            timestamps=self.timestamps,
            feature_names=self.feature_names,
            feature_means=self.feature_means.to(device),
            feature_stds=self.feature_stds.to(device),
            raw_prices=self.raw_prices.to(device) if self.raw_prices is not None else None,
            raw_features=self.raw_features.to(device) if self.raw_features is not None else None,
        )


class StockDataProcessor:
    """Process raw stock data into aligned tensors."""
    
    # Standard OHLCV columns
    OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]
    
    # Nifty extra market microstructure columns
    NIFTY_EXTRA_COLS = ["VWAP", "Turnover", "Trades", "Deliverable Volume", "%Deliverble"]
    
    # Micro indicators (asset-specific)
    MICRO_INDICATORS = [
        "RSI_14", "MACD", "MACD_Signal", "BBand_Upper", "BBand_Middle", "BBand_Lower"
    ]
    
    # Macro indicators (systemic)
    MACRO_INDICATORS = [
        "VIX", "DXY", "TNX", "MOVE", "SKEW"
    ]
    
    # Sentiment & Fundamentals
    SENTIMENT_COLS = ["Sentiment_Score"]
    FUNDAMENTAL_COLS = ["PE_Ratio", "Dividend_Yield", "Market_Cap"]
    
    def __init__(
        self,
        lookback_window: int = 60,
        normalization: str = "zscore",
        fill_method: str = "forward",
        device: torch.device = None,
    ):
        """
        Initialize processor.
        
        Args:
            lookback_window: Number of trading days to look back
            normalization: "zscore" or "minmax"
            fill_method: "forward" for forward-fill, "drop" to skip
            device: torch device for tensors
        """
        self.lookback_window = lookback_window
        self.normalization = normalization
        self.fill_method = fill_method
        self.device = device or torch.device("cpu")
        
    def process(
        self,
        df: pd.DataFrame,
        asset_col: str = "Ticker",
        date_col: str = "Date",
        ohlcv_only: bool = False,
    ) -> MarketTensor:
        """
        Process raw DataFrame into aligned MarketTensor.
        
        Args:
            df: DataFrame with columns [Date, Ticker, OHLCV, indicators, ...]
            asset_col: Column name for asset ticker
            date_col: Column name for trading date
            ohlcv_only: If True, only use OHLCV columns (Open, High, Low, Close, Volume)
            
        Returns:
            MarketTensor with X, M, metadata
        """
        df = df.copy()
        
        # Auto-detect date column
        if date_col not in df.columns:
            for candidate in ["DateTime", "Date", "date", "datetime", "timestamp", "Timestamp"]:
                if candidate in df.columns:
                    date_col = candidate
                    break
            else:
                raise ValueError(f"Could not find date column. Tried candidates and specified: {date_col}")
        
        # Auto-detect or default asset column
        if asset_col not in df.columns:
            for candidate in ["Ticker", "Symbol", "ticker", "symbol", "Asset", "asset"]:
                if candidate in df.columns:
                    asset_col = candidate
                    break
            else:
                # Single-asset index data
                asset_col = "Ticker"
                df[asset_col] = "INDEX"
                
        # Handle zero Volume fallback to TickVolume
        if "Volume" in df.columns and "TickVolume" in df.columns:
            vol_sum = df["Volume"].fillna(0).abs().sum()
            tick_sum = df["TickVolume"].fillna(0).abs().sum()
            if vol_sum == 0 and tick_sum > 0:
                df["Volume"] = df["TickVolume"]
        elif "Volume" not in df.columns and "TickVolume" in df.columns:
            df["Volume"] = df["TickVolume"]
        
        # Parse dates and sort chronologically (oldest to newest)
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(by=date_col).reset_index(drop=True)
        
        # Get unique assets and dates
        assets = sorted(df[asset_col].unique())
        dates = sorted(df[date_col].unique())
        
        # Select feature columns
        feature_cols = self._select_features(df.columns, ohlcv_only=ohlcv_only)
        
        # Create aligned tensor (N, T, F)
        N = len(assets)
        T = len(dates)
        F = len(feature_cols)
        
        X = np.full((N, T, F), np.nan, dtype=np.float32)
        raw_prices = np.full((N, T), np.nan, dtype=np.float32)
        M = np.zeros((N, T), dtype=np.int32)  # Validity mask
        
        # Find Close column index if present
        close_col = "Close" if "Close" in df.columns else None
        
        # Fill tensor
        for i, asset in enumerate(assets):
            asset_df = df[df[asset_col] == asset].set_index(date_col)
            
            for j, date in enumerate(dates):
                if date in asset_df.index:
                    row = asset_df.loc[date]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    # Get feature values
                    for f, col in enumerate(feature_cols):
                        if col in row.index:
                            val = row[col]
                            if pd.notna(val):
                                X[i, j, f] = float(val)
                                M[i, j] = 1  # Valid
                    if close_col and close_col in row.index and pd.notna(row[close_col]):
                        raw_prices[i, j] = float(row[close_col])
        
        # Handle missing data
        X = self._fill_missing(X, M)
        
        # If raw_prices has NaNs, forward-fill them
        if np.isnan(raw_prices).any():
            for i in range(N):
                for j in range(1, T):
                    if np.isnan(raw_prices[i, j]) and not np.isnan(raw_prices[i, j-1]):
                        raw_prices[i, j] = raw_prices[i, j-1]
            # Any remaining NaNs fill with nanmean
            mean_p = np.nanmean(raw_prices)
            raw_prices = np.where(np.isnan(raw_prices), mean_p if not np.isnan(mean_p) else 100.0, raw_prices)
        
        # Normalize
        feature_means, feature_stds = self._compute_stats(X, M)
        X_norm = self._normalize(X, feature_means, feature_stds)
        
        # Convert to tensors
        X_tensor = torch.from_numpy(X_norm).to(self.device)
        M_tensor = torch.from_numpy(M).to(self.device)
        raw_prices_tensor = torch.from_numpy(raw_prices).to(self.device)
        feature_means_tensor = torch.from_numpy(feature_means).to(self.device)
        feature_stds_tensor = torch.from_numpy(feature_stds).to(self.device)
        
        return MarketTensor(
            features=X_tensor,
            validity_mask=M_tensor,
            asset_ids=assets,
            timestamps=np.array(dates),
            feature_names=feature_cols,
            feature_means=feature_means_tensor,
            feature_stds=feature_stds_tensor,
            raw_prices=raw_prices_tensor,
            raw_features=torch.from_numpy(X).to(self.device),
        )
    
    def create_sequences(
        self,
        market_tensor: MarketTensor,
        lookback: int = 60,
        horizon: int = 5,
        stride: int = 1,
        stationary_features: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create sliding window sequences for training/evaluation with strict zero lookahead.
        
        Args:
            market_tensor: MarketTensor containing features, validity_mask, and raw_prices
            lookback: Lookback window length (T)
            horizon: Forecast horizon (H)
            stride: Step size between consecutive sequences
            stationary_features: If True and OHLCV available, transforms raw prices into
                                scale-invariant stationary indicators (returns, spreads, body)
            
        Returns:
            X: (num_seq, N, lookback, F)
            M: (num_seq, N, lookback)
            Y: (num_seq, N, horizon)
        """
        X = market_tensor.features.cpu().numpy()  # (N, T, F)
        M = market_tensor.validity_mask.cpu().numpy()  # (N, T)
        N, T, F = X.shape
        
        # Check raw prices
        if market_tensor.raw_prices is not None:
            prices = market_tensor.raw_prices.cpu().numpy()  # (N, T)
        else:
            close_idx = 3 if F > 3 else 0
            prices = X[:, :, close_idx]
            
        # Check raw unnormalized features for stationary calculation
        raw_feat = market_tensor.raw_features.cpu().numpy() if market_tensor.raw_features is not None else None
        has_ohlcv = (
            raw_feat is not None
            and "Open" in market_tensor.feature_names
            and "High" in market_tensor.feature_names
            and "Low" in market_tensor.feature_names
            and "Close" in market_tensor.feature_names
        )
        
        if has_ohlcv:
            o_idx = market_tensor.feature_names.index("Open")
            h_idx = market_tensor.feature_names.index("High")
            l_idx = market_tensor.feature_names.index("Low")
            c_idx = market_tensor.feature_names.index("Close")
            v_idx = market_tensor.feature_names.index("Volume") if "Volume" in market_tensor.feature_names else -1
            # Nifty extra columns (optional — absent for Dow30)
            vwap_idx   = market_tensor.feature_names.index("VWAP")              if "VWAP"               in market_tensor.feature_names else -1
            turn_idx   = market_tensor.feature_names.index("Turnover")          if "Turnover"           in market_tensor.feature_names else -1
            trade_idx  = market_tensor.feature_names.index("Trades")            if "Trades"             in market_tensor.feature_names else -1
            delv_idx   = market_tensor.feature_names.index("Deliverable Volume") if "Deliverable Volume" in market_tensor.feature_names else -1
            pct_idx    = market_tensor.feature_names.index("%Deliverble")       if "%Deliverble"        in market_tensor.feature_names else -1
            has_nifty_extras = (vwap_idx >= 0 and turn_idx >= 0)

        sequences_X = []
        sequences_M = []
        sequences_Y = []
        
        for start in range(0, T - lookback - horizon + 1, stride):
            end = start + lookback
            seq_M = M[:, start:end]      # (N, lookback)
            
            if stationary_features and has_ohlcv:
                # Compute stationary scale-invariant representations strictly from this window
                # Shape for window: (N, lookback, F)
                win_open  = raw_feat[:, start:end, o_idx]
                win_high  = raw_feat[:, start:end, h_idx]
                win_low   = raw_feat[:, start:end, l_idx]
                win_close = raw_feat[:, start:end, c_idx]
                win_vol   = raw_feat[:, start:end, v_idx] if v_idx >= 0 else np.ones_like(win_close)
                
                # 1. 1-day return: (Close_t - Close_{t-1}) / Close_{t-1}
                # For first step in window, use (Close_0 - Open_0) / Open_0
                win_ret = np.zeros_like(win_close)
                win_ret[:, 0] = (win_close[:, 0] - win_open[:, 0]) / np.maximum(np.abs(win_open[:, 0]), 1e-6)
                win_ret[:, 1:] = (win_close[:, 1:] - win_close[:, :-1]) / np.maximum(np.abs(win_close[:, :-1]), 1e-6)
                
                # 2. Normalized range: (High - Low) / Close
                win_hl = (win_high - win_low) / np.maximum(np.abs(win_close), 1e-6)
                
                # 3. Normalized candle body: (Close - Open) / Open
                win_co = (win_close - win_open) / np.maximum(np.abs(win_open), 1e-6)
                
                # 4. Upper shadow: (High - max(Open, Close)) / Close
                win_max_oc = np.maximum(win_open, win_close)
                win_shadow = (win_high - win_max_oc) / np.maximum(np.abs(win_close), 1e-6)
                
                # 5. Volume z-score normalized strictly within window
                vol_mean = np.mean(win_vol, axis=1, keepdims=True)
                vol_std  = np.std(win_vol,  axis=1, keepdims=True) + 1e-6
                win_vol_norm = (win_vol - vol_mean) / vol_std

                base_features = [win_ret, win_hl, win_co, win_shadow, win_vol_norm]

                if has_nifty_extras:
                    # 6. VWAP deviation: (Close - VWAP) / VWAP  — mean-reversion signal
                    win_vwap = raw_feat[:, start:end, vwap_idx]
                    win_vwap_dev = (win_close - win_vwap) / np.maximum(np.abs(win_vwap), 1e-6)

                    # 7. Turnover log-return + within-window z-score  — activity momentum
                    win_turn = raw_feat[:, start:end, turn_idx]
                    win_turn = np.maximum(win_turn, 1e-6)           # prevent log(0)
                    turn_logret = np.zeros_like(win_turn)
                    turn_logret[:, 1:] = np.log(win_turn[:, 1:] / win_turn[:, :-1])
                    tr_mean = np.mean(turn_logret, axis=1, keepdims=True)
                    tr_std  = np.std(turn_logret,  axis=1, keepdims=True) + 1e-6
                    win_turn_norm = (turn_logret - tr_mean) / tr_std

                    # 8. Trade count z-score within window  — liquidity signal
                    if trade_idx >= 0:
                        win_trades = raw_feat[:, start:end, trade_idx]
                    else:
                        win_trades = np.ones_like(win_close)
                    trd_mean = np.mean(win_trades, axis=1, keepdims=True)
                    trd_std  = np.std(win_trades,  axis=1, keepdims=True) + 1e-6
                    win_trades_norm = (win_trades - trd_mean) / trd_std

                    # 9. Deliverable ratio: Deliverable Volume / Volume  — conviction signal
                    if delv_idx >= 0:
                        win_delv = raw_feat[:, start:end, delv_idx]
                        win_delv_ratio = np.clip(win_delv / np.maximum(win_vol, 1e-6), 0.0, 1.0)
                    else:
                        win_delv_ratio = np.full_like(win_close, 0.5)

                    # 10. %Deliverable z-score within window  — institutional activity
                    if pct_idx >= 0:
                        win_pct = raw_feat[:, start:end, pct_idx]
                    else:
                        win_pct = np.full_like(win_close, 50.0)
                    pct_mean = np.mean(win_pct, axis=1, keepdims=True)
                    pct_std  = np.std(win_pct,  axis=1, keepdims=True) + 1e-6
                    win_pct_norm = (win_pct - pct_mean) / pct_std

                    extra_features = [win_vwap_dev, win_turn_norm, win_trades_norm, win_delv_ratio, win_pct_norm]
                    seq_X = np.stack(base_features + extra_features, axis=2)  # (N, lookback, 10)
                else:
                    seq_X = np.stack(base_features, axis=2)  # (N, lookback, 5)
            else:
                seq_X = X[:, start:end, :].copy()  # (N, lookback, F)
                # Local window z-score to prevent lookahead from global normalization
                for i in range(N):
                    valid = seq_M[i] == 1
                    if valid.sum() > 1:
                        w_mean = np.mean(seq_X[i, valid, :], axis=0, keepdims=True)
                        w_std = np.std(seq_X[i, valid, :], axis=0, keepdims=True) + 1e-6
                        seq_X[i] = (seq_X[i] - w_mean) / w_std
            
            # Future returns: reference price is the LAST observed step in the lookback window
            current_prices = prices[:, end - 1 : end]  # (N, 1)
            future_prices = prices[:, end : end + horizon]  # (N, horizon)
            
            # Compute percentage returns: (future_price - current_price) / current_price
            safe_current = np.where(np.abs(current_prices) < 1e-6, 1.0, current_prices)
            returns = (future_prices / safe_current) - 1.0  # (N, horizon)
            
            sequences_X.append(seq_X)
            sequences_M.append(seq_M)
            sequences_Y.append(returns)
            
        if len(sequences_X) == 0:
            raise ValueError(f"Total time steps {T} is insufficient for lookback {lookback} + horizon {horizon}")
            
        X_tensor = torch.from_numpy(np.stack(sequences_X, axis=0)).float()
        M_tensor = torch.from_numpy(np.stack(sequences_M, axis=0)).float()
        Y_tensor = torch.from_numpy(np.stack(sequences_Y, axis=0)).float()
        
        return X_tensor, M_tensor, Y_tensor

    def _select_features(self, columns, ohlcv_only: bool = False) -> List[str]:
        """Select relevant feature columns from DataFrame."""
        selected = []
        
        for col in self.OHLCV_COLS:
            if col in columns:
                selected.append(col)
        
        if ohlcv_only:
            if not selected:
                raise ValueError("No OHLCV columns found in DataFrame")
            return selected
        
        # Nifty market microstructure extras (included whenever present)
        for col in self.NIFTY_EXTRA_COLS:
            if col in columns:
                selected.append(col)
        
        for col in self.MICRO_INDICATORS:
            if col in columns:
                selected.append(col)
        
        for col in self.MACRO_INDICATORS:
            if col in columns:
                selected.append(col)
        
        for col in self.SENTIMENT_COLS:
            if col in columns:
                selected.append(col)
        
        for col in self.FUNDAMENTAL_COLS:
            if col in columns:
                selected.append(col)
        
        if not selected:
            raise ValueError("No valid features found in DataFrame")
        
        return selected
    
    def _fill_missing(self, X: np.ndarray, M: np.ndarray) -> np.ndarray:
        """Fill missing values."""
        X = X.copy()
        N, T, F = X.shape
        
        if self.fill_method == "forward":
            # Forward-fill per asset per feature
            for i in range(N):
                for f in range(F):
                    for t in range(1, T):
                        if np.isnan(X[i, t, f]) and not np.isnan(X[i, t-1, f]):
                            X[i, t, f] = X[i, t-1, f]
            
            # If still NaN, use feature mean
            for f in range(F):
                mean_val = np.nanmean(X[:, :, f])
                X[:, :, f] = np.where(np.isnan(X[:, :, f]), mean_val, X[:, :, f])
        
        return X
    
    def _compute_stats(
        self,
        X: np.ndarray,
        M: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute mean/std only on valid data."""
        F = X.shape[2]
        means = np.zeros(F)
        stds = np.ones(F)
        
        for f in range(F):
            valid_data = X[:, :, f][M == 1]
            if len(valid_data) > 0:
                means[f] = np.nanmean(valid_data)
                stds[f] = np.nanstd(valid_data)
                if stds[f] < 1e-6:
                    stds[f] = 1.0  # Prevent division by zero
        
        return means.astype(np.float32), stds.astype(np.float32)
    
    def _normalize(
        self,
        X: np.ndarray,
        means: np.ndarray,
        stds: np.ndarray,
    ) -> np.ndarray:
        """Normalize features."""
        if self.normalization == "zscore":
            return (X - means) / stds
        elif self.normalization == "minmax":
            # Simplified minmax
            return (X - np.min(X, axis=(0, 1), keepdims=True)) / \
                   (np.max(X, axis=(0, 1), keepdims=True) - np.min(X, axis=(0, 1), keepdims=True) + 1e-8)
        else:
            return X


class DataValidator:
    """Validate data quality and detect issues."""
    
    @staticmethod
    def validate_market_tensor(mt: MarketTensor) -> Dict[str, any]:
        """
        Validate MarketTensor and return report.
        
        Returns:
            {
                "valid": bool,
                "issues": List[str],
                "coverage": Dict[str, float],
                "stats": Dict
            }
        """
        issues = []
        X = mt.features.numpy() if isinstance(mt.features, torch.Tensor) else mt.features
        M = mt.validity_mask.numpy() if isinstance(mt.validity_mask, torch.Tensor) else mt.validity_mask
        
        N, T, F = mt.shape
        
        # Check NaN
        nan_count = np.isnan(X).sum()
        if nan_count > 0:
            issues.append(f"Found {nan_count} NaN values")
        
        # Check coverage per asset
        coverage_per_asset = M.sum(axis=1) / T
        if (coverage_per_asset < 0.5).any():
            sparse_assets = [
                mt.asset_ids[i] for i in range(N) if coverage_per_asset[i] < 0.5
            ]
            issues.append(f"Low coverage (<50%): {sparse_assets}")
        
        # Check temporal continuity
        dates_diff = np.diff(mt.timestamps.astype('datetime64[D]'))
        if (dates_diff > np.timedelta64(5, 'D')).any():
            issues.append("Large gaps in time series (>5 days)")
        
        # Check for constant features
        for f, fname in enumerate(mt.feature_names):
            if np.nanstd(X[:, :, f]) < 1e-6:
                issues.append(f"Constant feature: {fname}")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "coverage": {
                "overall": float(M.sum() / (N * T)),
                "per_asset": {
                    asset: float(M[i].sum() / T)
                    for i, asset in enumerate(mt.asset_ids)
                }
            },
            "shape": {
                "num_assets": N,
                "lookback_window": T,
                "num_features": F
            },
            "date_range": {
                "start": str(mt.timestamps[0]),
                "end": str(mt.timestamps[-1])
            }
        }
