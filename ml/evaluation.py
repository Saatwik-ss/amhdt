"""
Evaluation metrics and trading backtesting engine for Dependency-Aware Trading System.

Computes:
1. Regression metrics (MSE, RMSE, MAE, Pearson IC, R^2)
2. Directional classification metrics (Accuracy, Precision, Recall, F1, Balanced Acc, Confusion Matrix)
3. Financial backtesting metrics (Gross/Net Cumulative Return, Annualized Return, Sharpe, Sortino, Max Drawdown)
4. Trading diagnostics (Win Rate, Profit Factor, Turnover, Trade Count, Avg Win/Loss)
5. Benchmark comparisons (Buy & Hold Benchmark, Naive Momentum Baseline)
"""

from typing import Dict, Any, Optional
import numpy as np
import torch


def compute_financial_metrics(
    returns: np.ndarray,
    weights_diff: Optional[np.ndarray] = None,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> Dict[str, float]:
    """
    Compute trading financial performance metrics with strict compounding and costs.
    
    Args:
        returns: (num_steps,) realized 1-day strategy returns
        weights_diff: (num_steps,) or (num_steps, N) absolute change in position weights
        transaction_cost_bps: Brokerage/fee per basis point (default 10 bps = 0.10%)
        slippage_bps: Execution slippage per basis point (default 5 bps = 0.05%)
        
    Returns:
        Dict of financial performance and risk metrics
    """
    n_steps = len(returns)
    if n_steps == 0:
        return {
            "cumulative_return": 0.0,
            "annualized_return": 0.0,
            "volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_duration": 0,
            "turnover": 0.0,
            "total_transaction_costs": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "num_trades": 0,
            "avg_win": 0.0,
            "avg_loss": 0.0,
            "win_loss_ratio": 0.0,
        }
        
    ann_factor = 252.0
    
    # 1. Turnover and transaction costs
    turnover = 0.0
    total_costs = 0.0
    net_returns = returns.copy()
    num_trades = 0
    
    if weights_diff is not None:
        flat_diff = np.sum(np.abs(weights_diff), axis=-1) if weights_diff.ndim > 1 else np.abs(weights_diff)
        turnover = float(np.mean(flat_diff))
        total_fee_bps = transaction_cost_bps + slippage_bps
        cost_per_step = flat_diff * (total_fee_bps / 10000.0)
        net_returns = returns - cost_per_step
        total_costs = float(np.sum(cost_per_step))
        num_trades = int(np.sum(flat_diff > 1e-4))
        
    # 2. Cumulative and Annualized Return (Geometric CAGR)
    # Clip returns at -0.999 to prevent log-domain errors on extreme moves
    safe_returns = np.maximum(net_returns, -0.999)
    equity = np.cumprod(1.0 + safe_returns)
    total_return = float(equity[-1] - 1.0) if len(equity) > 0 else 0.0
    
    if n_steps > 0 and equity[-1] > 0:
        ann_return = float((equity[-1] ** (ann_factor / n_steps)) - 1.0)
    else:
        ann_return = float(np.mean(safe_returns) * ann_factor)
        
    # 3. Volatility and Risk-Adjusted Ratios
    mean_ret = float(np.mean(safe_returns))
    std_ret = float(np.std(safe_returns))
    ann_vol = float(std_ret * np.sqrt(ann_factor))
    
    sharpe = float((mean_ret / (std_ret + 1e-8)) * np.sqrt(ann_factor))
    
    downside = safe_returns[safe_returns < 0]
    if len(downside) > 0:
        downside_std = float(np.std(downside) * np.sqrt(ann_factor))
        sortino = float((mean_ret * ann_factor) / (downside_std + 1e-8))
    else:
        sortino = float(sharpe)
        
    # 4. Maximum Drawdown & Underwater Duration
    running_max = np.maximum.accumulate(equity)
    drawdowns = (equity - running_max) / (running_max + 1e-8)
    max_dd = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    # Calculate longest drawdown duration
    dd_duration = 0
    current_dd_len = 0
    for dd in drawdowns:
        if dd < 0:
            current_dd_len += 1
            if current_dd_len > dd_duration:
                dd_duration = current_dd_len
        else:
            current_dd_len = 0
            
    # 5. Trade Diagnostics: Win Rate, Profit Factor, Avg Win/Loss
    wins = safe_returns[safe_returns > 0]
    losses = safe_returns[safe_returns < 0]
    
    win_rate = float(len(wins) / n_steps) if n_steps > 0 else 0.0
    sum_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    sum_losses = float(np.sum(np.abs(losses))) if len(losses) > 0 else 0.0
    profit_factor = float(sum_wins / (sum_losses + 1e-8))
    
    avg_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    avg_loss = float(np.mean(np.abs(losses))) if len(losses) > 0 else 0.0
    win_loss_ratio = float(avg_win / (avg_loss + 1e-8))
    
    return {
        "cumulative_return": total_return,
        "annualized_return": ann_return,
        "volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": max_dd,
        "max_drawdown_duration": int(dd_duration),
        "turnover": turnover,
        "total_transaction_costs": total_costs,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "num_trades": num_trades,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "win_loss_ratio": win_loss_ratio,
    }


def compute_classification_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute directional classification metrics (UP vs DOWN).
    
    Args:
        preds: Continuous or directional predictions
        targets: Actual returns
        
    Returns:
        Dict containing Accuracy, Precision, Recall, F1, Balanced Acc, Confusion Matrix
    """
    pred_dir = preds > 0
    true_dir = targets > 0
    
    tp = int(np.sum(pred_dir & true_dir))
    fp = int(np.sum(pred_dir & ~true_dir))
    tn = int(np.sum(~pred_dir & ~true_dir))
    fn = int(np.sum(~pred_dir & true_dir))
    total = len(preds)
    
    accuracy = float((tp + tn) / total) if total > 0 else 0.0
    prec_up = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec_up = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    prec_down = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    rec_down = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    f1 = float(2 * prec_up * rec_up / (prec_up + rec_up + 1e-8))
    balanced_acc = float(0.5 * (rec_up + rec_down))
    
    return {
        "directional_accuracy": accuracy,
        "precision_up": prec_up,
        "recall_up": rec_up,
        "precision_down": prec_down,
        "recall_down": rec_down,
        "f1_score": f1,
        "balanced_accuracy": balanced_acc,
        "confusion_matrix": {
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
        },
    }


def evaluate_predictions(
    outputs: Dict[str, Any],
    targets: torch.Tensor,
    output_mode: str,
    transaction_cost_bps: float = 10.0,
    slippage_bps: float = 5.0,
) -> Dict[str, Any]:
    """
    Comprehensive evaluation of predicted outputs against targets.
    
    Args:
        outputs: Dictionary returned by pipeline forward pass
        targets: (batch, N, horizon) future returns
        output_mode: "portfolio_weights" or "forecasting"
        transaction_cost_bps: Brokerage cost per basis point
        slippage_bps: Execution slippage per basis point
        
    Returns:
        Comprehensive dict of regression, classification, and backtest metrics
    """
    metrics = {}
    targets_np = targets.detach().cpu().numpy()
    batch_size = targets_np.shape[0]
    N = targets_np.shape[1]
    horizon = targets_np.shape[2]
    
    if output_mode == "portfolio_weights":
        weights = outputs["output"].weights.detach().cpu().numpy()
        T = weights.shape[1]
        weights = weights.reshape(batch_size, N, T, N)
        pred_weights = weights[:, :, -1, :]  # (batch, N, N)
        
        # Realized 1-day step return for daily rebalancing
        target_1d = targets_np[:, :, 0]  # (batch, N)
        
        portfolio_returns = np.zeros((batch_size, N))
        for b in range(batch_size):
            portfolio_returns[b] = np.dot(pred_weights[b], target_1d[b])
            
        step_returns = portfolio_returns.mean(axis=1)  # (batch,)
        
        # Turnover
        if N == 1:
            pw_flat = pred_weights.reshape(batch_size)
            weights_diff = np.diff(pw_flat, prepend=0)
            directional_accuracy = float(np.mean(np.sign(pw_flat) == np.sign(target_1d.reshape(batch_size))))
        else:
            weights_diff = np.diff(pred_weights.mean(axis=1), axis=0, prepend=0)
            directional_accuracy = float(np.mean(np.sign(pred_weights.mean(axis=1)) == np.sign(target_1d)))
            
        financial_metrics = compute_financial_metrics(
            step_returns,
            weights_diff=weights_diff,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        metrics.update(financial_metrics)
        metrics["directional_accuracy"] = directional_accuracy
        metrics["mean_portfolio_return"] = float(np.mean(step_returns))
        
    else:  # forecasting
        forecasts = outputs["output"].forecasts.detach().cpu().numpy()
        T = forecasts.shape[1]
        forecasts = forecasts.reshape(batch_size, N, T, N, horizon)
        
        # Forecasts at the last timestep T-1
        last_forecasts = forecasts[:, :, -1, :, :]  # (batch, N, N, horizon)
        
        # Extract self-predictions
        if N == 1:
            diag_forecasts = last_forecasts[:, 0, 0, :].reshape(batch_size, 1, horizon)
        else:
            diag_forecasts = np.zeros((batch_size, N, horizon))
            for b in range(batch_size):
                for i in range(N):
                    diag_forecasts[b, i, :] = last_forecasts[b, i, i, :]
                    
        # Targets & predictions for 1-day return
        pred_1d = diag_forecasts[:, :, 0].reshape(-1)
        true_1d = targets_np[:, :, 0].reshape(-1)
        
        # 1. Regression Metrics (Continuous Returns)
        mse_1d = float(np.mean((pred_1d - true_1d) ** 2))
        mae_1d = float(np.mean(np.abs(pred_1d - true_1d)))
        rmse_1d = float(np.sqrt(mse_1d))
        
        mse_overall = float(np.mean((diag_forecasts - targets_np) ** 2))
        mae_overall = float(np.mean(np.abs(diag_forecasts - targets_np)))
        
        # Pearson Correlation / Information Coefficient (IC)
        if np.std(pred_1d) > 1e-8 and np.std(true_1d) > 1e-8:
            ic_corr = float(np.corrcoef(pred_1d, true_1d)[0, 1])
        else:
            ic_corr = 0.0
            
        # R^2 score for 1-day returns
        ss_res = np.sum((true_1d - pred_1d) ** 2)
        ss_tot = np.sum((true_1d - np.mean(true_1d)) ** 2)
        r2_1d = float(1.0 - (ss_res / (ss_tot + 1e-8)))
        
        metrics.update({
            "mse": mse_1d,
            "mae": mae_1d,
            "rmse": rmse_1d,
            "mse_overall": mse_overall,
            "mae_overall": mae_overall,
            "information_coefficient": ic_corr,
            "r2_score": r2_1d,
        })
        
        # 2. Classification Metrics (UP / DOWN direction)
        class_metrics = compute_classification_metrics(pred_1d, true_1d)
        metrics.update(class_metrics)
        
        # 3. Strategy Backtesting with Daily Realized Returns & Costs
        # Strategy position: Long (+1) or Short (-1) based on predicted return
        trade_positions = np.sign(pred_1d)
        
        # Calculate daily position difference for turnover
        pos_diff = np.diff(trade_positions, prepend=0)
        
        # Daily gross returns
        gross_returns = trade_positions * true_1d
        
        # Strategy Net Performance
        strat_net_metrics = compute_financial_metrics(
            gross_returns,
            weights_diff=pos_diff,
            transaction_cost_bps=transaction_cost_bps,
            slippage_bps=slippage_bps,
        )
        metrics.update(strat_net_metrics)
        
        # Strategy Gross Performance (before costs)
        strat_gross_metrics = compute_financial_metrics(
            gross_returns,
            weights_diff=None,
        )
        metrics["gross_cumulative_return"] = strat_gross_metrics["cumulative_return"]
        metrics["gross_annualized_return"] = strat_gross_metrics["annualized_return"]
        metrics["gross_sharpe_ratio"] = strat_gross_metrics["sharpe_ratio"]
        metrics["gross_max_drawdown"] = strat_gross_metrics["max_drawdown"]
        
        # 4. Benchmark: Buy & Hold (holding index 100% long)
        bnh_metrics = compute_financial_metrics(
            true_1d,
            weights_diff=None,
        )
        metrics["benchmark_buy_and_hold"] = {
            "cumulative_return": bnh_metrics["cumulative_return"],
            "annualized_return": bnh_metrics["annualized_return"],
            "sharpe_ratio": bnh_metrics["sharpe_ratio"],
            "sortino_ratio": bnh_metrics["sortino_ratio"],
            "max_drawdown": bnh_metrics["max_drawdown"],
            "volatility": bnh_metrics["volatility"],
        }
        
        # 5. Benchmark: Naive Momentum (following previous day's direction)
        if len(true_1d) > 1:
            mom_positions = np.sign(np.roll(true_1d, 1))
            mom_positions[0] = 1.0  # start long
            mom_returns = mom_positions * true_1d
            mom_diff = np.diff(mom_positions, prepend=0)
            mom_metrics = compute_financial_metrics(
                mom_returns,
                weights_diff=mom_diff,
                transaction_cost_bps=transaction_cost_bps,
                slippage_bps=slippage_bps,
            )
            metrics["benchmark_momentum"] = {
                "cumulative_return": mom_metrics["cumulative_return"],
                "annualized_return": mom_metrics["annualized_return"],
                "sharpe_ratio": mom_metrics["sharpe_ratio"],
                "max_drawdown": mom_metrics["max_drawdown"],
            }
            
    return metrics
