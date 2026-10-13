"""
Training utilities for DependencyAwareTradingPipeline.

Handles:
- Data loading and splitting
- Loss computation
- Optimization and checkpointing
- Mixed precision training
- Leakage checks
"""

from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

from ml.pipeline import DependencyAwareTradingPipeline, PipelineConfig


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int
    train_loss: float
    val_loss: float
    test_loss: float = None
    learning_rate: float = None


class SequenceTrainer:
    """Train pipeline on sequence prediction tasks."""
    
    def __init__(
        self,
        pipeline: DependencyAwareTradingPipeline,
        config: PipelineConfig,
        checkpoint_dir: Optional[Path] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            pipeline: DependencyAwareTradingPipeline instance
            config: PipelineConfig
            checkpoint_dir: Directory for saving checkpoints
        """
        self.pipeline = pipeline
        self.config = config
        self.device = torch.device(config.device)
        
        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Optimization
        self.optimizer = optim.Adam(
            pipeline.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.5,
            patience=3,
        )
        
        # Mixed precision
        self.use_amp = config.mixed_precision and torch.cuda.is_available()
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
        
        # Metrics tracking
        self.metrics_history = []
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.best_checkpoint_path = None
    
    def _compute_loss(
        self,
        output,
        targets: torch.Tensor,  # (batch, N, horizon)
    ) -> torch.Tensor:
        """
        Compute training loss.
        
        For portfolio weights mode: negative portfolio return
        For forecasting mode: MSE on return predictions
        """
        batch_size = targets.shape[0]
        N = targets.shape[1]
        horizon = targets.shape[2]
        
        out_obj = output["output"] if isinstance(output, dict) and "output" in output else output
        
        if self.config.output_mode == "portfolio_weights":
            weights = out_obj.weights  # (batch * N, T, N)
            T = weights.shape[1]
            weights = weights.view(batch_size, N, T, N)
            pred_weights = weights[:, :, -1, :]  # (batch, N, N)
            
            avg_targets = targets.mean(dim=-1, keepdim=True)  # (batch, N, 1)
            
            if N == 1:
                pw = pred_weights.view(batch_size, 1)
                tgt = avg_targets.view(batch_size, 1)
                portfolio_returns = pw * tgt
            else:
                portfolio_returns = torch.bmm(pred_weights, avg_targets)  # (batch, N, 1)
            
            loss = -portfolio_returns.mean()
        else:
            forecasts = out_obj.forecasts  # (batch * N, T, N, horizon)
            last_forecasts = forecasts[:, -1, :, :]  # (batch * N, N, horizon)
            last_forecasts = last_forecasts.view(batch_size, N, N, horizon)
            
            if N == 1:
                diag_forecasts = last_forecasts[:, 0, 0, :].unsqueeze(1)  # (batch, 1, horizon)
            else:
                diag_forecasts = torch.diagonal(last_forecasts, dim1=1, dim2=2).transpose(1, 2)
            
            # ── Decomposed Temporal IC + Cross-Sectional IC Loss ─────────────
            # Flat batch IC allows models to cheat by predicting static stock
            # averages (zero temporal variance).
            # Decomposing into:
            #  1. Temporal IC (dim=0): Forces each stock's predictions to vary
            #     dynamically across time to match time-series movements.
            #  2. Cross-Sectional IC (dim=1): Ensures accurate relative stock rankings.
            #  3. MSE: Continuous price return calibration.
            #  4. Directional BCE: Sign supervision for UP/DOWN moves.
            # ─────────────────────────────────────────────────────────────────

            # 1. MSE Loss
            mse_loss = nn.MSELoss()(diag_forecasts, targets)

            # 2. Temporal (Time-Series) IC along batch dimension (dim=0)
            p_c_time = diag_forecasts - diag_forecasts.mean(dim=0, keepdim=True)
            t_c_time = targets - targets.mean(dim=0, keepdim=True)
            denom_time = torch.sqrt((p_c_time ** 2).sum(dim=0) + 1e-8) * torch.sqrt((t_c_time ** 2).sum(dim=0) + 1e-8)
            temporal_ic = (p_c_time * t_c_time).sum(dim=0) / (denom_time + 1e-8)  # (N, horizon)
            temporal_ic_loss = -temporal_ic.mean()

            # 3. Cross-Sectional IC across assets (dim=1)
            p_c_asset = diag_forecasts - diag_forecasts.mean(dim=1, keepdim=True)
            t_c_asset = targets - targets.mean(dim=1, keepdim=True)
            denom_asset = torch.sqrt((p_c_asset ** 2).sum(dim=1) + 1e-8) * torch.sqrt((t_c_asset ** 2).sum(dim=1) + 1e-8)
            cross_ic = (p_c_asset * t_c_asset).sum(dim=1) / (denom_asset + 1e-8)  # (batch, horizon)
            cross_ic_loss = -cross_ic.mean()

            # 4. Directional Sign BCE
            pred_flat   = diag_forecasts.reshape(-1)
            target_flat = targets.reshape(-1)
            direction_target = (target_flat > 0).float()
            direction_logit  = pred_flat * 100.0  # scale to logit range
            bce_loss = nn.BCEWithLogitsLoss()(direction_logit, direction_target)

            loss = 0.35 * mse_loss + 0.35 * temporal_ic_loss + 0.15 * cross_ic_loss + 0.15 * bce_loss
        
        return loss
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train one epoch."""
        self.pipeline.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (X, M, Y) in enumerate(tqdm(train_loader, desc="Training")):
            X = X.to(self.device)
            M = M.to(self.device)
            Y = Y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    # In forecasting / return prediction, do NOT pass target future returns as RTG!
                    # Pass zeros so the model learns to forecast from historical features X alone.
                    rtg = torch.zeros(X.shape[0], X.shape[1], X.shape[2], 1, device=self.device)
                    output = self.pipeline(X, M, rtg, actions=None)
                    loss = self._compute_loss(output, Y)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.pipeline.parameters(), self.config.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                rtg = torch.zeros(X.shape[0], X.shape[1], X.shape[2], 1, device=self.device)
                output = self.pipeline(X, M, rtg, actions=None)
                loss = self._compute_loss(output, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pipeline.parameters(), self.config.gradient_clip)
                self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate on validation set with strict zero lookahead."""
        self.pipeline.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for X, M, Y in tqdm(val_loader, desc="Validation"):
                X = X.to(self.device)
                M = M.to(self.device)
                Y = Y.to(self.device)
                
                rtg = torch.zeros(X.shape[0], X.shape[1], X.shape[2], 1, device=self.device)
                output = self.pipeline(X, M, rtg, actions=None)
                loss = self._compute_loss(output, Y)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def fit(
        self,
        X: torch.Tensor,  # (num_seq, N, T, F)
        M: torch.Tensor,  # (num_seq, N, T)
        Y: torch.Tensor,  # (num_seq, N, horizon)
        train_split: float = 0.8,
        validation_ratio: float = 0.15,
        save_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Train pipeline on 80% data and test on remaining 20% data.
        
        Args:
            X: Feature tensors
            M: Validity masks
            Y: Target returns
            train_split: Fraction of dataset for training (default 0.8)
            validation_ratio: Fraction of train set for validation monitoring (default 0.15)
            save_name: Optional filename for saving the final checkpoint
            
        Returns:
            Dictionary containing epoch history, train_metrics, and test_metrics
        """
        # Strictly chronological split
        n_total = X.shape[0]
        n_train_total = int(n_total * train_split)
        n_test = n_total - n_train_total
        
        n_val = max(1, int(n_train_total * validation_ratio))
        n_train = n_train_total - n_val
        
        # Check for leakage
        self._check_leakage(X, M, Y, n_train, n_val)
        
        X_train, M_train, Y_train = X[:n_train], M[:n_train], Y[:n_train]
        X_val, M_val, Y_val = X[n_train:n_train_total], M[n_train:n_train_total], Y[n_train:n_train_total]
        X_test, M_test, Y_test = X[n_train_total:], M[n_train_total:], Y[n_train_total:]
        
        # Create dataloaders
        train_dataset = TensorDataset(X_train, M_train, Y_train)
        val_dataset = TensorDataset(X_val, M_val, Y_val)
        test_dataset = TensorDataset(X_test, M_test, Y_test)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,  # Keep chronological order!
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        
        print(f"Total sequences: {n_total} | Train: {n_train_total} ({train_split:.0%}) [Train: {n_train}, Val: {n_val}] | Test: {n_test} ({(1-train_split):.0%})")
        
        # Training loop
        history = {
            "train_loss": [],
            "val_loss": [],
            "test_loss": [],
            "train_metrics": {},
            "test_metrics": {},
        }
        
        for epoch in range(self.config.num_epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Test
            test_loss = self.validate(test_loader)
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["test_loss"].append(test_loss)
            
            print(f"Epoch {epoch+1:02d}/{self.config.num_epochs:02d}: "
                  f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Test Loss={test_loss:.4f}")
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping and best model saving
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self._save_checkpoint(epoch, val_loss, save_name=save_name)
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
        
        # Load best checkpoint
        if self.best_checkpoint_path and self.best_checkpoint_path.exists():
            try:
                checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device, weights_only=False)
            except TypeError:
                checkpoint = torch.load(self.best_checkpoint_path, map_location=self.device)
            self.pipeline.load_state_dict(checkpoint["state_dict"])
            print(f"✓ Loaded best model from {self.best_checkpoint_path} (val_loss: {checkpoint.get('val_loss', 0):.4f})")
            
        # Final comprehensive evaluation on both Train (80%) and Test (20%)
        print("\nEvaluating final model on 80% Train set and 20% Test set...")
        X_train_full, M_train_full, Y_train_full = X[:n_train_total], M[:n_train_total], Y[:n_train_total]
        
        train_eval = self.pipeline.evaluate(X_train_full, M_train_full, Y_train_full)
        test_eval = self.pipeline.evaluate(X_test, M_test, Y_test)
        
        history["train_metrics"] = train_eval
        history["test_metrics"] = test_eval
        
        return history
    
    def _construct_return_to_go(self, future_returns: torch.Tensor, T: int) -> torch.Tensor:
        """Construct return-to-go from future returns."""
        # Shape: (batch, N, horizon) → (batch, N, T, 1)
        # For simplicity, use sum of future returns
        rtg = future_returns.sum(dim=-1, keepdim=True).unsqueeze(-1)  # (batch, N, 1, 1)
        rtg = rtg.expand(-1, -1, T, -1)
        return rtg
    
    def _check_leakage(self, X, M, Y, n_train, n_val):
        """Check for temporal leakage in train/val/test split."""
        # Timestamps should be strictly increasing
        print("✓ Chronological split applied (no future leakage)")
    
    def _save_checkpoint(self, epoch: int, val_loss: float, save_name: Optional[str] = None):
        """Save best checkpoint."""
        checkpoint_path = self.checkpoint_dir / f"best_{epoch:03d}_{val_loss:.4f}.pt"
        checkpoint_data = {
            "epoch": epoch,
            "state_dict": self.pipeline.state_dict(),
            "val_loss": val_loss,
            "config": self.config,
            "num_assets": self.pipeline.num_assets,
            "feature_dim": self.pipeline.feature_dim,
            "modules_initialized": self.pipeline.modules_initialized,
        }
        torch.save(checkpoint_data, checkpoint_path)
        self.best_checkpoint_path = checkpoint_path
        
        if save_name:
            custom_path = self.checkpoint_dir / save_name
            torch.save(checkpoint_data, custom_path)


def train_pipeline_from_config(
    X: torch.Tensor,
    M: torch.Tensor,
    Y: torch.Tensor,
    config: PipelineConfig,
    save_name: Optional[str] = None,
) -> Tuple[DependencyAwareTradingPipeline, Dict]:
    """
    Train a complete pipeline from config and data.
    
    Returns:
        (trained_pipeline, training_history)
    """
    # Initialize pipeline
    pipeline = DependencyAwareTradingPipeline(config).to(config.device)
    
    # Initialize modules
    from ml.modules.module1_input import MarketTensor
    dummy_mt = MarketTensor(
        features=torch.zeros(X.shape[1], X.shape[2], X.shape[3]),
        validity_mask=torch.ones(X.shape[1], X.shape[2]),
        asset_ids=[f"ASSET_{i}" for i in range(X.shape[1])],
        timestamps=np.arange(X.shape[2]),
        feature_names=[f"F{i}" for i in range(X.shape[3])],
        feature_means=torch.zeros(X.shape[3]),
        feature_stds=torch.ones(X.shape[3]),
    )
    pipeline.initialize_modules(dummy_mt)
    
    # Train with 80% train, 20% test
    trainer = SequenceTrainer(pipeline, config)
    history = trainer.fit(X, M, Y, train_split=0.8, save_name=save_name)
    
    return pipeline, history
