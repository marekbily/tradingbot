"""Backtest a DreamerV3 checkpoint on a forward XAUUSD window.

This is a lightweight runner that mirrors the evaluation preprocessing used by
`evaluate_model.py`:
- loads ultimate 150+ features
- computes train-only normalization statistics
- evaluates the model on a forward window
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.ultimate_150_features import make_ultimate_features
from models.dreamer_agent import DreamerV3Agent
from train.train_ultimate_150 import TradingEnvironment, WINDOW, COST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(equity_curve, positions):
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)

    returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-12)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / np.maximum(peak, 1e-12)

    trades = int(np.sum(np.abs(np.diff(positions)) > 0))
    win_rate = float(np.mean(returns > 0) * 100.0) if len(returns) else 0.0

    return {
        'final_equity': float(equity_curve[-1]) if len(equity_curve) else 1.0,
        'total_return': float((equity_curve[-1] - 1.0) * 100.0) if len(equity_curve) else 0.0,
        'max_drawdown': float(drawdown.max() * 100.0) if len(drawdown) else 0.0,
        'num_trades': trades,
        'win_rate': win_rate,
        'avg_return': float(np.mean(returns) * 100.0) if len(returns) else 0.0,
    }


def main():
    parser = argparse.ArgumentParser(description='DreamerV3 forward backtest')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint .pt file. If omitted, latest in train/dreamer_ultimate is used')
    parser.add_argument('--train-cutoff', type=str, default='2022-01-01')
    parser.add_argument('--start', type=str, default='2022-01-14')
    parser.add_argument('--end', type=str, default='2025-12-31')
    parser.add_argument('--base-tf', type=str, default='M5')
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device

    logger.info('📊 Loading Ultimate 150+ features...')
    X, returns, timestamps = make_ultimate_features(base_timeframe=args.base_tf)
    logger.info(f'✅ Loaded {X.shape[1]} features, {len(X):,} samples')

    train_cutoff = np.datetime64(args.train_cutoff)
    train_idx = np.searchsorted(timestamps, train_cutoff, side='left')
    if train_idx <= 0:
        train_idx = len(X) // 2

    feature_mean = X[:train_idx].mean(axis=0, keepdims=True)
    feature_std = X[:train_idx].std(axis=0, keepdims=True)
    feature_std = np.maximum(feature_std, 1e-6)
    X = (X - feature_mean) / feature_std

    mask = (timestamps >= args.start) & (timestamps < args.end)
    X_eval = X[mask]
    r_eval = returns[mask]
    ts_eval = timestamps[mask]

    if len(X_eval) == 0:
        raise SystemExit(f'No samples in forward window {args.start} to {args.end}')

    logger.info(f'📅 Forward window: {ts_eval[0]} to {ts_eval[-1]}')
    logger.info(f'   Samples: {len(X_eval):,}')

    env = TradingEnvironment(X_eval, r_eval, window=WINDOW, cost_per_trade=COST, device=device)

    # Resolve checkpoint path
    from utils.checkpoint_utils import ensure_checkpoint_path
    ckpt = ensure_checkpoint_path(args.checkpoint, default_folder='train/dreamer_ultimate')
    if ckpt is None:
        raise SystemExit('No checkpoint found. Train the model first or provide --checkpoint')
    args.checkpoint = ckpt

    agent = DreamerV3Agent(
        obs_dim=env.observation_space,
        action_dim=env.action_space,
        embed_dim=256,
        hidden_dim=512,
        stoch_dim=32,
        num_categories=32,
        device=device,
    )
    agent.load(args.checkpoint)

    obs = env.reset()
    h, z = None, None
    equity_curve = [1.0]
    positions = [0]
    rewards = []

    while True:
        action, (h, z) = agent.act(obs, h, z, deterministic=True)
        obs, reward, done, info = env.step(action)

        rewards.append(float(reward.item()) if torch.is_tensor(reward) else float(reward))
        equity_curve.append(info['equity'])
        positions.append(info['position'])

        if done:
            break

    metrics = compute_metrics(equity_curve, positions)

    logger.info('\n' + '=' * 70)
    logger.info('📊 DREAMER BACKTEST RESULTS')
    logger.info('=' * 70)
    logger.info(f"Checkpoint:      {args.checkpoint}")
    logger.info(f"Final Equity:    {metrics['final_equity']:.4f}x")
    logger.info(f"Total Return:    {metrics['total_return']:.2f}%")
    logger.info(f"Max Drawdown:    {metrics['max_drawdown']:.2f}%")
    logger.info(f"Trades:          {metrics['num_trades']}")
    logger.info(f"Win Rate:        {metrics['win_rate']:.2f}%")
    logger.info(f"Avg Return:      {metrics['avg_return']:.4f}%")
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
