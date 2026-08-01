"""Quick proof-of-concept evaluation for a DreamerV3 checkpoint.

This script is intentionally small and fast:
- loads only a narrow date window plus warmup context
- uses the ultimate 150 feature pipeline on that slice only
- runs a short deterministic rollout and prints metrics

Useful for catching checkpoint / API / shape errors in 1-2 minutes instead of
waiting for a full-history evaluation to finish.
"""

import argparse
import logging
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.ultimate_150_features import make_ultimate_features
from models.dreamer_agent import DreamerV3Agent
from train.train_ultimate_150 import TradingEnvironment, WINDOW, COST

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_metrics(equity_curve, positions, rewards):
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)

    returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-12)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve) / np.maximum(peak, 1e-12)

    metrics = {
        'final_equity': float(equity_curve[-1]) if len(equity_curve) else 1.0,
        'total_return': float((equity_curve[-1] - 1.0) * 100.0) if len(equity_curve) else 0.0,
        'max_drawdown': float(drawdown.max() * 100.0) if len(drawdown) else 0.0,
        'num_trades': int(np.sum(np.abs(np.diff(positions)) > 0)) if len(positions) > 1 else 0,
        'win_rate': float(np.mean(rewards > 0) * 100.0) if len(rewards) else 0.0,
        'long_percentage': float(np.mean(positions == 1) * 100.0) if len(positions) else 0.0,
        'sharpe_proxy': float(np.mean(returns) / (np.std(returns) + 1e-8)) if len(returns) else 0.0,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Quick DreamerV3 smoke-test evaluator')
    parser.add_argument('--checkpoint', type=str, default='train/dreamer/dreamer_xauusd_final.pt')
    parser.add_argument('--start', type=str, default='2025-12-01', help='Eval window start date')
    parser.add_argument('--end', type=str, default='2025-12-16', help='Eval window end date')
    parser.add_argument('--warmup-days', type=int, default=30, help='Extra lookback to compute indicators')
    parser.add_argument('--base-tf', type=str, default='M5')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--max-steps', type=int, default=25000, help='Hard cap on rollout steps')
    parser.add_argument('--debug-steps', type=int, default=10, help='Print detailed info for the first N steps')
    parser.add_argument('--report-every', type=int, default=250, help='Print periodic rollout summaries')
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

    logger.info("⚡ Quick Dreamer evaluation")
    logger.info(f"   Checkpoint: {args.checkpoint}")
    logger.info(f"   Window: {args.start} -> {args.end}")
    logger.info(f"   Warmup: {args.warmup_days} days")
    logger.info(f"   Device: {device}")

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    load_start = str(np.datetime64(args.start) - np.timedelta64(args.warmup_days, 'D'))

    logger.info("📊 Loading sliced ultimate features...")
    X, returns, timestamps = make_ultimate_features(
        base_timeframe=args.base_tf,
        start_date=load_start,
        end_date=args.end,
        warmup_days=0,
    )

    # Restrict to the actual evaluation window only.
    mask = (timestamps >= np.datetime64(args.start)) & (timestamps < np.datetime64(args.end))
    X_eval = X[mask]
    r_eval = returns[mask]
    ts_eval = timestamps[mask]

    if len(X_eval) == 0:
        raise SystemExit(f"No samples found in evaluation window {args.start} -> {args.end}")

    # Quick smoke-test normalization: fit on the loaded slice so the run is self-contained.
    feature_mean = X_eval.mean(axis=0, keepdims=True)
    feature_std = np.maximum(X_eval.std(axis=0, keepdims=True), 1e-6)
    X_eval = (X_eval - feature_mean) / feature_std

    logger.info("📐 Slice diagnostics")
    logger.info(f"   X_eval shape: {X_eval.shape}")
    logger.info(f"   r_eval shape: {r_eval.shape}")
    logger.info(f"   Feature mean (overall): {float(X_eval.mean()):.6f}")
    logger.info(f"   Feature std  (overall): {float(X_eval.std()):.6f}")
    logger.info(f"   Return mean: {float(r_eval.mean()):.8f}")
    logger.info(f"   Return std : {float(r_eval.std()):.8f}")

    env = TradingEnvironment(X_eval, r_eval, window=WINDOW, cost_per_trade=COST, device=device)

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
    action_hist = {0: 0, 1: 0}
    position_hist = {0: 0, 1: 0}
    first_actions = []

    max_steps = min(args.max_steps, len(X_eval))
    logger.info(f"🏁 Rolling out up to {max_steps:,} steps...")
    logger.info("🔎 Debug telemetry enabled")

    for step in tqdm(range(max_steps), desc='Quick eval'):
        prev_equity = equity_curve[-1]
        prev_pos = positions[-1]
        action, (h, z) = agent.act(obs, h, z, deterministic=True)
        action_idx = int(torch.argmax(action).item())
        obs, reward, done, info = env.step(action)

        reward_value = float(reward.item()) if torch.is_tensor(reward) else float(reward)
        rewards.append(float(reward.item()) if torch.is_tensor(reward) else float(reward))
        equity_curve.append(info['equity'])
        positions.append(info['position'])
        action_hist[action_idx] = action_hist.get(action_idx, 0) + 1
        position_hist[info['position']] = position_hist.get(info['position'], 0) + 1
        if step < args.debug_steps:
            first_actions.append(action_idx)
            logger.info(
                f"step={step:05d} action={action_idx} pos {prev_pos}->{info['position']} "
                f"reward={reward_value:+.8f} equity {prev_equity:.6f}->{info['equity']:.6f} "
                f"obs_mean={float(obs.mean()):+.6f} obs_std={float(obs.std()):.6f}"
            )
        elif args.report_every > 0 and (step + 1) % args.report_every == 0:
            recent_rewards = np.asarray(rewards[-args.report_every:], dtype=np.float64)
            recent_equity = equity_curve[-1]
            recent_positions = np.asarray(positions[-(args.report_every + 1):], dtype=np.int64)
            recent_trades = int(np.sum(np.abs(np.diff(recent_positions)) > 0)) if len(recent_positions) > 1 else 0
            logger.info(
                f"summary@{step+1:05d} trades={recent_trades} "
                f"flat%={100.0 * position_hist.get(0, 0) / max(1, len(positions)-1):.2f} "
                f"long%={100.0 * position_hist.get(1, 0) / max(1, len(positions)-1):.2f} "
                f"mean_r={recent_rewards.mean():+.8f} std_r={recent_rewards.std():.8f} "
                f"equity={recent_equity:.6f}"
            )

        if done:
            break

    metrics = compute_metrics(equity_curve, positions, rewards)

    logger.info("\n" + "=" * 70)
    logger.info("📊 QUICK EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Samples:         {len(X_eval):,}")
    logger.info(f"Date range:      {ts_eval[0]} to {ts_eval[-1]}")
    logger.info(f"Final Equity:    {metrics['final_equity']:.4f}x")
    logger.info(f"Total Return:    {metrics['total_return']:.2f}%")
    logger.info(f"Max Drawdown:    {metrics['max_drawdown']:.2f}%")
    logger.info(f"Trades:          {metrics['num_trades']}")
    logger.info(f"Win Rate:        {metrics['win_rate']:.2f}%")
    logger.info(f"Long %:          {metrics['long_percentage']:.2f}%")
    logger.info(f"Sharpe Proxy:    {metrics['sharpe_proxy']:.4f}")
    logger.info(f"Action histogram: {action_hist}")
    logger.info(f"Position histogram: {position_hist}")
    logger.info(f"First actions:    {first_actions}")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
