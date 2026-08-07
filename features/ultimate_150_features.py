import pandas as pd
import numpy as np
import logging
from pathlib import Path

from data.load_data import load_ohlc_csv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def make_ultimate_features(
    base_timeframe='M5',
    data_dir='data',
    start_date=None,
    end_date=None,
    warmup_days=30,
    export_path: str | Path | None = None,
    export_full_path: str | Path | None = None,
    export_rows: int = 5000,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Create complete 150+ feature set

    Args:
        base_timeframe: Base timeframe to use ('M5' recommended for speed)
        data_dir: Directory containing data files
        start_date: Optional inclusive start date for the final output window
        end_date: Optional exclusive end date for the final output window
        warmup_days: Extra lookback window to keep before start_date for rolling indicators
        export_path: Optional CSV path to write a preview of the generated training table
        export_full_path: Optional path to write the full assembled training dataset
        export_rows: Number of rows to export when export_path is provided

    Returns:
        features (ndarray): Shape (N, 152+), dtype float32
        returns (ndarray): Shape (N,), target returns
        timestamps (DatetimeIndex): Shape (N,), timestamps for each sample
    """
    logger.info("="*70)
    logger.info("🚀 ULTIMATE 150+ FEATURE SYSTEM")
    logger.info("="*70)
    logger.info(f"Base timeframe: {base_timeframe}")
    logger.info(f"Data directory: {data_dir}")
    if start_date is not None or end_date is not None:
        logger.info(f"Date window: {start_date} -> {end_date} (warmup {warmup_days} days)")
    logger.info("")

    load_start = start_date
    if start_date is not None and warmup_days:
        load_start = pd.Timestamp(start_date) - pd.Timedelta(days=warmup_days)

    # ========== STEP 1: LOAD TIMEFRAME FEATURES (96) ==========
    logger.info("📊 STEP 1/5: Loading timeframe features...")
    logger.info("-" * 70)

    from features.timeframe_features import load_and_compute_all_timeframes

    tf_features = load_and_compute_all_timeframes(
        base_timeframe=base_timeframe,
        data_dir=data_dir,
        start_date=load_start,
        end_date=end_date,
    )

    logger.info(f"✅ Loaded {len(tf_features)} timeframes")
    total_tf_features = sum(df.shape[1] for df in tf_features.values())
    logger.info(f"✅ Total timeframe features: {total_tf_features}")

    # ========== STEP 2: COMPUTE CROSS-TIMEFRAME FEATURES (12) ==========
    logger.info("\n🔄 STEP 2/5: Computing cross-timeframe features...")
    logger.info("-" * 70)

    from features.cross_timeframe import compute_all_cross_tf_features

    cross_tf_features = compute_all_cross_tf_features(tf_features)

    logger.info(f"✅ Cross-timeframe features: {cross_tf_features.shape[1]}")

    # ========== STEP 3: COMPUTE MACRO FEATURES (24) ==========
    logger.info("\n🌍 STEP 3/5: Computing macro features...")
    logger.info("-" * 70)

    from features.macro_features import load_macro_data, compute_macro_features

    # Load base timeframe data with close prices
    base_data_file = {
        'M5': 'xauusd_m5.csv',
        'M15': 'xauusd_m15.csv',
        'H1': 'xauusd_h1_from_m1.csv',
    }.get(base_timeframe, 'xauusd_m5.csv')

    df_gold = load_ohlc_csv(f"{data_dir}/{base_data_file}")
    if load_start is not None:
        df_gold = df_gold[df_gold['time'] >= pd.Timestamp(load_start)]
    if end_date is not None:
        df_gold = df_gold[df_gold['time'] < pd.Timestamp(end_date)]
    df_gold = df_gold.set_index('time').sort_index()

    macro_data = load_macro_data(data_dir=data_dir)
    macro_features = compute_macro_features(df_gold, macro_data)

    logger.info(f"✅ Macro features: {macro_features.shape[1]}")

    # ========== STEP 4: COMPUTE CALENDAR FEATURES (8) ==========
    logger.info("\n📅 STEP 4/5: Computing economic calendar features...")
    logger.info("-" * 70)

    from features.calendar_features import load_economic_calendar, compute_calendar_features

    calendar = load_economic_calendar(filepath=f"{data_dir}/economic_events_2015_2025.json")

    # Use the base timeframe index
    base_index = tf_features[base_timeframe].index

    if start_date is not None:
        base_index = base_index[base_index >= pd.Timestamp(start_date)]
    if end_date is not None:
        base_index = base_index[base_index < pd.Timestamp(end_date)]

    calendar_features = compute_calendar_features(base_index, calendar)

    logger.info(f"✅ Calendar features: {calendar_features.shape[1]}")

    # ========== STEP 5: COMPUTE MICROSTRUCTURE FEATURES (12) ==========
    logger.info("\n🏛️  STEP 5/5: Computing market microstructure features...")
    logger.info("-" * 70)

    from features.microstructure_features import compute_all_microstructure_features

    microstructure_features = compute_all_microstructure_features(df_gold)

    logger.info(f"✅ Microstructure features: {microstructure_features.shape[1]}")

    # ========== STEP 6: COMBINE ALL FEATURES ==========
    logger.info("\n🔗 COMBINING ALL FEATURES...")
    logger.info("-" * 70)

    # Align all feature DataFrames to the same index (base_index)
    all_feature_dfs = []

    # Add all timeframe features
    for tf_name in sorted(tf_features.keys()):
        df_tf = tf_features[tf_name]
        df_aligned = df_tf.reindex(base_index, method='ffill')
        all_feature_dfs.append(df_aligned)
        logger.info(f"   • {tf_name}: {df_aligned.shape[1]} features")

    # Add cross-timeframe
    cross_tf_aligned = cross_tf_features.reindex(base_index, method='ffill')
    all_feature_dfs.append(cross_tf_aligned)
    logger.info(f"   • Cross-TF: {cross_tf_aligned.shape[1]} features")

    # Add macro
    macro_aligned = macro_features.reindex(base_index, method='ffill')
    all_feature_dfs.append(macro_aligned)
    logger.info(f"   • Macro: {macro_aligned.shape[1]} features")

    # Add calendar
    calendar_aligned = calendar_features.reindex(base_index, method='ffill')
    all_feature_dfs.append(calendar_aligned)
    logger.info(f"   • Calendar: {calendar_aligned.shape[1]} features")

    # Add microstructure
    micro_aligned = microstructure_features.reindex(base_index, method='ffill')
    all_feature_dfs.append(micro_aligned)
    logger.info(f"   • Microstructure: {micro_aligned.shape[1]} features")

    # Concatenate everything
    all_features = pd.concat(all_feature_dfs, axis=1)

    # ========== STEP 7: CLEAN AND PREPARE ==========
    logger.info("\n🧹 CLEANING DATA...")
    logger.info("-" * 70)

    # Fill any remaining NaNs with 0
    nan_count_before = all_features.isna().sum().sum()
    if nan_count_before > 0:
        logger.info(f"   • Filling {nan_count_before:,} NaN values with 0")
        all_features = all_features.fillna(0.0)

    # Replace inf values
    inf_count = np.isinf(all_features.values).sum()
    if inf_count > 0:
        logger.info(f"   • Replacing {inf_count:,} inf values with 0")
        all_features = all_features.replace([np.inf, -np.inf], 0.0)

    # Convert to float32 for memory efficiency
    all_features = all_features.astype(np.float32)

    # ========== STEP 8: COMPUTE TARGET RETURNS ==========
    logger.info("\n🎯 COMPUTING TARGET RETURNS...")
    logger.info("-" * 70)

    # Use base timeframe close prices for returns
    df_gold_aligned = df_gold.reindex(base_index, method='ffill')
    returns = df_gold_aligned['close'].pct_change().fillna(0.0).values.astype(np.float32)

    logger.info(f"   • Return samples: {len(returns):,}")

    if export_path is not None:
        export_path = Path(export_path)
        export_path.parent.mkdir(parents=True, exist_ok=True)

        export_df = all_features.head(export_rows).copy()
        export_df.insert(0, 'time', export_df.index)
        export_df['target_return'] = pd.Series(returns, index=all_features.index).head(export_rows).values
        export_df.to_csv(export_path, index=False)
        logger.info(f"💾 Exported preview dataset: {export_path} ({len(export_df):,} rows)")

    if export_full_path is not None:
        export_full_path = Path(export_full_path)
        export_full_path.parent.mkdir(parents=True, exist_ok=True)

        if export_full_path.suffix.lower() in {'.parquet', '.pq'}:
            full_export_df = all_features.copy()
            full_export_df.insert(0, 'time', full_export_df.index)
            full_export_df['target_return'] = returns
            full_export_df.to_parquet(export_full_path, index=False)
            logger.info(
                f"💾 Exported full training dataset: {export_full_path} "
                f"({len(full_export_df):,} rows, {full_export_df.shape[1]} columns)"
            )
        else:
            chunk_rows = 50_000
            total_rows = len(all_features)
            total_columns = all_features.shape[1] + 2

            with export_full_path.open('w', encoding='utf-8', newline='') as handle:
                for start in range(0, total_rows, chunk_rows):
                    end = min(start + chunk_rows, total_rows)
                    chunk = all_features.iloc[start:end].copy()
                    chunk.insert(0, 'time', chunk.index)
                    chunk['target_return'] = returns[start:end]
                    chunk.to_csv(handle, index=False, header=(start == 0))

            logger.info(
                f"💾 Exported full training dataset: {export_full_path} "
                f"({total_rows:,} rows, {total_columns} columns)"
            )

    # ========== FINAL SUMMARY ==========
    logger.info("\n" + "="*70)
    logger.info("✅ ULTIMATE FEATURES CREATED!")
    logger.info("="*70)

    logger.info(f"\n📊 Feature Summary:")
    logger.info(f"   • Total features: {all_features.shape[1]}")
    logger.info(f"   • Total samples: {len(all_features):,}")
    logger.info(f"   • Memory usage: {all_features.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    logger.info(f"   • Date range: {base_index[0]} to {base_index[-1]}")

    # Feature breakdown
    logger.info(f"\n📈 Feature Breakdown:")

    feature_counts = {
        'Timeframe (M5)': 16,
        'Timeframe (M15)': 16,
        'Timeframe (H1)': 16,
        'Timeframe (H4)': 16,
        'Timeframe (D1)': 16,
        'Timeframe (W1)': 16 if 'W1' in tf_features else 0,
        'Cross-Timeframe': cross_tf_aligned.shape[1],
        'Macro': macro_aligned.shape[1],
        'Calendar': calendar_aligned.shape[1],
        'Microstructure': micro_aligned.shape[1],
    }

    for name, count in feature_counts.items():
        if count > 0:
            logger.info(f"   • {name:20} {count:3} features")

    logger.info(f"\n🎯 Ready for training!")
    logger.info(f"   • Observation space: {all_features.shape[1]} features")
    logger.info(f"   • Action space: 3 (buy/hold/sell)")
    logger.info(f"   • Training samples: {len(all_features):,}")

    # Return as numpy arrays
    return (
        all_features.values,  # Features (N, 152+)
        returns,              # Returns (N,)
        all_features.index    # Timestamps (N,)
    )


def test_ultimate_features(
    export_path: str | Path | None = None,
    export_full_path: str | Path | None = None,
):
    """
    Quick test to verify the complete system works
    """
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING ULTIMATE FEATURE SYSTEM")
    logger.info("="*70)

    try:
        # Generate features
        X, r, timestamps = make_ultimate_features(
            base_timeframe='M5',
            export_path=export_path,
            export_full_path=export_full_path,
        )

        logger.info("\n✅ Ultimate feature system test PASSED!")

        logger.info(f"\n📊 Output shapes:")
        logger.info(f"   • Features (X): {X.shape}")
        logger.info(f"   • Returns (r): {r.shape}")
        logger.info(f"   • Timestamps: {len(timestamps)}")

        logger.info(f"\n📈 Feature statistics:")
        logger.info(f"   • Mean: {X.mean():.6f}")
        logger.info(f"   • Std: {X.std():.6f}")
        logger.info(f"   • Min: {X.min():.6f}")
        logger.info(f"   • Max: {X.max():.6f}")
        logger.info(f"   • NaN count: {np.isnan(X).sum()}")
        logger.info(f"   • Inf count: {np.isinf(X).sum()}")

        logger.info(f"\n🎯 Return statistics:")
        logger.info(f"   • Mean return: {r.mean():.6f}")
        logger.info(f"   • Std return: {r.std():.6f}")
        logger.info(f"   • Sharpe (approx): {r.mean() / (r.std() + 1e-8):.4f}")

        return X, r, timestamps

    except Exception as e:
        logger.error(f"❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    # Run full system test
    X, r, timestamps = test_ultimate_features(
        export_path=Path("data/ultimate_150_features_preview.csv"),
        export_full_path=Path("data/ultimate_150_features_full.csv"),
    )

    logger.info("\n" + "="*70)
    logger.info("🚀 ULTIMATE 150+ FEATURE SYSTEM READY!")
    logger.info("="*70)

    logger.info("""
📋 USAGE IN TRAINING:
    from features.ultimate_150_features import make_ultimate_features

    # Generate all 150+ features
    X, returns, timestamps = make_ultimate_features(base_timeframe='M5')

    # X is now ready for training with shape (N, 152+)
    # Use with DreamerV3 or any RL algorithm

    # Example:
    # env = TradingEnvironment(X, returns)
    # agent.train(env, steps=1000000)
    """)

    logger.info("\n🎉 You're ready to train the GOD MODE AI!")
