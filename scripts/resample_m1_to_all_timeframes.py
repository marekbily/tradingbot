"""
Resample M1 Data to All Timeframes

Converts M1 (1-minute) XAUUSD data to:
- M5 (5-minute)
- M15 (15-minute)
- H1 (1-hour)
- H4 (4-hour)
- D1 (Daily)
- W1 (Weekly)

This maximizes the value of your high-resolution M1 data.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_m1_data(filepath):
    """
    Load M1 data from MetaTrader format

    Format: <DATE>\t<TIME>\t<OPEN>\t<HIGH>\t<LOW>\t<CLOSE>\t<TICKVOL>\t<VOL>\t<SPREAD>
    """
    logger.info(f"📥 Loading M1 data from {filepath}...")

    # Load with tab separator
    df = pd.read_csv(filepath, sep='\t')

    # Combine date and time
    df['datetime'] = pd.to_datetime(df['<DATE>'] + ' ' + df['<TIME>'], format='%Y.%m.%d %H:%M:%S')

    # Rename columns to standard format
    df = df.rename(columns={
        '<OPEN>': 'open',
        '<HIGH>': 'high',
        '<LOW>': 'low',
        '<CLOSE>': 'close',
        '<TICKVOL>': 'volume',
    })

    # Select relevant columns
    df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]

    # Set datetime as index
    df = df.set_index('datetime')

    # Sort by time
    df = df.sort_index()

    logger.info(f"   ✅ Loaded {len(df):,} M1 bars")
    logger.info(f"   Date range: {df.index[0]} to {df.index[-1]}")

    return df


def resample_ohlcv(df, rule, name):
    """
    Resample OHLCV data to different timeframe

    Args:
        df: DataFrame with OHLCV data
        rule: Pandas resample rule ('5T', '15T', '1H', '4H', '1D')
        name: Name for logging (e.g., 'M5', 'H1')

    Returns:
        Resampled DataFrame
    """
    logger.info(f"📊 Resampling to {name}...")

    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Drop rows with NaN (incomplete periods)
    resampled = resampled.dropna()

    logger.info(f"   ✅ {name}: {len(resampled):,} bars")

    return resampled


def save_to_csv(df, filepath):
    """Save DataFrame to CSV"""
    # Reset index to make datetime a column
    df_save = df.reset_index()
    df_save = df_save.rename(columns={'datetime': 'time'})

    # Save
    df_save.to_csv(filepath, index=False)

    logger.info(f"   💾 Saved to: {filepath}")


def main():
    """Main function"""

    logger.info("="*70)
    logger.info("🚀 RESAMPLING M1 DATA TO ALL TIMEFRAMES")
    logger.info("="*70)

    # Configuration
    M1_FILE = 'data/XAUUSD_M1.csv'
    OUTPUT_DIR = 'data'

    # Load M1 data
    logger.info("\n📥 Loading M1 data...")
    df_m1 = load_m1_data(M1_FILE)

    # Resample to all timeframes
    logger.info("\n📊 Resampling to multiple timeframes...\n")

    timeframes = {
        'M5': ('5min', 'xauusd_m5.csv'),
        'M15': ('15min', 'xauusd_m15.csv'),
        'H1': ('1h', 'xauusd_h1_from_m1.csv'),
        'H4': ('4h', 'xauusd_h4_from_m1.csv'),
        'D1': ('1d', 'xauusd_d1_from_m1.csv'),
        'W1': ('1w', 'xauusd_w1.csv'),
    }

    results = {}

    for name, (rule, filename) in timeframes.items():
        df_resampled = resample_ohlcv(df_m1, rule, name)
        filepath = f"{OUTPUT_DIR}/{filename}"
        save_to_csv(df_resampled, filepath)
        results[name] = len(df_resampled)
        print()  # Blank line for readability

    # Summary
    logger.info("="*70)
    logger.info("📊 RESAMPLING SUMMARY")
    logger.info("="*70)

    logger.info(f"\n✅ Original M1 data: {len(df_m1):,} bars")
    logger.info(f"\n📊 Generated timeframes:")

    for name, count in results.items():
        compression_ratio = len(df_m1) / count
        logger.info(f"   • {name:4} {count:7,} bars (compression: {compression_ratio:.1f}x)")

    logger.info(f"\n💾 All files saved to: {OUTPUT_DIR}/")

    logger.info("\n" + "="*70)
    logger.info("✅ RESAMPLING COMPLETE!")
    logger.info("="*70)

    logger.info("""
📋 NEXT STEPS:
1. ✅ You now have all timeframes (M5, M15, H1, H4, D1)
2. 🔜 Next: Build ultimate features with ALL timeframes
3. 🔜 Then: Create ultimate training script
4. 🚀 Finally: Train for 1M steps!

⏱️  Time to completion: ~3 hours of my work, then training!
    """)

    return results


if __name__ == "__main__":
    results = main()
    print("\n🔥 All timeframes ready for training!")
