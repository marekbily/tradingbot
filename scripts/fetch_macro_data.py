"""Fetch common macro CSVs into the repository `data/` directory.

Usage:
    python scripts/fetch_macro_data.py --all
    python scripts/fetch_macro_data.py --names spx,bitcoin --start 2004-01-01 --end 2025-12-31

The script uses `yfinance` and will try a few alternate tickers for some symbols.
"""
import argparse
import logging
from pathlib import Path
import pandas as pd
import yfinance as yf


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYMBOLS = {
    'dxy': {
        'filenames': ['dxy_daily.csv'],
        'tickers': ['DX-Y.NYB', 'DXY', '^DXY']
    },
    'spx': {
        'filenames': ['spx_daily.csv'],
        'tickers': ['^GSPC']
    },
    'us10y': {
        'filenames': ['us10y_daily.csv'],
        'tickers': ['^TNX']
    },
    'vix': {
        'filenames': ['vix_daily.csv'],
        'tickers': ['^VIX']
    },
    'oil': {
        'filenames': ['oil_wti_daily.csv'],
        'tickers': ['CL=F', 'BZ=F']
    },
    'bitcoin': {
        'filenames': ['bitcoin_daily.csv'],
        'tickers': ['BTC-USD']
    },
    'eur': {
        'filenames': ['eurusd_daily.csv'],
        'tickers': ['EURUSD=X']
    },
    'silver': {
        'filenames': ['silver_daily.csv'],
        'tickers': ['SI=F', 'XAGUSD=X']
    },
    'gld': {
        'filenames': ['gld_etf_daily.csv'],
        'tickers': ['GLD']
    }
}


def standardize_and_save(df: pd.DataFrame, out_path: Path):
    """Normalize column names and write CSV with a `time` and `close` column at minimum."""
    if df is None or df.empty:
        raise ValueError("Empty DataFrame")

    df = df.copy()
    # Reset index into a time column
    df = df.reset_index()
    # Some yfinance downloads use 'Date' as index name, reset_index covers that
    # Normalize column names to lowercase
    df.columns = [str(c).lower() for c in df.columns]

    # Ensure we have a time column. Accept 'date', 'datetime', 'index', or any column containing 'date'/'time'.
    renamed = False
    for col in df.columns:
        if col == 'time':
            renamed = True
            break
        if col in ('date', 'datetime', 'index') or 'date' in col or 'time' in col:
            df = df.rename(columns={col: 'time'})
            renamed = True
            break

    if not renamed:
        # fallback: try to locate a datetime-like column by dtype
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df = df.rename(columns={col: 'time'})
                renamed = True
                break

    if 'time' not in df.columns:
        raise ValueError(f'No time column found; columns: {list(df.columns)}')

    # Prefer 'adj close' over 'close' if available
    if 'adj close' in df.columns:
        df['close'] = df['adj close']

    # Keep canonical columns
    cols = ['time']
    for c in ['open', 'high', 'low', 'close', 'volume']:
        if c in df.columns:
            cols.append(c)
        else:
            df[c] = pd.NA
            cols.append(c)

    out_df = df[cols].copy()
    # Ensure time formatting
    out_df['time'] = pd.to_datetime(out_df['time'])

    out_df.to_csv(out_path, index=False)
    logger.info(f"Saved {out_path} ({len(out_df):,} rows)")


def fetch_symbol(name: str, dest_dir: Path, start: str, end: str):
    meta = SYMBOLS.get(name)
    if not meta:
        logger.error(f"Unknown symbol: {name}")
        return False

    tickers = meta['tickers']
    filename = meta['filenames'][0]
    out_path = dest_dir / filename

    for tk in tickers:
        logger.info(f"Attempting {name} using ticker {tk}...")
        try:
            df = yf.download(tk, start=start, end=end, interval='1d', progress=False)
        except Exception as e:
            logger.warning(f"yfinance download failed for {tk}: {e}")
            df = None

        if df is None or df.empty:
            logger.warning(f"No data for ticker {tk}, trying next candidate if any.")
            continue

        try:
            standardize_and_save(df, out_path)
            return True
        except Exception as e:
            logger.warning(f"Failed to standardize/save for {tk}: {e}")
            continue

    logger.error(f"Failed to fetch any ticker for {name}")
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true', help='Fetch all symbols')
    parser.add_argument('--names', type=str, help='Comma-separated list of symbol keys to fetch (spx, dxy, ... )')
    parser.add_argument('--start', type=str, default='2004-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default='2025-12-31', help='End date (YYYY-MM-DD)')
    parser.add_argument('--out', type=str, default='data', help='Output directory')

    args = parser.parse_args()
    dest = Path(args.out)
    dest.mkdir(parents=True, exist_ok=True)

    if args.all:
        to_fetch = list(SYMBOLS.keys())
    elif args.names:
        to_fetch = [n.strip() for n in args.names.split(',') if n.strip()]
    else:
        parser.error('Either --all or --names must be provided')

    success = {}
    for name in to_fetch:
        ok = fetch_symbol(name, dest, args.start, args.end)
        success[name] = ok

    logger.info('Fetch summary:')
    for k, v in success.items():
        logger.info(f"  {k}: {'OK' if v else 'FAILED'}")


if __name__ == '__main__':
    main()
