import yfinance as yf
import pandas as pd
import os

"""add 

WARNING:features.macro_features:⚠️  VIX file not found: vix_daily.csv
WARNING:features.macro_features:⚠️  OIL file not found: oil_wti_daily.csv
WARNING:features.macro_features:⚠️  BTC file not found: bitcoin_daily.csv
WARNING:features.macro_features:⚠️  EUR file not found: eurusd_daily.csv
WARNING:features.macro_features:⚠️  SILVER file not found: silver_daily.csv
WARNING:features.macro_features:⚠️  GLD file not found: gld_etf_daily.csv

"""


def standardize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Return a canonical OHLCV table with columns in a stable order."""
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]

    if 'date' in df.columns:
        df = df.rename(columns={'date': 'time'})
    elif 'datetime' in df.columns:
        df = df.rename(columns={'datetime': 'time'})

    if 'adj close' in df.columns and 'close' not in df.columns:
        df['close'] = df['adj close']

    cols = ['time', 'open', 'high', 'low', 'close', 'volume']
    for col in cols:
        if col not in df.columns:
            df[col] = pd.NA

    out = df[cols].copy()
    out['time'] = pd.to_datetime(out['time'])
    return out

def fetch_data():
    print("🚀 Fetching Macro Data...")
    
    # 1. Gold Futures (Reference)
    # 2. DXY (US Dollar Index) - Critical for Gold
    # 3. SPX (S&P 500) - Risk Sentiment
    # 4. US10Y (Treasury Yields) - Opportunity Cost of Gold
    
    tickers = {
        "GC=F": "gold_futures", 
        "DX-Y.NYB": "dxy_daily", 
        "^GSPC": "spx_daily", 
        "^TNX": "us10y_daily",
        "^VIX": "vix_daily",
        "CL=F": "oil_wti_daily",
        "EURUSD=X": "eurusd_daily",
        "SI=F": "silver_daily",
        "GLD": "gld_etf_daily",
    }
    
    for symbol, name in tickers.items():
        print(f"📥 Downloading {name} ({symbol})...")
        try:
            # FORCE DAILY DATA to get long history (25 years)
            # We will forward-fill this to hourly later
            df = yf.download(symbol, period="25y", interval="1d", progress=False)

            print(df)
            
            if df is not None and not df.empty:
                # Standardize into a stable OHLCV layout so downstream code
                # sees the same column order across MT5 and macro sources.
                df = standardize_ohlcv(df)
                
                # Save
                path = f"data/{name}.csv"
                df.to_csv(path, index=False)
                print(f"✅ Saved {path} ({len(df)} rows)")
            else:
                print(f"❌ Failed to download {name}")
                
        except Exception as e:
            print(f"❌ Error downloading {name}: {e}")

if __name__ == "__main__":
    fetch_data()
