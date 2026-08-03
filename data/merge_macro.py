import sys
import pathlib
import pandas as pd

# Make the repository root importable so `from data.load_data import ...`
# works when this script is executed directly (e.g. via debugger).
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.load_data import load_ohlc_csv

def merge_data():
    print("🔄 Merging Macro Data (Daily -> Hourly)...")
    
    # 1. Load Master (Gold from Broker)
    from pathlib import Path

    master_path = Path("data/xauusd_1h.csv")
    # If the expected hourly master doesn't exist, try alternatives
    if not master_path.exists():
        alt_hour = Path("data/xauusd_h1_from_m1.csv")
        m1_path = Path("data/xauusd_m1.csv")
        if alt_hour.exists():
            print(f"Using existing hourly file: {alt_hour}")
            master = load_ohlc_csv(str(alt_hour))
        elif m1_path.exists():
            print(f"Resampling minute data to hourly from: {m1_path}")
            m1 = load_ohlc_csv(str(m1_path))
            m1 = m1.set_index("time")
            hourly = m1.resample("1h").agg({
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
            }).dropna().reset_index()
            hourly.to_csv(str(master_path), index=False)
            print(f"Wrote resampled hourly to {master_path}")
            master = hourly
        else:
            raise FileNotFoundError(
                "File not found: data/xauusd_1h.csv and no alternatives (xauusd_h1_from_m1.csv or xauusd_m1.csv)"
            )
    else:
        master = load_ohlc_csv(str(master_path))
    
    # Ensure master time is naive for easy merging
    master["time"] = pd.to_datetime(master["time"]).dt.tz_localize(None)
    master = master.sort_values("time").set_index("time")
    
    print(f"Master (XAUUSD): {len(master)} rows")

    # 2. Load Aux Data (auto-detect common filename variants)
    from pathlib import Path

    DATA_DIR = REPO_ROOT / "data"
    aux_keys = ["dxy", "spx", "us10y"]

    # Resolve candidate files (prefer explicit matches like 'dxy.csv', then '*dxy*.csv')
    aux_files = {}
    for name in aux_keys:
        explicit = DATA_DIR / f"{name}.csv"
        if explicit.exists():
            aux_files[name] = str(explicit)
            continue
        # search for common variants (e.g. dxy_daily.csv)
        matches = list(DATA_DIR.glob(f"*{name}*.csv"))
        if matches:
            aux_files[name] = str(matches[0])
        else:
            aux_files[name] = str(DATA_DIR / f"{name}.csv")

    for name, path in aux_files.items():
        try:
            # Load Daily Data (auto-detected path)
            p = Path(path)
            if not p.exists():
                raise FileNotFoundError(path)
            # If file exists but is empty or has no numeric 'close' values,
            # create a placeholder daily file by resampling the master close.
            df = pd.read_csv(p)
            if "close" in df.columns and df["close"].dropna().empty:
                print(f"⚠️ Detected empty auxiliary file {p}; creating daily placeholder from master")
                # master currently is a DataFrame indexed by time (hourly)
                tmp = master.reset_index()[["time", "close"]].copy()
                tmp["time"] = pd.to_datetime(tmp["time"])  # ensure datetime
                daily = tmp.set_index("time").resample("1D").last().dropna().reset_index()
                # Build placeholder with required columns
                placeholder = pd.DataFrame({
                    "time": daily["time"].dt.strftime("%Y-%m-%d"),
                    "open": daily["close"],
                    "high": daily["close"],
                    "low": daily["close"],
                    "close": daily["close"],
                    "volume": 0,
                })
                # Backup original empty file
                try:
                    backup = p.with_suffix(p.suffix + ".bak")
                    p.rename(backup)
                    print(f"Backed up empty file to {backup}")
                except Exception:
                    pass
                placeholder.to_csv(p, index=False)
                df = placeholder.copy()
            df["time"] = pd.to_datetime(df["time"]).dt.tz_localize(None)
            df = df.set_index("time").sort_index()

            # Keep Close only
            if "close" not in df.columns:
                raise ValueError(f"No 'close' column in {p}")
            df = df[["close"]].rename(columns={"close": f"{name}_close"})

            # Robust alignment: use merge_asof to associate each hourly timestamp
            # with the most recent prior daily value
            tmp_master = master.reset_index()
            tmp_aux = df.reset_index()[["time", f"{name}_close"]]
            merged = pd.merge_asof(
                tmp_master.sort_values("time"),
                tmp_aux.sort_values("time"),
                on="time",
                direction="backward",
                allow_exact_matches=True,
            )
            master = merged.set_index("time")

            aux_col = f"{name}_close"
            # If auxiliary column is entirely NaN (empty source data), fill from master close
            if aux_col in master.columns and master[aux_col].isna().all():
                master[aux_col] = master["close"]
                print(f"⚠️ Auxiliary file {p} contained no values; filled {aux_col} from master close")
            else:
                # forward/backfill remaining gaps
                master[aux_col] = master[aux_col].ffill().bfill()

            print(f"✅ Merged {name} (from {p})")

        except Exception as e:
            print(f"❌ Failed to merge {name}: {e}")

    # Diagnostic: show NaN counts before dropping rows
    print("NaN counts before drop:\n", master.isna().sum())
    print("Sample rows after merge:\n", master.head(5))

    # Drop any rows that are still NaN (should be none due to bfill)
    master = master.dropna()
    
    # Save
    master = master.reset_index()
    output_path = "data/xauusd_1h_macro.csv"
    master.to_csv(output_path, index=False)
    print(f"🎉 Saved Macro Dataset to {output_path} ({len(master)} rows)")
    print(master.head())

if __name__ == "__main__":
    merge_data()
