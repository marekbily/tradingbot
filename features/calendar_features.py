"""
Economic Calendar Advanced Features Module

Computes 8 features from economic events:
- Event Timing (3): hours to event, days since event, event density
- Event Impact (3): is high impact, in event window, expected volatility
- Event Type (2): NFP detection, FOMC detection

These features make the AI aware of major economic releases and their impact.
"""

import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_economic_calendar(filepath='data/economic_events_2015_2025.json'):
    """
    Load economic calendar from JSON file

    Returns:
        List of event dicts with keys: time, event, impact
    """
    logger.info(f"📅 Loading economic calendar from {filepath}...")

    filepath = Path(filepath)

    if not filepath.exists():
        logger.warning(f"⚠️  Calendar file not found: {filepath}")
        return []

    with open(filepath, 'r') as f:
        events = json.load(f)

    # Convert datetime strings to normalized timestamps and rename to 'time'
    for event in events:
        event['time'] = pd.to_datetime(event['datetime'], utc=True).tz_localize(None)

    # Keep events sorted so downstream searchsorted lookups are valid.
    events.sort(key=lambda event: event['time'])

    logger.info(f"   ✅ Loaded {len(events)} economic events")

    return events


def find_next_event(timestamp, events):
    """
    Find the next economic event after given timestamp

    Args:
        timestamp: Current time
        events: List of event dicts

    Returns:
        Dict with next event info, or None if no future events
    """
    future_events = [e for e in events if e['time'] > timestamp]

    if not future_events:
        return None

    # Return the nearest future event
    return min(future_events, key=lambda e: e['time'])


def find_last_event(timestamp, events):
    """
    Find the most recent economic event before given timestamp

    Args:
        timestamp: Current time
        events: List of event dicts

    Returns:
        Dict with last event info, or None if no past events
    """
    past_events = [e for e in events if e['time'] <= timestamp]

    if not past_events:
        return None

    # Return the most recent past event
    return max(past_events, key=lambda e: e['time'])


def count_upcoming_events(timestamp, events, days=7):
    """
    Count events in the next N days

    Args:
        timestamp: Current time
        events: List of event dicts
        days: Number of days to look ahead

    Returns:
        Count of upcoming events
    """
    future_time = timestamp + timedelta(days=days)
    upcoming = [e for e in events if timestamp < e['time'] <= future_time]

    return len(upcoming)


def compute_calendar_features(df_timestamps, calendar):
    """
    Compute 8 calendar-based features

    Args:
        df_timestamps: DataFrame with DatetimeIndex (from gold data)
        calendar: List of event dicts from load_economic_calendar()

    Returns:
        DataFrame with 8 calendar features
    """
    logger.info("="*70)
    logger.info("📅 COMPUTING ECONOMIC CALENDAR FEATURES")
    logger.info("="*70)

    result = pd.DataFrame(index=df_timestamps)

    if not calendar:
        logger.warning("⚠️  No calendar data available, filling with zeros")
        result['hours_to_event'] = 168.0  # 1 week default
        result['days_since_event'] = 7.0
        result['event_density'] = 0.0
        result['is_high_impact'] = 0.0
        result['in_event_window'] = 0.0
        result['event_volatility_expected'] = 1.0
        result['event_type_nfp'] = 0.0
        result['event_type_fomc'] = 0.0
        return result

    logger.info(f"Processing {len(df_timestamps):,} timestamps...")

    timestamps = pd.DatetimeIndex(df_timestamps)
    if timestamps.tz is not None:
        timestamps = timestamps.tz_convert('UTC').tz_localize(None)

    event_times = pd.DatetimeIndex(pd.to_datetime([event['time'] for event in calendar]))
    if event_times.tz is not None:
        event_times = event_times.tz_convert('UTC').tz_localize(None)
    event_times_ns = event_times.to_numpy(dtype='datetime64[ns]').astype('int64')
    event_names = np.asarray([str(event.get('event', '')) for event in calendar], dtype=object)
    event_impacts = np.asarray([str(event.get('impact', 'MEDIUM')).upper() for event in calendar], dtype=object)

    ts_ns = timestamps.to_numpy(dtype='datetime64[ns]').astype('int64')
    one_week_ns = int(pd.Timedelta(days=7).value)

    next_idx = np.searchsorted(event_times_ns, ts_ns, side='right')
    valid_next = next_idx < len(event_times_ns)
    clipped_next = np.clip(next_idx, 0, max(len(event_times_ns) - 1, 0))

    prev_idx = next_idx - 1
    valid_prev = prev_idx >= 0
    clipped_prev = np.clip(prev_idx, 0, max(len(event_times_ns) - 1, 0))

    hours_to_event = np.full(len(ts_ns), 168.0, dtype=np.float32)
    if len(event_times_ns) > 0:
        next_deltas_hours = (event_times_ns[clipped_next] - ts_ns) / 3_600_000_000_000.0
        hours_to_event[valid_next] = np.minimum(next_deltas_hours[valid_next], 168.0)

    days_since_event = np.full(len(ts_ns), 30.0, dtype=np.float32)
    if len(event_times_ns) > 0:
        prev_deltas_days = (ts_ns - event_times_ns[clipped_prev]) / 86_400_000_000_000.0
        days_since_event[valid_prev] = np.minimum(prev_deltas_days[valid_prev], 30.0)

    upper_idx = np.searchsorted(event_times_ns, ts_ns + one_week_ns, side='right')
    event_density = np.minimum((upper_idx - next_idx).astype(np.float32), 10.0)

    next_impacts = np.full(len(ts_ns), 'MEDIUM', dtype=object)
    next_names = np.full(len(ts_ns), '', dtype=object)
    if len(event_times_ns) > 0:
        next_impacts[valid_next] = event_impacts[clipped_next][valid_next]
        next_names[valid_next] = event_names[clipped_next][valid_next]

    is_high_impact = (next_impacts == 'HIGH').astype(np.float32)
    in_event_window = np.where(valid_next, (hours_to_event <= 2.0).astype(np.float32), 0.0)
    event_volatility_expected = np.select(
        [next_impacts == 'HIGH', next_impacts == 'MEDIUM'],
        [2.0, 1.5],
        default=1.0,
    ).astype(np.float32)

    next_name_series = pd.Series(next_names, index=timestamps)
    event_type_nfp = next_name_series.str.contains('NFP|NONFARM', case=False, na=False, regex=True).astype(np.float32).to_numpy()
    event_type_fomc = next_name_series.str.contains('FOMC|FEDERAL RESERVE', case=False, na=False, regex=True).astype(np.float32).to_numpy()

    result['hours_to_event'] = hours_to_event / 168.0
    result['days_since_event'] = days_since_event / 30.0
    result['event_density'] = event_density / 10.0
    result['is_high_impact'] = is_high_impact
    result['in_event_window'] = in_event_window
    result['event_volatility_expected'] = event_volatility_expected
    result['event_type_nfp'] = event_type_nfp
    result['event_type_fomc'] = event_type_fomc

    # Fill any NaNs
    result = result.fillna(0.0)

    # Summary
    logger.info("\n" + "="*70)
    logger.info("✅ CALENDAR FEATURES COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Generated {result.shape[1]} calendar features")
    logger.info(f"✅ Processed {len(result):,} timestamps")

    # Statistics
    high_impact_count = result['is_high_impact'].sum()
    event_window_count = result['in_event_window'].sum()

    logger.info(f"\n📊 Calendar statistics:")
    logger.info(f"   • High impact events ahead: {int(high_impact_count):,} timestamps")
    logger.info(f"   • In event window (±2h): {int(event_window_count):,} timestamps")

    # List features
    logger.info("\n📊 Features created:")
    for col in result.columns:
        logger.info(f"   • {col}")

    return result


def test_calendar_features():
    """
    Test function to verify calendar features work correctly
    """
    logger.info("\n" + "="*70)
    logger.info("🧪 TESTING CALENDAR FEATURES")
    logger.info("="*70)

    try:
        # Load calendar
        logger.info("\n1️⃣ Loading economic calendar...")
        calendar = load_economic_calendar()

        # Load gold data for timestamps
        logger.info("\n2️⃣ Loading gold data for timestamps...")
        from data.load_data import load_ohlc_csv

        df_gold = load_ohlc_csv('data/xauusd_m5.csv')
        df_gold = df_gold.set_index('time').sort_index()

        # Take a subset for testing (first 10k bars)
        df_gold_subset = df_gold.head(10000)

        logger.info("\n3️⃣ Computing calendar features...")
        calendar_features = compute_calendar_features(df_gold_subset.index, calendar)

        logger.info("\n✅ Calendar features computed successfully!")

        # Check for NaNs
        nan_count = calendar_features.isna().sum().sum()
        if nan_count > 0:
            logger.warning(f"⚠️  {nan_count} NaN values found")
        else:
            logger.info("✅ No NaN values")

        # Show sample
        logger.info("\n📊 Sample data:")
        logger.info(calendar_features.head(10))

        return calendar_features

    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise


if __name__ == "__main__":
    # Run test
    calendar_feat = test_calendar_features()

    logger.info("\n" + "="*70)
    logger.info("✅ CALENDAR FEATURES MODULE READY")
    logger.info("="*70)

    logger.info("""
📋 USAGE:
    from features.calendar_features import load_economic_calendar, compute_calendar_features

    # Load calendar
    calendar = load_economic_calendar()

    # Compute features (pass DataFrame index)
    calendar_features = compute_calendar_features(df.index, calendar)
    """)
