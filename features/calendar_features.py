"""
Economic Calendar Advanced Features Module

Consumes the UTC-normalized forex and metalsmine event feeds and computes
timing, impact, type, source, and surprise-based features.
"""

import pandas as pd
import numpy as np
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


DEFAULT_CALENDAR_GLOB_PATTERNS = ("forex_day_*.json", "metals_day_*.json")
DEFAULT_EVENT_WINDOW_HOURS = 2.0
DEFAULT_LOOKAHEAD_DAYS = 7.0
UNIT_FAMILIES = ("percent", "count", "rate", "index", "currency", "level", "unknown")


def _parse_number(value):
    if value is None:
        return None
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)

    text = str(value).strip()
    if not text or text.lower() in {"n/a", "na", "none", "tbd", "tba", "-"}:
        return None

    cleaned = text.replace(",", "").replace("+", "")
    cleaned = cleaned.replace("−", "-").replace("—", "-")

    # Prefer the first explicit numeric token so pipe-delimited or range-like
    # values still contribute a usable scalar.
    token_match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[kKmMbBtT%])?", cleaned)
    if not token_match:
        return None

    token = token_match.group(0)
    multiplier = 1.0
    if token[-1] in {"K", "k", "M", "m", "B", "b", "T", "t"}:
        multiplier = {"K": 1_000.0, "k": 1_000.0, "M": 1_000_000.0, "m": 1_000_000.0, "B": 1_000_000_000.0, "b": 1_000_000_000.0, "T": 1_000_000_000_000.0, "t": 1_000_000_000_000.0}[token[-1]]
        token = token[:-1]

    if token.endswith("%"):
        token = token[:-1]

    try:
        return float(token) * multiplier
    except ValueError:
        return None


def _normalize_impact(value):
    impact = str(value or "n/a").strip().upper()
    if impact in {"HIGH", "MEDIUM", "LOW"}:
        return impact
    return "N/A"


def _impact_to_score(value):
    return {
        "HIGH": 1.0,
        "MEDIUM": 0.6,
        "LOW": 0.3,
        "N/A": 0.0,
    }.get(_normalize_impact(value), 0.0)


def _infer_source_from_path(path: Path) -> str:
    name = path.stem.lower()
    if "metals" in name:
        return "metals"
    if "forex" in name:
        return "forex"
    return "unknown"


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _canonical_event_key(event_name: str) -> str:
    return re.sub(r"\s+", " ", str(event_name).strip().lower())


def _classify_unit_family(event_name: str, forecast: str | None, actual: str | None, previous: str | None) -> str:
    name = _canonical_event_key(event_name)
    raw_values = " ".join(str(value) for value in (forecast, actual, previous) if value is not None).lower()
    combined = f"{name} {raw_values}"

    if "%" in combined or _contains_any(
        combined,
        (r"\b(percent|pct|y/y|m/m|q/q|w/w|rate|inflation|cpi|ppi|pce|gdp|unemployment|yield|interest|fed funds|policy rate)\b",),
    ):
        return "percent"

    if _contains_any(
        combined,
        (r"\d(?:\.\d+)?[kKmMbBtT]", r"\b(k|m|b|t)\b", r"\b(jobs?|payrolls?|claims?|inventories?|inventory|orders?|permits?|sales?|spending|trade balance|budget|exports?|imports?|retail sales)\b"),
    ):
        return "count"

    if _contains_any(combined, (r"\b(yield|interest rate|policy rate|benchmark rate|fomc|fed rate|central bank rate)\b",)):
        return "rate"

    if _contains_any(combined, (r"\b(index|pmi|survey|confidence|sentiment|ratio)\b",)):
        return "index"

    if _contains_any(combined, (r"\b(usd|currency|revenue|turnover|profit|cost|price|prices|value|balance)\b",)):
        return "currency"

    if _contains_any(combined, (r"\b(level|close|open|high|low|spot|spot price)\b",)):
        return "level"

    return "unknown"


def _unit_family_one_hot(family: str, prefix: str) -> dict[str, float]:
    family = family if family in UNIT_FAMILIES else "unknown"
    return {f"{prefix}_unit_is_{candidate}": float(family == candidate) for candidate in UNIT_FAMILIES}


def _robust_scale(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)].astype(np.float64, copy=False)
    if finite_values.size == 0:
        return 1.0

    median = float(np.median(finite_values))
    centered = np.subtract(finite_values, np.float64(median), dtype=np.float64)  # pyright: ignore[reportOperatorIssue]
    mad = float(np.median(np.abs(centered)))
    if mad > 1e-8:
        # 1.4826 makes MAD comparable to standard deviation under normality.
        return 1.4826 * mad

    std = float(np.std(finite_values))
    if std > 1e-8:
        return std
    return 1.0


def _resolve_calendar_files(filepath):
    base_path = Path(filepath or "data")

    search_roots = []
    if base_path.exists() and base_path.is_dir():
        search_roots.append(base_path)
    else:
        search_roots.append(base_path.parent)

    economic_calendar_dir = base_path.parent / "economic_calendar"
    if economic_calendar_dir.exists():
        search_roots.append(economic_calendar_dir)

    # Prefer modern forex/metals feeds anywhere under the data tree.
    if base_path.name.startswith("economic_events"):
        sibling_files = []
        seen_paths = set()
        for root in search_roots:
            for pattern in DEFAULT_CALENDAR_GLOB_PATTERNS:
                for candidate in sorted(root.rglob(pattern)):
                    resolved = candidate.resolve()
                    if resolved in seen_paths:
                        continue
                    seen_paths.add(resolved)
                    sibling_files.append(candidate)
        if sibling_files:
            return sibling_files

    if base_path.exists() and base_path.is_file():
        return [base_path]

    if base_path.exists() and base_path.is_dir():
        files = []
        seen_paths = set()
        for pattern in DEFAULT_CALENDAR_GLOB_PATTERNS:
            for candidate in sorted(base_path.rglob(pattern)):
                resolved = candidate.resolve()
                if resolved in seen_paths:
                    continue
                seen_paths.add(resolved)
                files.append(candidate)
        if files:
            return files
        return sorted(base_path.rglob("economic_events*.json"))

    return [base_path] if base_path.suffix.lower() == ".json" else []


def _normalize_calendar_event(event, source=None):
    if not isinstance(event, dict):
        return None

    time_text = event.get("event_time_utc") or event.get("datetime") or event.get("time")
    if time_text is None:
        return None

    timestamp = pd.Timestamp(str(time_text))
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    if pd.isna(timestamp):
        return None

    normalized = {
        "time": timestamp.tz_convert(None),
        "event": str(event.get("event", "")),
        "event_key": _canonical_event_key(str(event.get("event", ""))),
        "unit_family": _classify_unit_family(str(event.get("event", "")), event.get("forecast"), event.get("actual"), event.get("previous")),
        "impact": _normalize_impact(event.get("impact")),
        "forecast": event.get("forecast"),
        "actual": event.get("actual"),
        "previous": event.get("previous"),
        "forecast_value": _parse_number(event.get("forecast")),
        "actual_value": _parse_number(event.get("actual")),
        "previous_value": _parse_number(event.get("previous")),
        "source": source or event.get("source") or "unknown",
    }

    if "currency" in event:
        normalized["currency"] = event.get("currency")
    if "country" in event:
        normalized["country"] = event.get("country")

    return normalized


def _dedupe_calendar_events(events):
    seen = set()
    deduped = []
    for event in events:
        key = (
            event.get("time"),
            event.get("event"),
            event.get("impact"),
            event.get("forecast"),
            event.get("actual"),
            event.get("previous"),
            event.get("source"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(event)
    return deduped


def load_economic_calendar(filepath='data/economic_events_2015_2025.json'):
    """
    Load economic calendar events from one file, a calendar directory, or the
    repo's combined forex/metals feed set.

    The loader accepts the new UTC-normalized scrape format and also keeps the
    old generated calendar as a fallback.
    """
    logger.info(f"📅 Loading economic calendar from {filepath}...")

    resolved_files = _resolve_calendar_files(filepath)
    if not resolved_files:
        logger.warning(f"⚠️  Calendar file not found: {filepath}")
        return []

    events = []
    for file_path in resolved_files:
        if not file_path.exists():
            continue

        try:
            with open(file_path, 'r', encoding='utf-8') as handle:
                loaded = json.load(handle)
        except Exception as exc:
            logger.warning(f"⚠️  Could not load calendar file {file_path}: {exc}")
            continue

        if not isinstance(loaded, list):
            logger.warning(f"⚠️  Ignoring non-list calendar file: {file_path}")
            continue

        source = _infer_source_from_path(file_path)
        for event in loaded:
            normalized = _normalize_calendar_event(event, source=source)
            if normalized is not None:
                events.append(normalized)

    if not events:
        logger.warning("⚠️  No usable calendar events found")
        return []

    events = _dedupe_calendar_events(events)
    events.sort(key=lambda event: (event['time'], event.get('source', ''), event.get('event', '')))

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
    Compute calendar-based features from the combined forex and metals feeds.

    Returns 14 features:
    - timing: hours_to_event, days_since_event, event_density
    - impact: is_high_impact, in_event_window, event_volatility_expected
    - event type: event_type_nfp, event_type_fomc, event_type_cpi, event_type_gdp
    - source: next_event_is_forex, next_event_is_metals
    - release surprise: last_event_surprise_norm, last_event_revision_norm
    """
    logger.info("="*70)
    logger.info("📅 COMPUTING ECONOMIC CALENDAR FEATURES")
    logger.info("="*70)

    feature_names = [
        'hours_to_event',
        'days_since_event',
        'event_density',
        'is_high_impact',
        'in_event_window',
        'event_volatility_expected',
        'event_type_nfp',
        'event_type_fomc',
        'event_type_cpi',
        'event_type_gdp',
        'next_event_is_forex',
        'next_event_is_metals',
        'last_event_surprise_norm',
        'last_event_revision_norm',
    ]

    for prefix in ('next', 'last'):
        for candidate in UNIT_FAMILIES:
            feature_names.append(f'{prefix}_unit_is_{candidate}')

    result = pd.DataFrame(index=pd.DatetimeIndex(df_timestamps), columns=feature_names, dtype=np.float32)

    if not calendar:
        logger.warning("⚠️  No calendar data available, filling with neutral defaults")
        result['hours_to_event'] = 168.0 / 168.0
        result['days_since_event'] = 7.0 / 30.0
        result['event_density'] = 0.0
        result['is_high_impact'] = 0.0
        result['in_event_window'] = 0.0
        result['event_volatility_expected'] = 1.0
        result['event_type_nfp'] = 0.0
        result['event_type_fomc'] = 0.0
        result['event_type_cpi'] = 0.0
        result['event_type_gdp'] = 0.0
        result['next_event_is_forex'] = 0.0
        result['next_event_is_metals'] = 0.0
        result['last_event_surprise_norm'] = 0.0
        result['last_event_revision_norm'] = 0.0
        for prefix in ('next', 'last'):
            for candidate in UNIT_FAMILIES:
                result[f'{prefix}_unit_is_{candidate}'] = 0.0
        return result.fillna(0.0)

    logger.info(f"Processing {len(df_timestamps):,} timestamps...")

    timestamps = pd.DatetimeIndex(df_timestamps)
    if timestamps.tz is not None:
        timestamps = timestamps.tz_convert('UTC').tz_localize(None)

    event_times = pd.DatetimeIndex([pd.Timestamp(str(event['time'])) for event in calendar if event.get('time') is not None])
    if event_times.tz is not None:
        event_times = event_times.tz_convert('UTC').tz_localize(None)

    event_times_ns = event_times.to_numpy(dtype='datetime64[ns]').astype('int64')
    event_names = np.asarray([str(event.get('event', '')) for event in calendar], dtype=object)
    event_impacts = np.asarray([_normalize_impact(event.get('impact', 'N/A')) for event in calendar], dtype=object)
    event_sources = np.asarray([str(event.get('source', 'unknown')).lower() for event in calendar], dtype=object)
    event_keys = np.asarray([str(event.get('event_key', _canonical_event_key(event.get('event', '')))) for event in calendar], dtype=object)
    event_unit_families = np.asarray([str(event.get('unit_family', 'unknown')) for event in calendar], dtype=object)
    forecast_values = np.asarray([
        np.nan if event.get('forecast_value') is None else float(event.get('forecast_value'))
        for event in calendar
    ], dtype=np.float64)
    actual_values = np.asarray([
        np.nan if event.get('actual_value') is None else float(event.get('actual_value'))
        for event in calendar
    ], dtype=np.float64)
    previous_values = np.asarray([
        np.nan if event.get('previous_value') is None else float(event.get('previous_value'))
        for event in calendar
    ], dtype=np.float64)

    ts_ns = timestamps.to_numpy(dtype='datetime64[ns]').astype('int64')
    one_week_ns = int(pd.Timedelta(days=DEFAULT_LOOKAHEAD_DAYS).value)

    next_idx = np.searchsorted(event_times_ns, ts_ns, side='right').astype(np.int64, copy=False)
    valid_next = next_idx < len(event_times_ns)
    clipped_next = np.clip(next_idx, 0, max(len(event_times_ns) - 1, 0))

    prev_idx = np.subtract(next_idx, np.int64(1), dtype=np.int64)  # pyright: ignore[reportOperatorIssue]
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

    next_impacts = np.full(len(ts_ns), 'N/A', dtype=object)
    next_names = np.full(len(ts_ns), '', dtype=object)
    next_sources = np.full(len(ts_ns), 'unknown', dtype=object)
    next_unit_families = np.full(len(ts_ns), 'unknown', dtype=object)
    if len(event_times_ns) > 0:
        next_impacts[valid_next] = event_impacts[clipped_next][valid_next]
        next_names[valid_next] = event_names[clipped_next][valid_next]
        next_sources[valid_next] = event_sources[clipped_next][valid_next]
        next_unit_families[valid_next] = event_unit_families[clipped_next][valid_next]

    is_high_impact = (next_impacts == 'HIGH').astype(np.float32)
    in_event_window = np.where(valid_next, (hours_to_event <= DEFAULT_EVENT_WINDOW_HOURS).astype(np.float32), 0.0)
    event_volatility_expected = np.select(
        [next_impacts == 'HIGH', next_impacts == 'MEDIUM', next_impacts == 'LOW'],
        [2.0, 1.5, 1.2],
        default=1.0,
    ).astype(np.float32)

    event_type_nfp = np.asarray([_contains_any(str(name), ('NFP', 'NONFARM')) for name in next_names], dtype=np.float32)
    event_type_fomc = np.asarray([_contains_any(str(name), ('FOMC', 'FEDERAL RESERVE', 'FED CHAIR')) for name in next_names], dtype=np.float32)
    event_type_cpi = np.asarray([_contains_any(str(name), ('CPI', 'INFLATION')) for name in next_names], dtype=np.float32)
    event_type_gdp = np.asarray([_contains_any(str(name), ('GDP',)) for name in next_names], dtype=np.float32)

    next_event_is_forex = (next_sources == 'forex').astype(np.float32)
    next_event_is_metals = (next_sources == 'metals').astype(np.float32)
    next_unit_features = _unit_family_one_hot('unknown', 'next')
    for candidate in UNIT_FAMILIES:
        next_unit_features[f'next_unit_is_{candidate}'] = (next_unit_families == candidate).astype(np.float32)

    raw_surprise_by_event = np.where(
        np.isfinite(actual_values) & np.isfinite(forecast_values),
        actual_values - forecast_values,
        np.nan,
    )
    raw_revision_by_event = np.where(
        np.isfinite(actual_values) & np.isfinite(previous_values),
        actual_values - previous_values,
        np.nan,
    )

    surprise_std_by_event = np.zeros(len(calendar), dtype=np.float64)
    revision_std_by_event = np.zeros(len(calendar), dtype=np.float64)

    for event_key in np.unique(event_keys):
        key_mask = event_keys == event_key

        key_surprise = raw_surprise_by_event[key_mask]
        if np.isfinite(key_surprise).any():
            scale = _robust_scale(key_surprise)
            scaled = np.divide(key_surprise, scale, out=np.zeros_like(key_surprise), where=np.isfinite(key_surprise))
            surprise_std_by_event[key_mask] = np.clip(scaled, -10.0, 10.0)

        key_revision = raw_revision_by_event[key_mask]
        if np.isfinite(key_revision).any():
            scale = _robust_scale(key_revision)
            scaled = np.divide(key_revision, scale, out=np.zeros_like(key_revision), where=np.isfinite(key_revision))
            revision_std_by_event[key_mask] = np.clip(scaled, -10.0, 10.0)

    last_surprise_std = np.zeros(len(ts_ns), dtype=np.float64)
    last_revision_std = np.zeros(len(ts_ns), dtype=np.float64)
    last_unit_families = np.full(len(ts_ns), 'unknown', dtype=object)
    if len(event_times_ns) > 0:
        last_surprise_std[valid_prev] = surprise_std_by_event[clipped_prev][valid_prev]
        last_revision_std[valid_prev] = revision_std_by_event[clipped_prev][valid_prev]
        last_unit_families[valid_prev] = event_unit_families[clipped_prev][valid_prev]

    last_unit_features = _unit_family_one_hot('unknown', 'last')
    for candidate in UNIT_FAMILIES:
        last_unit_features[f'last_unit_is_{candidate}'] = (last_unit_families == candidate).astype(np.float32)

    result['hours_to_event'] = hours_to_event / 168.0
    result['days_since_event'] = days_since_event / 30.0
    result['event_density'] = event_density / 10.0
    result['is_high_impact'] = is_high_impact
    result['in_event_window'] = in_event_window
    result['event_volatility_expected'] = event_volatility_expected
    result['event_type_nfp'] = event_type_nfp
    result['event_type_fomc'] = event_type_fomc
    result['event_type_cpi'] = event_type_cpi
    result['event_type_gdp'] = event_type_gdp
    result['next_event_is_forex'] = next_event_is_forex
    result['next_event_is_metals'] = next_event_is_metals
    result['last_event_surprise_norm'] = np.clip(last_surprise_std, -10.0, 10.0).astype(np.float32)
    result['last_event_revision_norm'] = np.clip(last_revision_std, -10.0, 10.0).astype(np.float32)
    for candidate in UNIT_FAMILIES:
        result[f'next_unit_is_{candidate}'] = next_unit_features[f'next_unit_is_{candidate}']
        result[f'last_unit_is_{candidate}'] = last_unit_features[f'last_unit_is_{candidate}']

    result = result.fillna(0.0).astype(np.float32)

    logger.info("\n" + "="*70)
    logger.info("✅ CALENDAR FEATURES COMPLETE")
    logger.info("="*70)
    logger.info(f"✅ Generated {result.shape[1]} calendar features")
    logger.info(f"✅ Processed {len(result):,} timestamps")

    high_impact_count = result['is_high_impact'].sum()
    event_window_count = result['in_event_window'].sum()
    logger.info(f"\n📊 Calendar statistics:")
    logger.info(f"   • High impact events ahead: {int(high_impact_count):,} timestamps")
    logger.info(f"   • In event window (±{DEFAULT_EVENT_WINDOW_HOURS:g}h): {int(event_window_count):,} timestamps")

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
