"""Download forexfactory and metalsmine data using forex-pytory.

Ability of changing the timeframe of the data to be downloaded.

Download to json or csv format.

Convert all event times to UTC. Super important for backtesting and live trading.

Keep both scheduled time and actual publish time if available.

Keep revisions (first release vs revised values).

Include surprise features like actual minus forecast and actual minus previous.

Include currency, impact level, event name, and country consistently.
"""

from __future__ import annotations

import argparse
import csv
import errno
import json
import logging
import os
import time
from importlib import import_module
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

from functools import lru_cache

from forex_pytory.core.scraper import forex_factory_scraper, metals_mine_scraper

logger = logging.getLogger(__name__)

SOURCE_SCRAPERS = {
	"forex": forex_factory_scraper,
	"metals": metals_mine_scraper,
}

DEFAULT_SOURCE_TIMEZONE = "auto"
DEFAULT_OUTPUT_DIR = Path("economic_calendar")
DEFAULT_BLOCK_COOLDOWN_SECONDS = 30
DEFAULT_SCRAPE_RETRIES = 10

COUNTRY_BY_CURRENCY = {
	"USD": "United States",
	"EUR": "Eurozone",
	"GBP": "United Kingdom",
	"JPY": "Japan",
	"CAD": "Canada",
	"AUD": "Australia",
	"NZD": "New Zealand",
	"CHF": "Switzerland",
	"CNY": "China",
	"SEK": "Sweden",
	"NOK": "Norway",
	"DKK": "Denmark",
	"PLN": "Poland",
	"CZK": "Czech Republic",
	"HUF": "Hungary",
	"TRY": "Turkey",
	"ZAR": "South Africa",
	"SGD": "Singapore",
	"HKD": "Hong Kong",
	"MXN": "Mexico",
	"BRL": "Brazil",
	"INR": "India",
	"KRW": "South Korea",
	"THB": "Thailand",
	"PHP": "Philippines",
	"RUB": "Russia",
	"ILS": "Israel",
	"AED": "United Arab Emirates",
	"SAR": "Saudi Arabia",
}


@dataclass(frozen=True)
class DownloadResult:
	"""Paths produced by a download run."""

	json_path: Path | None
	csv_path: Path | None
	row_count: int


def _dedupe_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
	seen: set[tuple[Any, ...]] = set()
	deduped: list[dict[str, Any]] = []
	for record in records:
		key = (
			record.get("event_time_utc"),
			record.get("event"),
			record.get("impact"),
			record.get("forecast"),
			record.get("actual"),
			record.get("previous"),
		)
		if key in seen:
			continue
		seen.add(key)
		deduped.append(record)
	return deduped


def _build_output_paths(
	source: str,
	timeline: str,
	stamp: str,
	output_format: str,
	output_dir: str | os.PathLike[str],
) -> tuple[Path | None, Path | None]:
	output_dir_path = Path(output_dir)
	base_name = f"{source}_{timeline}_{stamp}"
	json_path = output_dir_path / f"{base_name}.json" if output_format in {"json", "both"} else None
	csv_path = output_dir_path / f"{base_name}.csv" if output_format in {"csv", "both"} else None
	return json_path, csv_path


def _parse_date(date_text: str | None) -> datetime:
	if not date_text:
		return datetime.now()
	return datetime.strptime(date_text, "%Y-%m-%d")


def _format_date_key(value: datetime) -> str:
	return value.strftime("%Y%m%d")


def _iter_dates(start_date: datetime, end_date: datetime):
	current = start_date.date()
	last = end_date.date()
	while current <= last:
		yield datetime.combine(current, datetime.min.time())
		current += timedelta(days=1)


def _parse_number(value: Any) -> float | None:
	if value is None:
		return None
	if isinstance(value, (int, float)):
		return float(value)

	text = str(value).strip()
	if not text or text.lower() in {"n/a", "na", "none", "tbd", "tba", "-"}:
		return None

	cleaned = text.replace(",", "").replace("%", "").replace("+", "")
	cleaned = cleaned.replace("−", "-").replace("—", "-")

	try:
		return float(cleaned)
	except ValueError:
		return None


def _currency_to_country(currency: str | None) -> str | None:
	if not currency:
		return None
	return COUNTRY_BY_CURRENCY.get(currency.upper())


def _infer_timezone(tz_name: str):
	if tz_name == "auto":
		return _detect_local_timezone()
	try:
		from zoneinfo import ZoneInfo

		return ZoneInfo(tz_name)
	except Exception:
		logger.warning("Falling back to UTC because timezone '%s' is unavailable", tz_name)
		return timezone.utc


@lru_cache(maxsize=1)
def _detect_local_timezone():
	"""Best-effort detection of the machine's local timezone."""

	try:
		get_localzone = import_module("tzlocal").get_localzone
		return get_localzone()
	except Exception:
		local_tz = datetime.now().astimezone().tzinfo
		if local_tz is not None:
			return local_tz
		logger.warning("Could not detect local timezone; falling back to UTC")
		return timezone.utc


def _to_utc_timestamp(time_text: str | None, source_timezone: str) -> str | None:
	if not time_text:
		return None

	local_dt: datetime | None = None
	for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S"):
		try:
			local_dt = datetime.strptime(time_text, fmt)
			break
		except ValueError:
			continue

	if local_dt is None:
		try:
			local_dt = datetime.fromisoformat(time_text)
		except ValueError:
			return None

	tzinfo = _infer_timezone(source_timezone)
	if local_dt.tzinfo is None:
		local_dt = local_dt.replace(tzinfo=tzinfo)
	else:
		local_dt = local_dt.astimezone(tzinfo)

	return local_dt.astimezone(timezone.utc).isoformat()


def _record_to_dict(record: Any, source: str, source_timezone: str) -> dict[str, Any]:
	if hasattr(record, "model_dump"):
		data = record.model_dump(by_alias=True)
	elif isinstance(record, dict):
		data = dict(record)
	else:
		data = dict(record)

	currency = data.get("Currency")
	event_time_local = data.get("Time")
	event_time_utc = _to_utc_timestamp(event_time_local, source_timezone)

	forecast = _parse_number(data.get("Forecast"))
	actual = _parse_number(data.get("Actual"))
	previous = _parse_number(data.get("Previous"))

	output = {
		"event_time_utc": event_time_utc,
		"event": data.get("Event"),
		"impact": data.get("Impact"),
		"forecast": data.get("Forecast"),
		"actual": data.get("Actual"),
		"previous": data.get("Previous"),
	}

	return output


def _is_transient_block_error(exc: Exception) -> bool:
	message = str(exc).lower()
	status_code = getattr(exc, "status_code", None)
	response = getattr(exc, "response", None)
	response_status_code = getattr(response, "status_code", None) if response is not None else None
	errno_value = getattr(exc, "errno", None)
	if errno_value in {errno.EMFILE, errno.ENFILE}:
		return True

	if status_code in {403, 408, 429, 502, 503, 504}:
		return True
	if response_status_code in {403, 408, 429, 502, 503, 504}:
		return True
	return any(
		marker in message
		for marker in (
			"403",
			"forbidden",
			"too many requests",
			"rate limit",
			"too many open files",
			"oserror(24)",
			"cloudflare",
			"captcha",
			"temporarily unavailable",
		)
	)


def _scrape_calendar_with_retry(
	source: str,
	date_text: str,
	timeline: str,
	source_timezone: str,
	max_retries: int = DEFAULT_SCRAPE_RETRIES,
	block_cooldown_seconds: int = DEFAULT_BLOCK_COOLDOWN_SECONDS,
) -> list[dict[str, Any]]:
	for attempt in range(1, max_retries + 1):
		try:
			print(f"Scrape attempt {attempt}/{max_retries} for {date_text}")
			return scrape_calendar(
				source=source,
				date_text=date_text,
				timeline=timeline,
				source_timezone=source_timezone,
			)
		except Exception as exc:
			is_transient_block = _is_transient_block_error(exc)
			if attempt < max_retries:
				if is_transient_block:
					print(
						f"Failsafe engaged for {date_text} after attempt {attempt}/{max_retries}; "
						f"cooling down {block_cooldown_seconds}s because the VPS or site is temporarily blocked"
					)
					logger.warning(
						"Temporary block, rate limit, or file-descriptor exhaustion while scraping %s on %s; sleeping %s seconds before retry %s/%s",
						source,
						date_text,
						block_cooldown_seconds,
						attempt + 1,
						max_retries,
					)
				else:
					print(
						f"Scrape retry for {date_text} after attempt {attempt}/{max_retries}; "
						f"sleeping {block_cooldown_seconds}s before retry"
					)
					logger.warning(
						"Scrape error while fetching %s on %s; sleeping %s seconds before retry %s/%s: %s",
						source,
						date_text,
						block_cooldown_seconds,
						attempt + 1,
						max_retries,
						exc,
					)
				time.sleep(block_cooldown_seconds)
				continue

			print(f"Retry exhausted for {date_text}; skipping this day for now and continuing")
			logger.error("Giving up on %s after %s attempts: %s", date_text, max_retries, exc)
			return []

	return []


def scrape_calendar(
	source: str = "forex",
	date_text: str | None = None,
	timeline: str = "day",
	source_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> list[dict[str, Any]]:
	"""Scrape a source calendar and return normalized event dictionaries."""

	if source not in SOURCE_SCRAPERS:
		raise ValueError(f"Unsupported source '{source}'. Choose from: {', '.join(SOURCE_SCRAPERS)}")

	scraper = SOURCE_SCRAPERS[source]
	dt_value = _parse_date(date_text)
	url = scraper.get_url(
		day=dt_value.day,
		month=dt_value.month,
		year=dt_value.year,
		timeline=timeline,
	)
	records = scraper.get_records(url)
	return [_record_to_dict(record, source=source, source_timezone=source_timezone) for record in records]


def scrape_calendar_range(
	source: str = "forex",
	start_date_text: str | None = None,
	end_date_text: str | None = None,
	timeline: str = "day",
	source_timezone: str = DEFAULT_SOURCE_TIMEZONE,
	checkpoint_json_path: Path | None = None,
) -> list[dict[str, Any]]:
	"""Scrape a date range, inclusive, and return normalized event dictionaries."""

	if not start_date_text or not end_date_text:
		raise ValueError("Both start_date_text and end_date_text are required for range scraping")

	start_date = _parse_date(start_date_text)
	end_date = _parse_date(end_date_text)
	if start_date > end_date:
		raise ValueError("start_date must be earlier than or equal to end_date")

	all_records: list[dict[str, Any]] = []
	for day_value in _iter_dates(start_date, end_date):
		date_text = day_value.strftime("%Y-%m-%d")
		try:
			day_records = _scrape_calendar_with_retry(
				source=source,
				date_text=date_text,
				timeline=timeline,
				source_timezone=source_timezone,
			)
		except Exception as exc:
			logger.error("Failed to scrape %s after retries: %s", date_text, exc)
			continue

		all_records.extend(day_records)
		print(f"Successfully scraped {date_text}")

		if checkpoint_json_path:
			next_day = day_value + timedelta(days=1)
			is_month_boundary = next_day.month != day_value.month
			is_end_of_range = day_value.date() == end_date.date()
			if is_month_boundary or is_end_of_range:
				checkpoint_records = _dedupe_records(all_records)
				_write_json(checkpoint_records, checkpoint_json_path)
				print(
					f"Monthly checkpoint saved to {checkpoint_json_path} "
					f"after {day_value:%Y-%m} ({len(checkpoint_records)} rows)"
				)

	return _dedupe_records(all_records)


def download_calendar_both_interleaved(
	start_date_text: str,
	end_date_text: str,
	timeline: str = "day",
	output_format: str = "json",
	output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
	source_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> dict[str, DownloadResult]:
	"""Scrape forex+metals day-by-day in an interleaved order with monthly checkpoints."""

	start_date = _parse_date(start_date_text)
	end_date = _parse_date(end_date_text)
	if start_date > end_date:
		raise ValueError("start_date must be earlier than or equal to end_date")

	stamp = f"{_format_date_key(start_date)}_to_{_format_date_key(end_date)}"
	sources = ["forex", "metals"]
	records_by_source: dict[str, list[dict[str, Any]]] = {source: [] for source in sources}
	paths_by_source: dict[str, tuple[Path | None, Path | None]] = {
		source: _build_output_paths(
			source=source,
			timeline=timeline,
			stamp=stamp,
			output_format=output_format,
			output_dir=output_dir,
		)
		for source in sources
	}

	for day_value in _iter_dates(start_date, end_date):
		date_text = day_value.strftime("%Y-%m-%d")
		for source in sources:
			day_records = _scrape_calendar_with_retry(
				source=source,
				date_text=date_text,
				timeline=timeline,
				source_timezone=source_timezone,
			)
			records_by_source[source].extend(day_records)
			print(f"Successfully scraped {source} {date_text}")

		next_day = day_value + timedelta(days=1)
		is_month_boundary = next_day.month != day_value.month
		is_end_of_range = day_value.date() == end_date.date()
		if is_month_boundary or is_end_of_range:
			for source in sources:
				records_by_source[source] = _dedupe_records(records_by_source[source])
				json_path, _ = paths_by_source[source]
				if json_path is not None:
					_write_json(records_by_source[source], json_path)
					print(
						f"Monthly checkpoint saved to {json_path} "
						f"after {day_value:%Y-%m} ({len(records_by_source[source])} rows)"
					)

	results: dict[str, DownloadResult] = {}
	for source in sources:
		records = _dedupe_records(records_by_source[source])
		json_path, csv_path = paths_by_source[source]

		if output_format in {"json", "both"}:
			if json_path is None:
				raise RuntimeError("Internal error: JSON output path is not set")
			json_path = _write_json(records, json_path)
		if output_format in {"csv", "both"}:
			if csv_path is None:
				raise RuntimeError("Internal error: CSV output path is not set")
			csv_path = _write_csv(records, csv_path)

		results[source] = DownloadResult(json_path=json_path, csv_path=csv_path, row_count=len(records))

	return results


def _write_json(records: list[dict[str, Any]], output_path: Path) -> Path:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	with output_path.open("w", encoding="utf-8") as handle:
		json.dump(records, handle, ensure_ascii=False, indent=2, default=str)
	return output_path


def _write_csv(records: list[dict[str, Any]], output_path: Path) -> Path:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	if not records:
		with output_path.open("w", encoding="utf-8", newline=""):
			pass
		return output_path

	fieldnames = sorted({key for record in records for key in record.keys()})
	with output_path.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		for record in records:
			row = dict(record)
			if "raw" in row and isinstance(row["raw"], (dict, list)):
				row["raw"] = json.dumps(row["raw"], ensure_ascii=False, default=str)
			writer.writerow(row)
	return output_path


def download_calendar(
	source: str = "forex",
	date_text: str | None = None,
	start_date_text: str | None = None,
	end_date_text: str | None = None,
	timeline: str = "day",
	output_format: str = "json",
	output_dir: str | os.PathLike[str] = DEFAULT_OUTPUT_DIR,
	source_timezone: str = DEFAULT_SOURCE_TIMEZONE,
) -> DownloadResult:
	"""Download calendar data and persist it to disk."""
	if source not in SOURCE_SCRAPERS:
		raise ValueError(f"Unsupported source '{source}'. Choose from: {', '.join(SOURCE_SCRAPERS)}")

	using_range = bool(start_date_text or end_date_text)
	if using_range:
		start_stamp = _format_date_key(_parse_date(start_date_text))
		end_stamp = _format_date_key(_parse_date(end_date_text))
		stamp = f"{start_stamp}_to_{end_stamp}"
		json_path, csv_path = _build_output_paths(
			source=source,
			timeline=timeline,
			stamp=stamp,
			output_format=output_format,
			output_dir=output_dir,
		)

		records = scrape_calendar_range(
			source=source,
			start_date_text=start_date_text,
			end_date_text=end_date_text,
			timeline=timeline,
			source_timezone=source_timezone,
			checkpoint_json_path=json_path,
		)
	else:
		date_text = _parse_date(date_text).strftime("%Y-%m-%d")
		stamp = _format_date_key(_parse_date(date_text))
		json_path, csv_path = _build_output_paths(
			source=source,
			timeline=timeline,
			stamp=stamp,
			output_format=output_format,
			output_dir=output_dir,
		)

		records = _scrape_calendar_with_retry(
			source=source,
			date_text=date_text,
			timeline=timeline,
			source_timezone=source_timezone,
		)

	if output_format in {"json", "both"}:
		if json_path is None:
			raise RuntimeError("Internal error: JSON output path is not set")
		json_path = _write_json(records, json_path)
	if output_format in {"csv", "both"}:
		if csv_path is None:
			raise RuntimeError("Internal error: CSV output path is not set")
		csv_path = _write_csv(records, csv_path)

	return DownloadResult(json_path=json_path, csv_path=csv_path, row_count=len(records))


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Download economic calendar data.")
	parser.add_argument("--source", choices=[*sorted(SOURCE_SCRAPERS), "both"], default="forex")
	parser.add_argument("--date", help="Date in YYYY-MM-DD format. Defaults to today.")
	parser.add_argument("--start-date", dest="start_date", help="Start date in YYYY-MM-DD format, inclusive.")
	parser.add_argument("--end-date", dest="end_date", help="End date in YYYY-MM-DD format, inclusive.")
	parser.add_argument("--timeline", default="day", help="Timeline passed to the upstream calendar URL.")
	parser.add_argument("--format", choices=["json", "csv", "both"], default="json")
	parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
	parser.add_argument(
		"--source-timezone",
		default=DEFAULT_SOURCE_TIMEZONE,
		help="Timezone used by the source calendar before conversion to UTC. Use 'auto' to detect the machine timezone.",
	)
	return parser


def main() -> int:
	logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
	args = build_arg_parser().parse_args()

	if (args.start_date is None) != (args.end_date is None):
		raise SystemExit("Both --start-date and --end-date are required when using range mode.")

	using_range = bool(args.start_date or args.end_date)
	if args.source == "both" and using_range:
		results = download_calendar_both_interleaved(
			start_date_text=args.start_date,
			end_date_text=args.end_date,
			timeline=args.timeline,
			output_format=args.format,
			output_dir=args.output_dir,
			source_timezone=args.source_timezone,
		)
		for source, result in results.items():
			logger.info("[%s] Downloaded %s rows", source, result.row_count)
			if result.json_path:
				logger.info("[%s] JSON saved to %s", source, result.json_path)
			if result.csv_path:
				logger.info("[%s] CSV saved to %s", source, result.csv_path)
		return 0

	sources_to_run = ["forex", "metals"] if args.source == "both" else [args.source]
	for source in sources_to_run:
		result = download_calendar(
			source=source,
			date_text=args.date,
			start_date_text=args.start_date,
			end_date_text=args.end_date,
			timeline=args.timeline,
			output_format=args.format,
			output_dir=args.output_dir,
			source_timezone=args.source_timezone,
		)

		logger.info("[%s] Downloaded %s rows", source, result.row_count)
		if result.json_path:
			logger.info("[%s] JSON saved to %s", source, result.json_path)
		if result.csv_path:
			logger.info("[%s] CSV saved to %s", source, result.csv_path)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

