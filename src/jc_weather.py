"""Join pilot departures to historical weather on explicit UTC timestamps."""
import argparse
import json

import pandas as pd
import requests

from src.jc_data import ROOT

API = 'https://archive-api.open-meteo.com/v1/archive'
FIELDS = ['temperature_2m', 'relative_humidity_2m', 'precipitation', 'wind_speed_10m']


def fetch_weather_range(start_date, end_date, latitude=40.73, longitude=-74.04):
    params = {
        'latitude': latitude, 'longitude': longitude,
        'start_date': pd.Timestamp(start_date).date().isoformat(),
        'end_date': pd.Timestamp(end_date).date().isoformat(),
        'hourly': ','.join(FIELDS), 'timezone': 'UTC',
    }
    response = requests.get(API, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    frame = pd.DataFrame(payload['hourly'])
    frame['timestamp_utc'] = pd.to_datetime(frame.pop('time'), utc=True)
    if frame['timestamp_utc'].duplicated().any():
        raise ValueError('Duplicate UTC weather timestamps')
    return frame.set_index('timestamp_utc'), {
        'request': params, 'resolved_latitude': payload['latitude'],
        'resolved_longitude': payload['longitude'],
        'hourly_units': payload['hourly_units'],
    }


def load_or_fetch_weather(start_date, end_date, cache_name):
    cache_dir = ROOT / 'data/processed'
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / f'{cache_name}.csv'
    meta_path = cache_dir / f'{cache_name}.json'
    if data_path.exists() and meta_path.exists():
        frame = pd.read_csv(data_path, parse_dates=['timestamp_utc'], index_col='timestamp_utc')
        frame.index = pd.DatetimeIndex(frame.index, tz='UTC')
        return frame, json.loads(meta_path.read_text(encoding='utf-8'))
    frame, metadata = fetch_weather_range(start_date, end_date)
    frame.to_csv(data_path, index_label='timestamp_utc')
    meta_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return frame, metadata


def fetch_weather(month, latitude=40.73, longitude=-74.04):
    start = pd.Timestamp(f'{month}-01')
    end_exclusive = start + pd.offsets.MonthBegin(1)
    # Request the complete UTC dates enclosing this local calendar month.
    utc_start = start.tz_localize('America/New_York').tz_convert('UTC')
    utc_end = end_exclusive.tz_localize('America/New_York').tz_convert('UTC')
    return fetch_weather_range(utc_start.date(),
        (utc_end - pd.Timedelta(seconds=1)).date(), latitude, longitude)


def join_pilot(hourly, weather):
    """Join naive local wall-clock hours, explicitly excluding DST ambiguity."""
    if hourly.index.tz is not None:
        raise ValueError('Expected naive local pilot index')
    if not weather.index.tz or str(weather.index.tz) != 'UTC':
        raise ValueError('Weather timestamps must be UTC-aware')
    local = hourly.copy()
    localized = local.index.tz_localize(
        'America/New_York', ambiguous='NaT', nonexistent='NaT')
    local['dst_ambiguous_or_nonexistent'] = localized.isna()
    local.index = localized
    local = local.loc[local.index.notna()]
    local.index = local.index.tz_convert('UTC')
    local.index.name = 'timestamp_utc'
    return local.join(weather, how='left', validate='one_to_one')


def join_hourly(hourly, weather):
    if not isinstance(hourly.index, pd.DatetimeIndex) or str(hourly.index.tz) != 'UTC':
        raise ValueError('Prepared hourly index must be UTC-aware')
    if not isinstance(weather.index, pd.DatetimeIndex) or str(weather.index.tz) != 'UTC':
        raise ValueError('Weather index must be UTC-aware')
    return hourly.join(weather, how='left', validate='one_to_one')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--month', default='2025-01', choices=['2025-01'])
    args = parser.parse_args()
    input_path = ROOT / 'results/metrics/jc_pilot_hourly.csv'
    hourly = pd.read_csv(input_path, parse_dates=['local_hour'], index_col='local_hour')
    weather, metadata = fetch_weather(args.month)
    joined = join_pilot(hourly, weather)
    report = {
        'month': args.month, 'departure_hours': len(joined),
        'weather_hours_in_api_response': len(weather),
        'matched_weather_hours': int(joined[FIELDS].notna().all(axis=1).sum()),
        'dst_ambiguous_or_nonexistent_hours': int(joined['dst_ambiguous_or_nonexistent'].sum()),
        'missing_weather_by_column': {k: int(v) for k, v in joined[FIELDS].isna().sum().items()},
        'weather_metadata': metadata,
        'note': 'Historical weather may be reanalysis; availability at prediction time is unverified',
    }
    output = ROOT / 'results/metrics'
    (output / 'jc_weather_pilot_audit.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    # Preserve the weather snapshot used by the pilot outside Git.
    cache = ROOT / 'data/processed'
    cache.mkdir(parents=True, exist_ok=True)
    joined.to_csv(cache / 'jc_weather_pilot_2025-01.csv')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
