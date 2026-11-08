"""Build a DST-aware UTC hourly target from the downloaded JC trip archives."""
import json
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import numpy as np

from src.jc_data import COLUMNS, ROOT

TIMEZONE = 'America/New_York'
START_LOCAL = pd.Timestamp('2024-01-01 00:00')
END_LOCAL = pd.Timestamp('2026-01-01 00:00')


def prepare_hourly(paths):
    expected_months = {f'{year}-{month:02d}' for year in (2024, 2025)
                       for month in range(1, 13)}
    # Parse the YYYYMM token explicitly rather than relying on extension naming.
    archive_months = {path.name[3:7] + '-' + path.name[7:9] for path in paths}
    if len(paths) != 24 or archive_months != expected_months:
        raise ValueError('Expected exactly one official archive for each month in 2024-2025')

    audit_path = ROOT / 'results/metrics/jc_full_audit.json'
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    conflicting_ids = set(audit.get('conflicting_duplicate_ids', []))

    seen_ids = set()
    hourly_counts = defaultdict(int)
    rows_read = 0
    duplicate_rows = 0
    conflict_rows = 0
    missing_ids = 0
    invalid_times = 0
    ambiguous_rows = 0
    nonexistent_rows = 0

    for path in sorted(paths):
        with ZipFile(path) as archive:
            members = [name for name in archive.namelist()
                       if name.lower().endswith('.csv') and not name.startswith('__MACOSX/')]
            if not members:
                raise ValueError(f'No CSV in {path.name}')
            for member in members:
                frame = pd.read_csv(archive.open(member), usecols=COLUMNS,
                                    dtype={'ride_id': 'string'})
                rows_read += len(frame)
                missing_ids += int(frame['ride_id'].isna().sum())
                frame = frame.loc[frame['ride_id'].notna()].copy()
                conflict = frame['ride_id'].isin(conflicting_ids)
                conflict_rows += int(conflict.sum())
                frame = frame.loc[~conflict].copy()
                row_count_before_dedup = len(frame)
                frame = frame.drop_duplicates('ride_id', keep='first')
                duplicate_rows += row_count_before_dedup - len(frame)
                repeated = frame['ride_id'].isin(seen_ids)
                duplicate_rows += int(repeated.sum())
                frame = frame.loc[~repeated].copy()
                seen_ids.update(frame['ride_id'].tolist())
                times = pd.to_datetime(frame['started_at'], errors='coerce')
                invalid_times += int(times.isna().sum())
                valid_mask = times.notna()
                times = times.loc[valid_mask]
                local_index = pd.DatetimeIndex(times)
                ambiguous_mask = local_index.tz_localize(
                    TIMEZONE, ambiguous='NaT', nonexistent='shift_forward').isna()
                nonexistent_mask = local_index.tz_localize(
                    TIMEZONE, ambiguous=True, nonexistent='NaT').isna()
                ambiguous_rows += int(ambiguous_mask.sum())
                nonexistent_rows += int(nonexistent_mask.sum())
                localized = local_index.tz_localize(
                    TIMEZONE, ambiguous='NaT', nonexistent='NaT')
                usable = ~localized.isna()
                utc_hours = localized[usable].floor('h').tz_convert('UTC')
                for timestamp, count in pd.Series(1, index=utc_hours).groupby(level=0).sum().items():
                    hourly_counts[timestamp] += int(count)

    start_utc = START_LOCAL.tz_localize(TIMEZONE).tz_convert('UTC')
    end_utc = END_LOCAL.tz_localize(TIMEZONE).tz_convert('UTC')
    index = pd.date_range(start_utc, end_utc, freq='h', inclusive='left', name='timestamp_utc')
    out = build_hourly_frame(index, hourly_counts)
    monthly_counts = out.groupby(out['local_time'].dt.to_period('M'))['departures'].agg(
        observed_hours='count', zero_hours=lambda values: int(values.eq(0).sum()),
        rides='sum')

    report = {
        'source_archives': len(paths), 'raw_rows': rows_read,
        'unique_ids_after_exclusions': len(seen_ids),
        'duplicate_rows_skipped': duplicate_rows,
        'conflicting_ids_excluded': len(conflicting_ids),
        'conflicting_rows_excluded': conflict_rows,
        'missing_ride_id_rows': missing_ids, 'invalid_started_at_rows': invalid_times,
        'ambiguous_dst_ride_rows_excluded': ambiguous_rows,
        'nonexistent_dst_ride_rows_excluded': nonexistent_rows,
        'hourly_index_rows': len(index),
        'ambiguous_dst_target_hours_unscorable': int(out['dst_ambiguous_target'].sum()),
        'scorable_target_hours': int(out['departures'].notna().sum()),
        'zero_departure_hours': int(out['departures'].eq(0).sum()),
        'monthly': {str(key): {k: (int(v) if k != 'rides' else float(v))
                               for k, v in row.items()}
                    for key, row in monthly_counts.to_dict(orient='index').items()},
        'zero_policy': 'Zero for hours in fully present monthly archives; possible source outages remain a limitation.',
        'duplicate_policy': 'Exclude every ID with conflicting started_at; otherwise keep first occurrence in sorted archive/member order.',
        'dst_policy': 'Localize source wall times to America/New_York; discard ambiguous/nonexistent trip timestamps. Mark both fall-back UTC target hours unscorable.',
    }
    return out, report


def build_hourly_frame(index, hourly_counts):
    local_wall = index.tz_convert(TIMEZONE).tz_localize(None)
    ambiguous_hours = local_wall.duplicated(keep=False)
    series = pd.Series(hourly_counts, dtype='int64').reindex(index, fill_value=0).astype('float64')
    series.iloc[ambiguous_hours] = float('nan')
    out = pd.DataFrame({'departures': series}, index=index)
    local = index.tz_convert(TIMEZONE)
    out['local_time'] = local.tz_localize(None)
    out['dst_ambiguous_target'] = ambiguous_hours
    out['hour'] = local.hour
    out['weekday'] = local.dayofweek
    out['month'] = local.month
    out['is_weekend'] = local.dayofweek >= 5
    out['hour_sin'] = np.sin(2 * np.pi * local.hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * local.hour / 24)
    out['weekday_sin'] = np.sin(2 * np.pi * local.dayofweek / 7)
    out['weekday_cos'] = np.cos(2 * np.pi * local.dayofweek / 7)
    return out

def main():
    paths = sorted((ROOT / 'data/raw/jc').glob('JC-*.zip'))
    hourly, report = prepare_hourly(paths)
    output = ROOT / 'data/processed'
    output.mkdir(parents=True, exist_ok=True)
    hourly.to_csv(output / 'jc_hourly_2024_2025.csv', index_label='timestamp_utc')
    metrics = ROOT / 'results/metrics'
    (metrics / 'jc_preparation_audit.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
