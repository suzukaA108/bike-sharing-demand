"""Audit downloaded JC archives before feature engineering."""
import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd

from src.jc_data import COLUMNS, ROOT


def audit(paths):
    all_ids = set()
    first_time_by_id = {}
    rows_by_file = {}
    monthly = {}
    duplicate_ids_within_csv = 0
    duplicate_ids_across_csvs = 0
    conflicting_duplicate_times = 0
    conflicting_duplicate_ids = set()
    duplicate_examples = []
    invalid_times = 0
    ambiguous_dst = 0
    nonexistent_dst = 0
    dst_examples = []
    for path in sorted(paths):
        file_rows = 0
        with ZipFile(path) as archive:
            names = [n for n in archive.namelist()
                     if n.lower().endswith('.csv') and not n.startswith('__MACOSX/')]
            for name in names:
                frame = pd.read_csv(archive.open(name), usecols=COLUMNS,
                                    dtype={'ride_id': 'string'})
                file_rows += len(frame)
                times = pd.to_datetime(frame['started_at'], errors='coerce')
                invalid_times += int(times.isna().sum())
                wall_times = pd.DatetimeIndex(times.dropna())
                localized = wall_times.tz_localize(
                    'America/New_York', ambiguous='NaT', nonexistent='NaT')
                ambiguous_mask = wall_times.tz_localize(
                    'America/New_York', ambiguous='NaT', nonexistent='shift_forward').isna()
                nonexistent_mask = wall_times.tz_localize(
                    'America/New_York', ambiguous=True, nonexistent='NaT').isna()
                ambiguous_dst += int(ambiguous_mask.sum())
                nonexistent_dst += int(nonexistent_mask.sum())
                for value, is_ambiguous, is_nonexistent in zip(
                        wall_times, ambiguous_mask, nonexistent_mask):
                    if is_ambiguous or is_nonexistent:
                        dst_examples.append({'started_at': str(value),
                            'ambiguous': bool(is_ambiguous),
                            'nonexistent': bool(is_nonexistent), 'archive': path.name})

                frame = frame.assign(_parsed_time=times)
                local_dupes = frame['ride_id'].notna() & frame['ride_id'].duplicated()
                duplicate_ids_within_csv += int(local_dupes.sum())
                for idx in frame.index[local_dupes][:10]:
                    duplicate_examples.append({'ride_id': str(frame.at[idx, 'ride_id']),
                        'started_at': str(frame.at[idx, '_parsed_time']), 'archive': path.name,
                        'csv': name, 'kind': 'within_csv'})
                unique_rows = frame.loc[frame['ride_id'].notna()].drop_duplicates('ride_id')
                cross_mask = unique_rows['ride_id'].isin(all_ids)
                duplicate_ids_across_csvs += int(cross_mask.sum())
                for _, row in unique_rows.loc[cross_mask].head(10).iterrows():
                    ride_id = str(row['ride_id'])
                    current_time = str(row['_parsed_time'])
                    previous_time = first_time_by_id[ride_id]
                    conflicting_duplicate_times += int(previous_time != current_time)
                    if previous_time != current_time:
                        conflicting_duplicate_ids.add(ride_id)
                    if len(duplicate_examples) < 30:
                        duplicate_examples.append({'ride_id': ride_id,
                            'first_started_at': previous_time, 'duplicate_started_at': current_time,
                            'archive': path.name, 'csv': name, 'kind': 'cross_csv'})
                for _, row in unique_rows.loc[~cross_mask].iterrows():
                    ride_id = str(row['ride_id'])
                    all_ids.add(ride_id)
                    first_time_by_id[ride_id] = str(row['_parsed_time'])
                valid = frame.loc[times.notna()].copy()
                valid['started_at'] = times.loc[times.notna()].array
                for month, group in valid.groupby(valid['started_at'].dt.to_period('M')):
                    key = str(month)
                    record = monthly.setdefault(key, {'rows': 0, 'files': set(),
                        'min': None, 'max': None})
                    record['rows'] += len(group)
                    record['files'].add(path.name)
                    low, high = group['started_at'].min(), group['started_at'].max()
                    record['min'] = str(low) if record['min'] is None else min(record['min'], str(low))
                    record['max'] = str(high) if record['max'] is None else max(record['max'], str(high))
        rows_by_file[path.name] = file_rows
    for record in monthly.values():
        record['files'] = sorted(record['files'])
    return {
        'archive_count': len(paths), 'archives': rows_by_file,
        'total_rows': sum(rows_by_file.values()), 'unique_ride_ids': len(all_ids),
        'duplicate_ride_ids_within_csv': duplicate_ids_within_csv,
        'duplicate_ride_ids_across_csvs': duplicate_ids_across_csvs,
        'duplicate_rows_total': duplicate_ids_within_csv + duplicate_ids_across_csvs,
        'duplicate_timestamp_conflicts': conflicting_duplicate_times,
        'conflicting_duplicate_ids': sorted(conflicting_duplicate_ids),
        'duplicate_examples': duplicate_examples[:30],
        'invalid_started_at': invalid_times, 'monthly': monthly,
        'dst_ambiguous_rows': ambiguous_dst, 'dst_nonexistent_rows': nonexistent_dst,
        'dst_examples': dst_examples[:30],
        'note': 'Timestamps are naive America/New_York wall time. Ambiguous and nonexistent DST records cannot be assigned reliably to UTC without an offset.',
    }


def main():
    paths = sorted((ROOT / 'data/raw/jc').glob('JC-*.zip'))
    if len(paths) != 24:
        raise FileNotFoundError(f'Expected 24 JC archives, found {len(paths)}')
    report = audit(paths)
    output = ROOT / 'results/metrics'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'jc_full_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'monthly'}, indent=2))
    print('monthly:')
    for key, value in report['monthly'].items():
        print(key, value['rows'], value['min'], value['max'], value['files'])


if __name__ == '__main__':
    main()
