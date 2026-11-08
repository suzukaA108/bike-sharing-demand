"""Audit a bounded Citi Bike JC pilot without conflating file month and trip month."""
import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.parse import urlencode
from zipfile import ZipFile
from xml.etree import ElementTree

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
BUCKET = 'https://s3.amazonaws.com/tripdata'
NS = {'s': 'http://s3.amazonaws.com/doc/2006-03-01/'}
COLUMNS = ['ride_id', 'started_at', 'start_station_id', 'start_lat', 'start_lng']


def month_keys(year):
    response = requests.get(BUCKET, params={
        'list-type': '2', 'prefix': f'JC-{year}', 'max-keys': 1000}, timeout=40)
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    if root.findtext('s:IsTruncated', namespaces=NS) != 'false':
        raise ValueError('Incomplete S3 listing; do not infer month coverage')
    return {item.findtext('s:Key', namespaces=NS):
            int(item.findtext('s:Size', namespaces=NS))
            for item in root.findall('s:Contents', NS)}


def download_keys(keys, directory):
    directory.mkdir(parents=True, exist_ok=True)
    metadata = {}
    for key in keys:
        # Keys must originate from our fixed configuration and the official listing.
        if '/' in key or not key.startswith('JC-') or not key.endswith('.zip'):
            raise ValueError(f'Unexpected archive name: {key}')
        path = directory / key
        if not path.exists():
            response = requests.get(f'{BUCKET}/{key}', timeout=120)
            response.raise_for_status()
            with ZipFile(io.BytesIO(response.content)) as archive:
                if archive.testzip() is not None:
                    raise ValueError(f'Corrupt archive: {key}')
            path.write_bytes(response.content)
        with ZipFile(path) as archive:
            bad_member = archive.testzip()
            if bad_member is not None:
                raise ValueError(f'Corrupt member {bad_member} in {path.name}')
        metadata[key] = {'bytes': path.stat().st_size,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    return metadata


def load_trips(paths):
    frames = []
    file_rows = {}
    for path in paths:
        with ZipFile(path) as archive:
            names = [name for name in archive.namelist()
                     if name.lower().endswith('.csv') and not name.startswith('__MACOSX/')]
            if not names:
                raise ValueError(f'No trip CSV in {path.name}')
            for name in names:
                frame = pd.read_csv(archive.open(name), usecols=COLUMNS,
                                    dtype={'ride_id': 'string', 'start_station_id': 'string'})
                frame['source_file'] = path.name
                frames.append(frame)
            file_rows[path.name] = sum(len(frame) for frame in frames
                                       if frame['source_file'].iloc[0] == path.name)
    return pd.concat(frames, ignore_index=True), file_rows


def aggregate_month(trips, month):
    """Return observed departures; keep wall-clock timezone ambiguity explicit."""
    start = pd.Timestamp(f'{month}-01')
    end = start + pd.offsets.MonthBegin(1)
    frame = trips.copy()
    frame['started_at'] = pd.to_datetime(frame['started_at'], errors='coerce')
    invalid_time = int(frame['started_at'].isna().sum())
    in_month = frame.loc[frame['started_at'].ge(start) & frame['started_at'].lt(end)].copy()
    missing_id = int(in_month['ride_id'].isna().sum())
    duplicate_id = int(in_month['ride_id'].duplicated().sum())
    in_month = in_month.loc[in_month['ride_id'].notna()]
    in_month = in_month.drop_duplicates('ride_id', keep='first')
    counts = in_month.groupby(in_month['started_at'].dt.floor('h')).size()
    hours = pd.date_range(start, end - pd.Timedelta(hours=1), freq='h', name='local_hour')
    counts = counts.reindex(hours, fill_value=0).rename('departures')
    report = {
        'month': month, 'input_rows': len(frame), 'invalid_time_rows': invalid_time,
        'rows_in_month_before_dedup': int(len(frame.loc[frame['started_at'].ge(start) & frame['started_at'].lt(end)])),
        'missing_ride_id_in_month': missing_id, 'duplicate_ride_id_in_month': duplicate_id,
        'deduplicated_departures': len(in_month), 'hours': len(counts),
        'zero_departure_hours': int(counts.eq(0).sum()),
        'rows_by_source_file': {str(k): int(v) for k, v in
                                in_month.groupby('source_file').size().items()},
        'time_warning': 'Naive local wall timestamps; DST not resolved in this pilot',
        'coverage_warning': 'A zero count assumes the three source files cover this pilot month',
    }
    return counts, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Download three small pilot archives')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/jc_pilot.json').read_text(encoding='utf-8'))
    keys = config['required_files']
    available = {}
    for year in (2024, 2025):
        available.update(month_keys(year))
    missing = set(keys) - available.keys()
    if missing:
        raise ValueError(f'Official index missing required files: {missing}')
    directory = ROOT / 'data/raw/jc'
    if args.download:
        file_metadata = download_keys(keys, directory)
    else:
        file_metadata = None
    paths = [directory / key for key in keys]
    if any(not path.exists() for path in paths):
        raise FileNotFoundError('Missing pilot archives; rerun with --download')
    trips, file_rows = load_trips(paths)
    counts, report = aggregate_month(trips, config['pilot_month'])
    report['source_file_rows'] = file_rows
    report['source_files'] = file_metadata
    output = ROOT / 'results/metrics'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'jc_pilot_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    counts.to_csv(output / 'jc_pilot_hourly.csv', index_label='local_hour')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
