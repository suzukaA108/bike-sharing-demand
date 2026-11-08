"""从 UCI 下载原始数据并建立真实小时索引，不填补缺失需求。"""
from pathlib import Path
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import argparse
import hashlib
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
URL = 'https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip'


def download():
    destination = ROOT / 'data/raw'
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / 'hour.csv').exists():
        return destination / 'hour.csv'
    with urlopen(URL, timeout=60) as response:
        payload = response.read()
    # Only extract named files; never extract arbitrary archive paths.
    with ZipFile(BytesIO(payload)) as archive:
        for filename in ('hour.csv', 'day.csv', 'Readme.txt'):
            if filename in archive.namelist():
                (destination / filename).write_bytes(archive.read(filename))
    if not (destination / 'hour.csv').exists():
        raise ValueError('UCI archive does not contain hour.csv')
    metadata = {'url': URL, 'archive_sha256': hashlib.sha256(payload).hexdigest()}
    (destination / 'download_metadata.json').write_text(
        json.dumps(metadata, indent=2), encoding='utf-8')
    return destination / 'hour.csv'


def load_hourly(path=None):
    frame = pd.read_csv(path or ROOT / 'data/raw/hour.csv')
    required = {'dteday', 'hr', 'cnt', 'casual', 'registered', 'weathersit',
                'temp', 'hum', 'windspeed', 'workingday', 'holiday'}
    if not required.issubset(frame.columns):
        raise ValueError(f'Missing columns: {required - set(frame.columns)}')
    timestamp = pd.to_datetime(frame['dteday']) + pd.to_timedelta(frame['hr'], unit='h')
    frame.index = pd.DatetimeIndex(timestamp, name='timestamp')
    if frame.index.has_duplicates:
        raise ValueError('Duplicate hourly timestamps; resolve before modeling')
    return frame.sort_index()


def audit(frame):
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq='h')
    missing = expected.difference(frame.index)
    return {
        'rows': len(frame), 'start': str(frame.index.min()),
        'end': str(frame.index.max()), 'expected_hours': len(expected),
        'missing_hours': len(missing),
        'missing_timestamps': [str(t) for t in missing],
        'duplicate_timestamps': int(frame.index.duplicated().sum()),
        'null_counts': {k: int(v) for k, v in frame.isna().sum().items()},
        'negative_counts': int((frame['cnt'] < 0).sum()),
        'target_sum_mismatch': int((frame['cnt'] != frame['casual'] + frame['registered']).sum()),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    if args.download:
        download()
    report = audit(load_hourly())
    output = ROOT / 'results/metrics'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'data_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'missing_timestamps'}, indent=2))


if __name__ == '__main__':
    main()
