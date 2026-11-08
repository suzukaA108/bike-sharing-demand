"""Download and verify the complete 2024-2025 official JC monthly archive set."""
import json

from src.jc_data import ROOT, month_keys, download_keys


def main():
    files = {}
    for year in (2024, 2025):
        month_files = month_keys(year)
        if len(month_files) != 12:
            raise ValueError(f'Expected 12 official JC files for {year}; found {len(month_files)}')
        files.update(month_files)
    keys = sorted(files)
    expected = {f'JC-{year}{month:02d}-citibike-tripdata' for year in (2024, 2025)
                for month in range(1, 13)}
    observed = {key.rsplit('.', 2)[0] for key in keys}
    if len(keys) != 24 or observed != expected:
        raise ValueError('Official JC archive listing is incomplete or has unexpected filenames')
    result = download_keys(keys, ROOT / 'data/raw/jc')
    report = {
        'source': 'https://s3.amazonaws.com/tripdata/index.html',
        'file_count': len(result),
        'compressed_bytes': sum(item['bytes'] for item in result.values()),
        'archives': result,
    }
    destination = ROOT / 'results/metrics'
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'jc_download_manifest.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'archives'}, indent=2))


if __name__ == '__main__':
    main()
