"""Create a verified manifest for all official JC monthly archives."""
import json
from pathlib import Path

from src.jc_data import ROOT, month_keys


def main():
    rows = []
    for year in (2024, 2025):
        for key, size in sorted(month_keys(year).items()):
            rows.append({'key': key, 'bytes': size, 'year': year})
    report = {
        'source': 'https://s3.amazonaws.com/tripdata/index.html',
        'years': [2024, 2025], 'files': rows,
        'file_count': len(rows), 'compressed_bytes': sum(x['bytes'] for x in rows),
        'compressed_gib': round(sum(x['bytes'] for x in rows) / 1024**3, 3),
        'coverage_note': 'Manifest only; files are not downloaded by this command.',
    }
    output = ROOT / 'results/metrics'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'jc_manifest_2024_2025.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
