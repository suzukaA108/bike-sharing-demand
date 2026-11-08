import tempfile
import unittest
from pathlib import Path
from zipfile import ZipFile

from src.jc_year_audit import audit


class YearAuditTests(unittest.TestCase):
    def test_cross_archive_ids_and_month_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name, csv in [
                ('a.zip', 'ride_id,started_at,start_station_id,start_lat,start_lng\na,2025-01-31 23:59,s,1,1\n'),
                ('b.zip', 'ride_id,started_at,start_station_id,start_lat,start_lng\na,2025-01-31 23:59,s,1,1\nb,2025-02-01 00:01,s,1,1\n'),
            ]:
                with ZipFile(root / name, 'w') as archive:
                    archive.writestr('trips.csv', csv)
            report = audit([root / 'a.zip', root / 'b.zip'])
            self.assertEqual(report['total_rows'], 3)
            self.assertEqual(report['duplicate_ride_ids_across_csvs'], 1)
            self.assertEqual(report['monthly']['2025-01']['rows'], 2)
            self.assertEqual(report['monthly']['2025-02']['rows'], 1)

    def test_dst_wall_times_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'dst.zip'
            csv = ('ride_id,started_at,start_station_id,start_lat,start_lng\n'
                   'a,2025-03-09 02:30,s,1,1\n'
                   'b,2025-11-02 01:30,s,1,1\n'
                   'c,2025-11-02 02:30,s,1,1\n')
            with ZipFile(path, 'w') as archive:
                archive.writestr('trips.csv', csv)
            report = audit([path])
            self.assertEqual(report['dst_nonexistent_rows'], 1)
            self.assertEqual(report['dst_ambiguous_rows'], 1)


if __name__ == '__main__':
    unittest.main()
