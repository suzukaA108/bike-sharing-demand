import unittest

import pandas as pd

from src.jc_data import aggregate_month
from src.jc_weather import join_pilot


class JCDataTests(unittest.TestCase):
    def test_actual_started_at_and_cross_file_deduplication(self):
        trips = pd.DataFrame({
            'ride_id': ['previous', 'one', 'one', 'two', 'next'],
            'started_at': ['2024-12-31 23:50', '2025-01-01 00:10',
                           '2025-01-01 00:10', '2025-01-01 02:00',
                           '2025-02-01 00:10'],
            'source_file': ['jan.zip', 'dec.zip', 'jan.zip', 'feb.zip', 'jan.zip'],
        })
        hourly, report = aggregate_month(trips, '2025-01')
        self.assertEqual(len(hourly), 744)
        self.assertEqual(hourly.loc['2025-01-01 00:00'], 1)
        self.assertEqual(hourly.loc['2025-01-01 01:00'], 0)
        self.assertEqual(hourly.loc['2025-01-01 02:00'], 1)
        self.assertEqual(int(hourly.sum()), 2)
        self.assertEqual(report['duplicate_ride_id_in_month'], 1)

    def test_january_weather_alignment_uses_utc(self):
        local_hours = pd.date_range('2025-01-01', periods=2, freq='h', name='local_hour')
        hourly = pd.DataFrame({'departures': [3, 4]}, index=local_hours)
        weather = pd.DataFrame({'temperature_2m': [10, 20]}, index=pd.date_range(
            '2025-01-01 05:00', periods=2, freq='h', tz='UTC', name='timestamp_utc'))
        joined = join_pilot(hourly, weather)
        self.assertEqual(joined['temperature_2m'].tolist(), [10, 20])
        self.assertEqual(str(joined.index.tz), 'UTC')

    def test_dst_ambiguous_hour_is_explicitly_excluded(self):
        hourly = pd.DataFrame({'departures': [1]}, index=pd.DatetimeIndex(
            ['2025-11-02 01:00'], name='local_hour'))
        weather = pd.DataFrame({'temperature_2m': [5]}, index=pd.date_range(
            '2025-11-02 05:00', periods=1, freq='h', tz='UTC'))
        joined = join_pilot(hourly, weather)
        self.assertEqual(len(joined), 0)


if __name__ == '__main__':
    unittest.main()
