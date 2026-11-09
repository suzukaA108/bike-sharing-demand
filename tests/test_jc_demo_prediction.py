import unittest

import pandas as pd

from src.jc_demo_prediction import TRAIN_END_UTC, resolve_historical_target


class JCDemoPredictionTests(unittest.TestCase):
    def setUp(self):
        start = pd.Timestamp('2024-12-31T00:00:00Z')
        end = pd.Timestamp('2025-01-03T00:00:00Z')
        index = pd.date_range(start, end, freq='h', inclusive='left')
        self.frame = pd.DataFrame({
            'local_time': index.tz_convert('America/New_York').tz_localize(None),
        }, index=index)

    def test_default_uses_latest_available_historical_hour(self):
        local, target_utc, predictor_utc = resolve_historical_target(self.frame)
        self.assertEqual(target_utc, self.frame.index[-1])
        self.assertEqual(predictor_utc + pd.Timedelta(hours=1), target_utc)
        self.assertEqual(local, target_utc.tz_convert('America/New_York').tz_localize(None))

    def test_rejects_target_inside_training_period(self):
        with self.assertRaises(ValueError):
            resolve_historical_target(self.frame, '2024-12-31 23:00')

    def test_rejects_dst_ambiguous_local_hour(self):
        with self.assertRaises(Exception):
            resolve_historical_target(self.frame, '2025-11-02 01:00')

    def test_rejects_time_after_available_data(self):
        with self.assertRaises(ValueError):
            resolve_historical_target(self.frame, '2025-01-05 12:00')


if __name__ == '__main__':
    unittest.main()
