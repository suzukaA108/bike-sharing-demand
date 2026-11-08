import unittest

import numpy as np
import pandas as pd

from src.jc_features import make_features


class JCFeatureTests(unittest.TestCase):
    def setUp(self):
        self.index = pd.date_range('2024-01-01', periods=200, freq='h', tz='UTC')
        self.frame = pd.DataFrame({
            'departures': np.arange(200, dtype=float),
            'temperature_2m': np.arange(200, dtype=float),
            'local_time': self.index.tz_convert('America/New_York').tz_localize(None),
            'dst_ambiguous_target': False,
        }, index=self.index)

    def test_one_step_alignment_and_history_only(self):
        x, y = make_features(self.frame)
        timestamp = self.index[168]
        self.assertEqual(y.loc[timestamp], 169)
        self.assertEqual(x.loc[timestamp, 'demand_lag_1'], 168)
        self.assertEqual(x.loc[timestamp, 'demand_lag_24'], 145)
        self.assertEqual(x.loc[timestamp, 'demand_lag_168'], 1)
        self.assertEqual(x.loc[timestamp, 'temperature_2m_lag_1'], 168)
        self.assertLess(timestamp + pd.Timedelta(hours=1), self.index[-1] + pd.Timedelta(hours=1))

    def test_does_not_score_dst_ambiguous_target(self):
        self.frame.iloc[170, self.frame.columns.get_loc('dst_ambiguous_target')] = True
        x, y = make_features(self.frame)
        self.assertNotIn(self.index[169], x.index)

    def test_rejects_gapped_index(self):
        with self.assertRaises(ValueError):
            make_features(self.frame.drop(self.index[50]))


if __name__ == '__main__':
    unittest.main()
