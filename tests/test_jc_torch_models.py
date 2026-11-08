import unittest

import numpy as np
import pandas as pd

from src.jc_torch_models import _make_sequences, _fold_masks
from src.jc_features import make_features


class JCTorchSequenceTests(unittest.TestCase):
    def test_168_hour_window_ends_at_predictor_time(self):
        index = pd.date_range('2024-01-01', periods=200, freq='h', tz='UTC')
        frame = pd.DataFrame({
            'departures': np.arange(200, dtype=float),
            'temperature_2m': np.arange(200, dtype=float) + 100,
            'relative_humidity_2m': np.arange(200, dtype=float) + 200,
            'precipitation': np.arange(200, dtype=float) + 300,
            'wind_speed_10m': np.arange(200, dtype=float) + 400,
            'local_time': index.tz_convert('America/New_York').tz_localize(None),
            'dst_ambiguous_target': False,
        }, index=index)
        x, y = make_features(frame)
        sequences, calendar, target, valid, columns = _make_sequences(
            frame, x, y, ['departures', 'temperature_2m', 'relative_humidity_2m',
                          'precipitation', 'wind_speed_10m'])
        pos = x.index.get_loc(index[168])
        self.assertEqual(sequences.shape[1], 168)
        self.assertEqual(sequences[pos, 0, 0], 1)
        self.assertEqual(sequences[pos, -1, 0], 168)
        self.assertEqual(target[pos], 169)
        self.assertTrue(valid[pos])

    def test_fold_mask_uses_target_timestamps(self):
        index = pd.date_range('2024-01-01', periods=10, freq='h', tz='UTC')
        target = pd.Series(range(10), index=index)
        fold = {'train_start_utc': '2024-01-01T00:00:00Z',
                'train_end_utc': '2024-01-01T05:00:00Z',
                'validation_start_utc': '2024-01-01T05:00:00Z',
                'validation_end_utc': '2024-01-01T09:00:00Z'}
        train, valid = _fold_masks(index, target, fold)
        self.assertTrue((index[train] + pd.Timedelta(hours=1) < pd.Timestamp(fold['train_end_utc'])).all())
        self.assertTrue(((index[valid] + pd.Timedelta(hours=1)) >= pd.Timestamp(fold['validation_start_utc'])).all())
        self.assertFalse((index[train] + pd.Timedelta(hours=1) > index[valid].min() + pd.Timedelta(hours=1)).any())


if __name__ == '__main__':
    unittest.main()
