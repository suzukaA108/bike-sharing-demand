import unittest

import pandas as pd

from src.jc_baselines import naive_predictions


class JCBaselineTests(unittest.TestCase):
    def test_previous_hour_and_week_alignment(self):
        index = pd.date_range('2024-01-01', periods=200, freq='h', tz='UTC')
        hourly = pd.DataFrame({'departures': range(200)}, index=index)
        target_index = index[168:170]
        prediction = naive_predictions(hourly, pd.Series(index=target_index, dtype=float))
        self.assertEqual(prediction.iloc[0].tolist(), [168.0, 1.0])
        self.assertEqual(prediction.iloc[1].tolist(), [169.0, 2.0])


if __name__ == '__main__':
    unittest.main()
