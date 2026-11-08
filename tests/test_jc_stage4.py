import unittest

import numpy as np
import pandas as pd

from src.jc_stage4 import common_model_table


class Stage4Tests(unittest.TestCase):
    def test_common_table_intersects_predictions_and_checks_targets(self):
        index = pd.date_range('2024-01-01', periods=3, freq='h', tz='UTC')
        left = pd.DataFrame({'actual': [1, 2, 3], 'rf': [1.1, 2.1, 3.1]}, index=index)
        right = pd.DataFrame({'actual': [2, 3], 'lstm': [1.9, 3.2]}, index=index[1:])
        result = common_model_table([left, right])
        self.assertEqual(result.index.tolist(), index[1:].tolist())
        self.assertEqual(result['actual'].tolist(), [2, 3])
        self.assertEqual(list(result.columns), ['actual', 'rf', 'lstm'])

    def test_common_table_rejects_disagreeing_actuals(self):
        index = pd.date_range('2024-01-01', periods=1, freq='h', tz='UTC')
        left = pd.DataFrame({'actual': [1], 'rf': [1]}, index=index)
        right = pd.DataFrame({'actual': [2], 'lstm': [2]}, index=index)
        with self.assertRaises(ValueError):
            common_model_table([left, right])


if __name__ == '__main__':
    unittest.main()
