import unittest

import pandas as pd

from plotting.qps_over_time import _bucket_width_s


class TestQpsBucketWidth(unittest.TestCase):
    def test_width_inferred_from_timepoints(self):
        df = pd.DataFrame({"timepoint": [0.0, 0.5, 1.0, 1.5]})
        self.assertAlmostEqual(_bucket_width_s(df), 0.5)

    def test_single_row_defaults_to_one_second(self):
        df = pd.DataFrame({"timepoint": [0.0]})
        self.assertEqual(_bucket_width_s(df), 1.0)


if __name__ == "__main__":
    unittest.main()
