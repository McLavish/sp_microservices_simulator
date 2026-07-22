import os
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from metrics import Metrics
from plotting.qps_over_time import _bucket_width_s


class TestQpsBucketWidth(unittest.TestCase):
    def test_width_inferred_from_timepoints(self):
        df = pd.DataFrame({"timepoint": [0.0, 0.5, 1.0, 1.5]})
        self.assertAlmostEqual(_bucket_width_s(df), 0.5)

    def test_single_row_defaults_to_one_second(self):
        df = pd.DataFrame({"timepoint": [0.0]})
        self.assertEqual(_bucket_width_s(df), 1.0)


class TestHeaderOnlyCsv(unittest.TestCase):
    def _header_only_csv(self, tmp):
        path = os.path.join(tmp, "empty.csv")
        Metrics(roots=[], attempts_total=0).export_csv(path)  # writes header only
        return path

    def test_require_rows_raises_system_exit(self):
        from plotting.utils.io import load_metrics, require_rows

        with tempfile.TemporaryDirectory() as tmp:
            path = self._header_only_csv(tmp)
            df = load_metrics(path)
            with self.assertRaises(SystemExit):
                require_rows(df, path)

    def test_failures_over_time_exits_cleanly_not_typeerror(self):
        import plotting.failures_over_time as failures_over_time

        with tempfile.TemporaryDirectory() as tmp:
            path = self._header_only_csv(tmp)
            with self.assertRaises(SystemExit):
                failures_over_time.main(path, str(Path(tmp) / "out.png"))


if __name__ == "__main__":
    unittest.main()
