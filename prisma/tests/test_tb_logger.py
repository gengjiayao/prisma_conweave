import os
import sys
import unittest


PRISMA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PRISMA_DIR not in sys.path:
    sys.path.insert(0, PRISMA_DIR)

from source.tb_logger import _safe_ratio  # noqa: E402


class SafeRatioTest(unittest.TestCase):
    def test_zero_denominator_is_reported_as_zero(self):
        self.assertEqual(_safe_ratio(7, 0), 0.0)

    def test_negative_denominator_is_reported_as_zero(self):
        self.assertEqual(_safe_ratio(7, -1), 0.0)

    def test_positive_denominator_preserves_rate(self):
        self.assertAlmostEqual(_safe_ratio(1, 4), 0.25)


if __name__ == "__main__":
    unittest.main()
