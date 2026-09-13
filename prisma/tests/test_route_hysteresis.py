import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source.routing_experiments import choose_hysteresis_action


class RouteHysteresisTest(unittest.TestCase):
    def test_zero_margin_preserves_native_ties_even_without_metadata(self):
        self.assertEqual(choose_hysteresis_action([.5, .5], 1, None, 0), 0)
        self.assertEqual(choose_hysteresis_action([.5, .5], 1, True, 0), 0)

    def test_first_flowlet_is_not_biased_toward_placeholder_action(self):
        self.assertEqual(choose_hysteresis_action([.5, .5625], 0, False, .125), 1)

    def test_small_gain_retains_route_including_exact_margin_boundary(self):
        self.assertEqual(choose_hysteresis_action([.5, .5625], 0, True, .0625), 0)
        self.assertEqual(choose_hysteresis_action([.5625, .5], 1, True, .125), 1)

    def test_larger_congestion_gap_allows_escape(self):
        self.assertEqual(choose_hysteresis_action([.25, .75], 0, True, .125), 1)

    def test_missing_ownership_and_invalid_previous_action_fail_closed(self):
        with self.assertRaises(ValueError):
            choose_hysteresis_action([.5, .6], 0, None, .05)
        with self.assertRaises(ValueError):
            choose_hysteresis_action([.5, .6], 2, True, .05)

    def test_nonfinite_scores_and_invalid_margins_are_rejected(self):
        for margin in [-.01, .21, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                choose_hysteresis_action([.5, .6], 0, True, margin)
        with self.assertRaises(ValueError):
            choose_hysteresis_action([np.nan, .6], 0, True, .05)


if __name__ == '__main__':
    unittest.main()
