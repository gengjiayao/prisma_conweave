import sys
from pathlib import Path
import unittest
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from source.routing_experiments import _mix64, choose_balanced_action


class FlowHashDiversityTest(unittest.TestCase):
    def test_splitmix64_known_vector_and_unsigned_range(self):
        self.assertEqual(_mix64(0), 0xE220A8397B1DCDAF)
        self.assertTrue(0 <= _mix64((1 << 64)-1) < 1 << 64)

    def test_zero_band_preserves_original_ties_without_key(self):
        self.assertEqual(choose_balanced_action([.5, .5], None, 0), 0)

    def test_bounded_deficit_and_single_eligible_candidate(self):
        for key in range(128):
            self.assertIn(choose_balanced_action([.5, .5625, .25], key, .0625), [0, 1])
            self.assertEqual(choose_balanced_action([.5, .5625, .25], key, .03), 1)

    def test_equal_paths_are_distributed_and_each_flow_is_stable(self):
        actions=[choose_balanced_action([.5]*8, key, .01) for key in range(4096)]
        counts=np.bincount(actions,minlength=8)
        self.assertTrue(np.all((counts > 400) & (counts < 620)), counts)
        self.assertEqual(actions,[choose_balanced_action([.5]*8, key, .01) for key in range(4096)])

    def test_removing_an_unselected_candidate_preserves_choice(self):
        q=np.array([.5]*8)
        for key in range(64):
            action=choose_balanced_action(q,key,.01)
            reduced=q.copy();reduced[(action+1)%8]=0
            self.assertEqual(choose_balanced_action(reduced,key,.01),action)

    def test_missing_inexact_and_out_of_range_keys_fail(self):
        for key in [None, float(2**63), -1, 2**64, True]:
            with self.assertRaises(ValueError):choose_balanced_action([.5,.5],key,.01)
        for key in [0,2**53+1,2**64-1]:
            self.assertIn(choose_balanced_action([.5,.5],key,.01),[0,1])

    def test_invalid_scores_and_bands_fail(self):
        for band in [-.01,.21,np.nan,np.inf]:
            with self.assertRaises(ValueError):choose_balanced_action([.5,.5],1,band)
        with self.assertRaises(ValueError):choose_balanced_action([.5,np.nan],1,.01)


if __name__=='__main__':unittest.main()
