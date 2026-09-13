import unittest
import numpy as np
from source.routing_experiments import service_delay_us, diagnostic_scores


class ServiceDelayTest(unittest.TestCase):
    def obs(self):
        # Two candidates have the same byte backlog but different rates.
        return np.array([[0, 0, .5, .5, 0, .5, 1, 1,
                         .5, .5, 0, .5, .5, .5]], dtype=np.float32)

    def test_same_backlog_takes_twice_as_long_on_half_rate_link(self):
        delays = service_delay_us(self.obs(), 50*1024**2, 1)
        self.assertAlmostEqual(delays[0, 1], 2*delays[0, 0])

    def test_idle_slow_link_beats_queued_fast_link(self):
        obs = self.obs(); obs[0, 8:10] = 0
        self.assertEqual(int(np.argmax(diagnostic_scores(obs, 'queue', 50*1024**2, 1))), 1)

    def test_guard_rejects_large_delay_regression_but_preserves_near_tie(self):
        q = np.array([[0., 10.]])
        scores = diagnostic_scores(self.obs(), 'guard', 50*1024**2, 1, slack_us=20, learned_q=q)
        self.assertEqual(int(np.argmax(scores)), 0)
        obs = self.obs(); obs[0, 13] = 1
        scores = diagnostic_scores(obs, 'guard', 50*1024**2, 1, slack_us=20, learned_q=q)
        self.assertEqual(int(np.argmax(scores)), 1)

    def test_growth_prediction_has_time_units_and_is_conservative(self):
        obs = self.obs(); obs[0, 5] = 1; obs[0, 11] = 0
        now = service_delay_us(obs, 50*1024**2, 1)
        later = service_delay_us(obs, 50*1024**2, 1, horizon_us=20)
        np.testing.assert_allclose(later-now, [[20, 0]])

    def test_nonfinite_model_scores_fail_before_statistics_are_updated(self):
        from source.forwarder import Forwarder
        forwarder = Forwarder.__new__(Forwarder)
        with self.assertRaises(ValueError):
            forwarder._record_q_values([[1.0, float("nan")]])

    def test_saturated_queue_is_finite_and_bad(self):
        obs = self.obs(); obs[0, 2] = 1
        delays = service_delay_us(obs, 50*1024**2, 1)
        self.assertTrue(np.all(np.isfinite(delays)))
        self.assertGreater(delays[0, 0], delays[0, 1])

    def test_candidate_permutation_permutes_scores(self):
        obs = self.obs(); permuted = np.concatenate([obs[:, :2], obs[:, 8:], obs[:, 2:8]], axis=1)
        a = diagnostic_scores(obs, 'predictive', 50*1024**2, 1)
        b = diagnostic_scores(permuted, 'predictive', 50*1024**2, 1)
        np.testing.assert_allclose(a[:, ::-1], b)

if __name__ == '__main__':
    unittest.main()
