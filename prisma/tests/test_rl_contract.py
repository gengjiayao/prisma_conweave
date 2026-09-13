import os
import sys
import unittest

import numpy as np


PRISMA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PRISMA_DIR not in sys.path:
    sys.path.insert(0, PRISMA_DIR)

from source.rl_contract import (  # noqa: E402
    EGRESS_FEATURE_NAMES,
    RL_CORE_VERSION,
    build_aligned_transition,
    checkpoint_manifest,
    is_valid_environment_step,
    native_action_prior,
    preprocess_observation,
    scheduled_gradient_steps,
    should_update_target,
    validate_observation_shape,
)


class ObservationContractTest(unittest.TestCase):
    def test_destination_ids_remain_distinct_for_one_hot_encoding(self):
        observations = np.zeros((8, 50), dtype=np.float32)
        observations[:, 0] = np.arange(8)
        processed = preprocess_observation(observations, num_actions=8)
        np.testing.assert_array_equal(processed[:, 0], np.arange(8))
        np.testing.assert_array_equal(processed[:, 0].astype(np.int64), np.arange(8))

    def test_quantized_simulator_fields_are_scaled_once(self):
        observation = np.zeros(50, dtype=np.float32)
        observation[0] = 7
        observation[1] = 7
        observation[2:] = np.linspace(0, 5000, 48)
        processed = preprocess_observation(observation, num_actions=8)
        self.assertEqual(processed[0], 7)
        self.assertEqual(processed[1], 1)
        np.testing.assert_allclose(processed[2:], observation[2:] / 5000.0)

    def test_normalized_last_action_contract_mismatch_fails_fast(self):
        observation = np.zeros(50, dtype=np.float32)
        observation[0] = 7
        observation[1] = 0.5
        with self.assertRaisesRegex(ValueError, "integer index"):
            preprocess_observation(observation, num_actions=8)

    def test_wrong_feature_scale_fails_fast(self):
        observation = np.zeros(50, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "fixed feature scale"):
            preprocess_observation(observation, num_actions=8, feature_scale=1.0)

    def test_shape_mismatch_fails_fast(self):
        with self.assertRaises(ValueError):
            validate_observation_shape(42, 8)

    def test_unbounded_simulator_feature_fails_fast(self):
        observation = np.zeros(50, dtype=np.float32)
        observation[2] = 5001
        with self.assertRaisesRegex(ValueError, "must be in"):
            preprocess_observation(observation, num_actions=8)

    def test_native_prior_prefers_idle_fast_links(self):
        observation = np.zeros((1, 50), dtype=np.float32)
        ports = np.array(
            [[0.0, 0.0, 0.0, 0.5, 1.0, 1.0]] * 4
            + [[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]] * 4,
            dtype=np.float32,
        )
        observation[0, 2:] = ports.reshape(-1)
        prior = native_action_prior(observation, num_actions=8)
        self.assertEqual(prior.shape, (1, 8))
        self.assertGreater(float(np.min(prior[0, :4])), float(np.max(prior[0, 4:])))

    def test_native_prior_rejects_raw_quantized_features(self):
        observation = np.zeros(50, dtype=np.float32)
        observation[2] = 5000.0
        with self.assertRaisesRegex(ValueError, "preprocessed"):
            native_action_prior(observation, num_actions=8)


class TransitionContractTest(unittest.TestCase):
    def test_step_result_is_paired_with_same_action_and_teacher_label(self):
        state = np.zeros(50, dtype=np.float32)
        next_state = np.ones(50, dtype=np.float32)
        transition = build_aligned_transition(
            state,
            action=3,
            reward=0.75,
            next_observation=next_state,
            done=False,
            ecmp_action=5,
            num_actions=8,
        )
        self.assertEqual(transition.action, 3)
        self.assertEqual(transition.ecmp_action, 5)
        self.assertEqual(transition.reward, 0.75)
        np.testing.assert_array_equal(transition.observation, state)
        np.testing.assert_array_equal(transition.next_observation, next_state)

    def test_contract_version_is_explicit(self):
        self.assertEqual(RL_CORE_VERSION, "rl-core-v8")
        self.assertEqual(
            checkpoint_manifest()["observation"]["producer"],
            "PrepareAndSendObservation_rl_native_uint32_quantized",
        )
        self.assertEqual(
            tuple(checkpoint_manifest()["observation"]["egress_features"]),
            EGRESS_FEATURE_NAMES,
        )
        self.assertEqual(
            EGRESS_FEATURE_NAMES,
            (
                "queue_occupancy",
                "queue_ema",
                "dre_utilization",
                "queue_trend",
                "headroom",
                "link_capacity",
            ),
        )
        self.assertEqual(
            checkpoint_manifest()["observation"]["state_source"],
            "rl_mac_tx_dre_local_queue",
        )
        self.assertEqual(
            checkpoint_manifest()["reward"]["version"],
            "rl_native_safe_advantage_v1",
        )
        self.assertEqual(
            checkpoint_manifest()["reward"]["reorder_sequence_scope"],
            "per_flow",
        )
        self.assertFalse(
            checkpoint_manifest()["observation"]["last_action_used_by_q_network"]
        )
        self.assertEqual(
            checkpoint_manifest()["q_network"],
            "safe_rl_native_prior_plus_shared_bounded_residual",
        )
        self.assertTrue(
            checkpoint_manifest()["flowlet_routing"][
                "fresh_action_each_new_flowlet"
            ]
        )
        self.assertFalse(
            checkpoint_manifest()["flowlet_routing"]["silent_ecmp_fallback"]
        )
        self.assertEqual(
            checkpoint_manifest()["safe_action_prior"]["learned_residual_bound"],
            0.1,
        )
        self.assertFalse(
            checkpoint_manifest()["safe_action_prior"][
                "uses_external_routing_labels"
            ]
        )
        self.assertEqual(
            checkpoint_manifest()["training"]["bootstrap"],
            "double_dqn",
        )
        self.assertEqual(
            checkpoint_manifest()["training"]["scheduler"],
            "per_node_replay_samples",
        )

    def test_gradient_schedule_is_transition_driven(self):
        self.assertEqual(scheduled_gradient_steps(255, 256, 4), 0)
        self.assertEqual(scheduled_gradient_steps(256, 256, 4), 1)
        self.assertEqual(scheduled_gradient_steps(259, 256, 4), 1)
        self.assertEqual(scheduled_gradient_steps(260, 256, 4), 2)
        self.assertEqual(scheduled_gradient_steps(1312, 256, 4), 265)

    def test_target_schedule_is_gradient_driven(self):
        self.assertFalse(should_update_target(99, 100))
        self.assertTrue(should_update_target(100, 100))
        self.assertFalse(should_update_target(101, 100))

    def test_disconnected_terminal_reply_cannot_write_stale_transition(self):
        self.assertFalse(
            is_valid_environment_step(
                connected=False,
                is_control=False,
                action_applied=True,
            )
        )
        self.assertTrue(
            is_valid_environment_step(
                connected=True,
                is_control=False,
                action_applied=True,
            )
        )


class RuntimeTimingManifestTest(unittest.TestCase):
    def test_manifest_records_actual_timing_and_rejects_invalid_timing(self):
        from source.rl_contract import checkpoint_manifest
        m = checkpoint_manifest(flowlet_gap_us=100, dre_tau_us=250)
        self.assertAlmostEqual(m['flowlet_routing']['boundary_seconds'], 100e-6)
        self.assertAlmostEqual(m['observation']['dre_time_constant_seconds'], 250e-6)
        with self.assertRaises(ValueError):
            checkpoint_manifest(flowlet_gap_us=0)
        with self.assertRaises(ValueError):
            checkpoint_manifest(dre_tau_us=float('nan'))


if __name__ == "__main__":
    unittest.main()
