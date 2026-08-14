import os
import sys
import tempfile
import unittest

import numpy as np
import tensorflow as tf


PRISMA_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PRISMA_DIR not in sys.path:
    sys.path.insert(0, PRISMA_DIR)

from source.learner import DQN_AGENT  # noqa: E402
from source.models import DQN_buffer_model  # noqa: E402
from source.rl_contract import RL_CORE_VERSION, native_action_prior  # noqa: E402
from source.utils import load_model, save_all_models  # noqa: E402


class DqnCoreTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.agent = DQN_AGENT(
            q_func=DQN_buffer_model,
            observation_shape=(50,),
            num_actions=8,
            num_nodes=16,
            lr=1e-4,
            input_size_splits=[1, 1, 48],
            neighbors_degrees=[8] * 8,
            gamma=0.9,
        )

    def test_q_decomposition_has_linear_value_and_bounded_residual(self):
        self.assertEqual(
            self.agent.q_network.get_layer("state_value").activation.__name__,
            "linear",
        )
        self.assertEqual(
            self.agent.q_network.get_layer(
                "learned_action_residual_raw"
            ).activation.__name__,
            "tanh",
        )

    @staticmethod
    def _observation_from_ports(ports, destination=0):
        observation = np.zeros((1, 50), dtype=np.float32)
        observation[0, 0] = destination
        observation[0, 2:] = np.asarray(ports, dtype=np.float32).reshape(-1)
        return observation

    def test_initial_q_values_follow_rl_native_prior(self):
        idle_ports = np.array(
            [[0.0, 0.0, 0.0, 0.5, 1.0, 1.0]] * 4
            + [[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]] * 4,
            dtype=np.float32,
        )
        q_values = self.agent.q_network(
            self._observation_from_ports(idle_ports)
        ).numpy()[0]
        self.assertGreater(float(np.min(q_values[:4])), float(np.max(q_values[4:])))
        self.assertGreater(float(np.mean(q_values[:4]) - np.mean(q_values[4:])), 0.2)

    def test_prior_only_ablation_matches_checkpoint_prior_layer(self):
        rng = np.random.RandomState(17)
        observation = np.zeros((3, 50), dtype=np.float32)
        observation[:, 0] = [0, 3, 7]
        observation[:, 2:] = rng.uniform(0, 1, size=(3, 48))
        prior_model = tf.keras.Model(
            inputs=self.agent.q_network.input,
            outputs=self.agent.q_network.get_layer("rl_native_action_prior").output,
        )
        np.testing.assert_allclose(
            native_action_prior(observation, num_actions=8),
            prior_model(observation).numpy(),
            atol=1e-7,
        )

    def test_saturated_fast_links_cannot_outrank_idle_slow_links(self):
        # Even the maximum difference between two learned residuals is 0.2.
        # The native-prior gap in this safety case is 0.62, so no trainable
        # weights can invert the required ordering.
        ports = np.array(
            [[0.9, 0.9, 1.0, 1.0, 0.0, 1.0]] * 4
            + [[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]] * 4,
            dtype=np.float32,
        )
        q_values = self.agent.q_network(
            self._observation_from_ports(ports)
        ).numpy()[0]
        self.assertGreater(float(np.min(q_values[4:])), float(np.max(q_values[:4])))

    def test_one_congested_fast_link_loses_to_idle_peer_fast_links(self):
        ports = np.array(
            [[0.0, 0.0, 0.0, 0.5, 1.0, 1.0]] * 4
            + [[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]] * 4,
            dtype=np.float32,
        )
        ports[0] = [0.9, 0.9, 1.0, 1.0, 0.0, 1.0]
        q_values = self.agent.q_network(
            self._observation_from_ports(ports)
        ).numpy()[0]
        self.assertLess(float(q_values[0]), float(np.min(q_values[1:4])))

    def test_shared_action_scorer_is_permutation_equivariant(self):
        rng = np.random.RandomState(7)
        observation = np.zeros((1, 50), dtype=np.float32)
        observation[0, 0] = 3
        previous_action = 2
        observation[0, 1] = previous_action / 7.0
        ports = rng.uniform(0, 1, size=(8, 6)).astype(np.float32)
        observation[0, 2:] = ports.reshape(-1)

        permutation = np.array([2, 0, 1, 3, 5, 4, 7, 6])
        permuted = observation.copy()
        permuted[0, 2:] = ports[permutation].reshape(-1)
        permuted_previous = int(np.where(permutation == previous_action)[0][0])
        permuted[0, 1] = permuted_previous / 7.0

        original_q = self.agent.q_network(observation).numpy()[0]
        permuted_q = self.agent.q_network(permuted).numpy()[0]
        np.testing.assert_allclose(
            permuted_q,
            original_q[permutation],
            atol=1e-6,
        )

    def test_last_action_cannot_create_a_port_identity_bias(self):
        rng = np.random.RandomState(11)
        observation = np.zeros((1, 50), dtype=np.float32)
        observation[0, 0] = 4
        observation[0, 2:] = rng.uniform(0, 1, size=48)
        previous_zero = observation.copy()
        previous_zero[0, 1] = 0.0
        previous_seven = observation.copy()
        previous_seven[0, 1] = 1.0
        np.testing.assert_allclose(
            self.agent.q_network(previous_zero).numpy(),
            self.agent.q_network(previous_seven).numpy(),
            atol=1e-7,
        )
        self.assertNotIn(
            "last_action_indicator",
            [layer.name for layer in self.agent.q_network.layers],
        )

    def test_target_starts_equal_to_online_network(self):
        for online, target in zip(
            self.agent.q_network.get_weights(), self.agent.target_q_network.get_weights()
        ):
            np.testing.assert_array_equal(online, target)

    def test_requested_epsilon_is_applied_to_current_step(self):
        observation = np.zeros((1, 50), dtype=np.float32)
        self.agent.step(observation, stochastic=True, update_eps=1.0)
        self.assertEqual(float(self.agent.eps.numpy()), 1.0)

    def test_action_diagnostics_separate_behavior_and_greedy_policy(self):
        observation = np.zeros((1, 50), dtype=np.float32)
        chosen, greedy, used_random, q_values = self.agent.step(
            observation,
            stochastic=True,
            update_eps=0.0,
            return_diagnostics=True,
        )
        self.assertFalse(bool(used_random.numpy().item()))
        self.assertEqual(chosen.numpy().item(), greedy.numpy().item())
        self.assertEqual(tuple(q_values.shape), (1, 8))

        _, _, used_random, _ = self.agent.step(
            observation,
            stochastic=True,
            update_eps=1.0,
            return_diagnostics=True,
        )
        self.assertTrue(bool(used_random.numpy().item()))

    def test_checkpoint_round_trip_requires_matching_contract(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = tmp_dir + os.sep
            save_all_models({0: self.agent}, [0], "unit", 1, 1, root=root)
            checkpoint_path = os.path.join(tmp_dir, "unit", "final")
            restored = load_model(
                checkpoint_path,
                node_index=0,
                expected_contract_version=RL_CORE_VERSION,
            )
            self.assertIsNotNone(restored[0])

            os.remove(os.path.join(checkpoint_path, "checkpoint_manifest.json"))
            with self.assertRaisesRegex(ValueError, "no RL contract manifest"):
                load_model(
                    checkpoint_path,
                    node_index=0,
                    expected_contract_version=RL_CORE_VERSION,
                )


if __name__ == "__main__":
    unittest.main()
