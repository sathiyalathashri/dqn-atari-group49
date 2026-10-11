"""NumPy-only checks: python -m unittest discover -s tests -p test_tabular_q_demo.py."""
import unittest

import numpy as np

from tabular_q_demo import (
    GAMMA, GOAL, PIT, STATES, greedy_rollout, optimal_action_count,
    q_learning_with_replay, step, value_iteration,
)


class TabularDemoTests(unittest.TestCase):
    def test_boundaries_rewards_and_absorbing_terminals(self):
        self.assertEqual(step((0, 0), 0), ((0, 0), -0.01, False))
        self.assertEqual(step((4, 3), 3), (GOAL, 1.0, True))
        self.assertEqual(step((2, 1), 3), (PIT, -1.0, True))
        for terminal in (GOAL, PIT):
            for action in range(4):
                self.assertEqual(step(terminal, action), (terminal, 0.0, True))

    def test_reference_satisfies_bellman_equation(self):
        q = value_iteration()
        for s in STATES:
            for a in range(4):
                s2, reward, done = step(s, a)
                target = reward + (0 if done else GAMMA * q[s2].max())
                self.assertAlmostEqual(q[s][a], target, places=9)
        np.testing.assert_array_equal(q[GOAL], 0)
        np.testing.assert_array_equal(q[PIT], 0)
        path, total, outcome = greedy_rollout(q)
        self.assertEqual((len(path) - 1, outcome), (8, "goal"))
        self.assertAlmostEqual(total, 0.93)

    def test_equally_optimal_actions_count_as_matches(self):
        q = value_iteration()
        alternate = np.zeros_like(q)
        for s in STATES:
            best = np.flatnonzero(np.isclose(q[s], q[s].max(), rtol=0, atol=1e-8))
            alternate[s][best[-1]] = 1
        self.assertEqual(optimal_action_count(q, alternate), len(STATES))

    def test_seeded_replay_wraparound_and_small_buffer(self):
        options = dict(episodes=20, seed=17, replay_capacity=3, batch=8, max_steps=10)
        first = q_learning_with_replay(**options)
        np.testing.assert_array_equal(first, q_learning_with_replay(**options))
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first[GOAL], 0)
        np.testing.assert_array_equal(first[PIT], 0)

    def test_invalid_parameters(self):
        for options in ({"episodes": 0}, {"batch": -1}, {"eps": 1.1},
                        {"lr": 0}, {"lr": float("nan")}, {"max_steps": 0},
                        {"replay_capacity": 0}, {"episodes": 1.5}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                q_learning_with_replay(**options)
        with self.assertRaises(ValueError):
            step((0, 0), -1)
        with self.assertRaises(ValueError):
            value_iteration(tol=-1)

    def test_rollout_detects_loop(self):
        path, total, outcome = greedy_rollout(np.zeros((5, 5, 4)))
        self.assertEqual(outcome, "loop")
        self.assertEqual(len(path), 2)
        self.assertEqual(total, -0.01)


if __name__ == "__main__":
    unittest.main()
