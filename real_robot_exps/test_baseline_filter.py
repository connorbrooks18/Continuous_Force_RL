import unittest

import numpy as np
from scipy.signal import welch

from real_robot_exps.collect_joint_velocity_baseline import lowpass_replay_velocities


class ReplayLowpassTest(unittest.TestCase):
    def setUp(self):
        self.rate = 1000.0
        t = np.arange(0.0, 6.0, 1.0 / self.rate)
        # slow quasi-static step motion (what the pull does) + 20 Hz controller jitter
        slow = 0.05 * (np.tanh(4 * (t - 1.5)) - np.tanh(4 * (t - 4.5)))
        self.t = t
        self.slow = np.column_stack([slow] * 7)
        self.noisy = self.slow + 0.02 * np.sin(2 * np.pi * 20.0 * t)[:, None]

    def test_removes_jitter_and_keeps_the_motion(self):
        filtered = lowpass_replay_velocities(self.noisy, self.rate, 5.0)
        f, p = welch(filtered[:, 0], fs=self.rate, nperseg=1024)
        self.assertLess(p[f > 10].sum() / p.sum(), 0.001)
        np.testing.assert_allclose(filtered, self.slow, atol=2e-3)
        np.testing.assert_allclose(np.trapz(filtered, self.t, axis=0), np.trapz(self.slow, self.t, axis=0), atol=1e-3)
        self.assertEqual(filtered.shape, self.noisy.shape)

    def test_zero_cutoff_replays_raw(self):
        self.assertIs(lowpass_replay_velocities(self.noisy, self.rate, 0.0), self.noisy)


if __name__ == "__main__":
    unittest.main()
