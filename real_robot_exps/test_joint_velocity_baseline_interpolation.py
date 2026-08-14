from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np
import yaml

from real_robot_exps.collect_joint_velocity_baseline import (
    load_joint_velocity_frames,
    resample_joint_velocity_frames,
)


class JointVelocityInterpolationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.robot_path = Path("s11-d07_robot.parquet")
        if not cls.robot_path.exists():
            raise unittest.SkipTest(f"Missing diagnostic input: {cls.robot_path}")

        config_path = Path("real_robot_exps/config.yaml")
        with config_path.open("r", encoding="utf-8") as stream:
            cls.config = yaml.safe_load(stream)

        cls.rows, cls.velocities, cls.timestamps = load_joint_velocity_frames(cls.robot_path)
        cls.replay_rows_1k, cls.replay_velocities_1k, cls.replay_t_1k = (
            resample_joint_velocity_frames(cls.rows, cls.velocities, cls.timestamps, 1000.0)
        )

    def test_interpolation_is_finite_and_smoother_than_source(self) -> None:
        source_dv = np.abs(np.diff(self.velocities, axis=0))
        replay_dv = np.abs(np.diff(self.replay_velocities_1k, axis=0))

        print("\nsource rows:", len(self.rows))
        print("replay rows:", len(self.replay_velocities_1k))
        print("source dt max:", float(np.max(np.diff(self.timestamps))))
        print("source abs dv max per joint:", np.max(source_dv, axis=0).tolist())
        print("source abs dv p99:", float(np.percentile(source_dv, 99)))
        print("replay abs dv max per joint:", np.max(replay_dv, axis=0).tolist())
        print("replay abs dv p99:", float(np.percentile(replay_dv, 99)))

        self.assertEqual(self.replay_velocities_1k.shape[1], 7)
        self.assertTrue(np.isfinite(self.replay_velocities_1k).all())
        self.assertTrue(np.isfinite(self.replay_t_1k).all())
        self.assertLessEqual(
            float(np.max(replay_dv)),
            float(np.max(source_dv)) + 1e-12,
        )

    def test_command_stream_saturates_current_delta_limit(self) -> None:
        max_delta = float(self.config["robot"]["joint_velocity_max_delta_per_step"])
        control_dt = 0.001
        max_accel = max_delta / control_dt
        max_jerk = 1000.0
        max_accel_step = max_jerk * control_dt

        dq_cmd = self.replay_velocities_1k[0].astype(np.float64).copy()
        accel_cmd = np.zeros(7, dtype=np.float64)
        streamed = np.empty_like(self.replay_velocities_1k)
        streamed[0] = dq_cmd

        for idx in range(1, len(self.replay_velocities_1k)):
            target_dq = self.replay_velocities_1k[idx]
            for joint in range(7):
                error = target_dq[joint] - dq_cmd[joint]
                desired_accel = max(-max_accel, min(max_accel, error / control_dt))
                accel_delta = desired_accel - accel_cmd[joint]
                accel_delta = max(-max_accel_step, min(max_accel_step, accel_delta))
                accel_cmd[joint] += accel_delta
                dq_cmd[joint] += accel_cmd[joint] * control_dt
            streamed[idx] = dq_cmd

        streamed_dv = np.abs(np.diff(streamed, axis=0))
        print("streamed abs dv max per joint:", np.max(streamed_dv, axis=0).tolist())
        print("streamed abs dv p99:", float(np.percentile(streamed_dv, 99)))

        self.assertTrue(np.isfinite(streamed).all())
        self.assertAlmostEqual(float(np.max(streamed_dv)), max_delta, places=12)


if __name__ == "__main__":
    unittest.main()
