import tempfile
import unittest
from pathlib import Path

import numpy as np

from real_robot_exps.ee_profiles import (
    identify,
    load_profiles,
    profile_differences,
    require_profile,
    save_profile,
)


def _state(mass, z_offset, yaw_deg=0.0, com=(0.0, 0.0, 0.05)):
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    pose = np.eye(4)
    pose[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    pose[2, 3] = z_offset
    return {"F_T_NE": pose.tolist(), "m_ee": mass, "F_x_Cee": list(com), "I_ee": [0.001] * 9}


BOARD = _state(0.35, 0.02)
GRIPPER = _state(0.9, 0.1034, yaw_deg=-45.0, com=(0.0, 0.0, 0.07))


class ProfileMatchTest(unittest.TestCase):
    def test_identify_and_explain(self):
        profiles = {"board": BOARD, "gripper": GRIPPER}
        self.assertEqual(identify(GRIPPER, profiles), "gripper")
        self.assertEqual(identify(_state(0.905, 0.1036, yaw_deg=-45.2, com=(0, 0, 0.071)), profiles), "gripper")
        reasons = profile_differences(BOARD, GRIPPER)
        self.assertTrue(any("mass" in r for r in reasons))
        self.assertTrue(any("offset" in r for r in reasons))
        self.assertTrue(any("rotation" in r for r in reasons))
        self.assertIsNone(identify(_state(2.0, 0.3), profiles))


class RequireProfileTest(unittest.TestCase):
    def _run(self, answers, states, profiles_path):
        answers, states, said = list(answers), list(states), []
        result = require_profile(
            "gripper", config={}, ask=lambda prompt: answers.pop(0), say=said.append,
            profiles_path=profiles_path, read_state=lambda config: states.pop(0),
        )
        return result, said, answers, states

    def test_waits_until_the_operator_switches_profiles(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ee_profiles.yaml"
            save_profile("board", BOARD, path)
            save_profile("gripper", GRIPPER, path)
            result, said, answers, states = self._run([""], [BOARD, GRIPPER], path)
            self.assertEqual(result["profile"], "gripper")
            self.assertIn("'board'", said[0])  # told which profile is active
            self.assertEqual((answers, states), ([], []))

    def test_captures_a_missing_profile_then_checks_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ee_profiles.yaml"
            result, _, _, _ = self._run([""], [GRIPPER, GRIPPER], path)
            self.assertEqual(result["profile"], "gripper")
            self.assertIn("gripper", load_profiles(path))

    def test_operator_can_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "ee_profiles.yaml"
            save_profile("gripper", GRIPPER, path)
            result, _, _, _ = self._run(["s"], [BOARD], path)
            self.assertTrue(result["skipped"])
            self.assertEqual(result["expected"], "gripper")


if __name__ == "__main__":
    unittest.main()
