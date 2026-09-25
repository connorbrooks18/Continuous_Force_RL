import unittest
from unittest.mock import patch

from real_robot_exps.field_pull import _open_gripper_with_retries


class FakeGripper:
    def __init__(self, fail_times: int):
        self.fail_times = fail_times
        self.calls = 0

    def send_request(self, value):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise TimeoutError(f"no reply (attempt {self.calls})")


class OpenGripperWithRetriesTest(unittest.TestCase):
    def test_succeeds_after_a_transient_failure(self):
        gripper = FakeGripper(fail_times=2)
        with patch("real_robot_exps.field_pull.time.sleep") as sleep:
            _open_gripper_with_retries(gripper, attempts=3, delay_s=2.0)
        self.assertEqual(gripper.calls, 3)
        self.assertEqual(sleep.call_count, 2)  # waited between attempts 1->2 and 2->3, not after success

    def test_raises_the_last_error_once_attempts_are_exhausted(self):
        gripper = FakeGripper(fail_times=5)
        with patch("real_robot_exps.field_pull.time.sleep"):
            with self.assertRaisesRegex(TimeoutError, "attempt 3"):
                _open_gripper_with_retries(gripper, attempts=3, delay_s=2.0)
        self.assertEqual(gripper.calls, 3)

    def test_first_attempt_succeeding_needs_no_retry(self):
        gripper = FakeGripper(fail_times=0)
        with patch("real_robot_exps.field_pull.time.sleep") as sleep:
            _open_gripper_with_retries(gripper)
        self.assertEqual(gripper.calls, 1)
        sleep.assert_not_called()


if __name__ == "__main__":
    unittest.main()
