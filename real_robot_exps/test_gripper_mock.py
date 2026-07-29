import builtins
import unittest

from real_robot_exps.gripper_test import GripperClient


class GripperMockTest(unittest.TestCase):
    def test_mock_client_does_not_import_ros_or_connect(self):
        original_import = builtins.__import__
        imported = []

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
            if name.startswith("rclpy") or name.startswith("std_srvs"):
                imported.append(name)
                raise AssertionError(f"Mock gripper should not import {name}")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded_import
        try:
            client = GripperClient(mock=True)
            self.assertIsNone(client.send_request(True))
            self.assertIsNone(client.send_request(False))
            client.terminate()
        finally:
            builtins.__import__ = original_import

        self.assertEqual(imported, [])


if __name__ == "__main__":
    unittest.main()
