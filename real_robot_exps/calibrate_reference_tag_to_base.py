"""Compatibility wrapper for the renamed camera-to-base calibration helper."""

from real_robot_exps.calibrate_camera_to_base import *  # noqa: F401,F403


if __name__ == "__main__":
    from real_robot_exps.calibrate_camera_to_base import main

    main()
