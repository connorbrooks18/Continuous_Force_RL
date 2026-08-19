# Setup

- Run lfd_apples as on the connorbrooks18 github
```
ros2 launch lfd_apples lfd_gripper.launch.py ssid:=alejos password:=harvesting
```
- Power on the gripper, and it should connect
- Make sure the pressurized air is attatched
- In franka desk, unlock joints, go to exec mode, and activate FCI
- Make sure the correct end-effector "Dual mode (connor)" is selected in settings of Desk
- Realsense camera should be connected to computer via usb c and located as close as possible to proxy

## Direction Choice

- Directions in real_robot_exps/directions.json are used
- Defined by (altitude, azimuth) in radians. More info in README.md

## Camera to Base Calibration

```
ros2 launch charuco eye_on_base_calib.launch.py
```

- Attach the checkerboard of aruco markers to the gripper
- In Franka, make sure to select "No End Effector"
- charuco ros command should pull up GUI, prompting to take sample pictures
- Take 15-25 pictures, varying the orientation of the board. 
- Make sure to press the save button at the end
- Then, you'll have to run the below command and paste the outputted matrix into real_robot_exps/static_constants.py

```
python3 real_robot_exps/calibrate_camera_to_base.py 
```

### Confirming Calibration

- With Franka in program mode and with the grippper reattached and end-effector settings set accordingly, move the arm up to the apple on the proxy. 
- Run the below command with Arm in Exec mode (with FCI enabled) and check that the apple-tcp distance is reasonable
```
python3 real_robot_exps/print_apple_tcp_base.py 
```



# Main Command:

```
python3 -m real_robot_exps.runner --skip-enter --record --no-manual-setup --stops 5 --distance .05
```

- You'll be prompted to choose a structure. If a new structure, enter "n" or "no," and it will prompt you to create a new one. If a new apple, you might have to add new constants directly to real_robot_exps/structure_constants.json

- If it fails, you can pick it back up at a chosen direction index with --start-at N

```
python3 -m real_robot_exps.runner --skip-enter --record --no-manual-setup --stops 5 --distance .05 --start-at 3
```

# Inspecting Parquet Files

- To look at the metadata and the first N rows, you can use the below command and look at the new plaintext file
```
python3 real_robot_exps/dump_parquet_preview.py --rows N s09-d00.parquet > FILENAME
```




