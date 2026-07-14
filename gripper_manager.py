"""
Unified Gripper Manager

Orchestrates the Wi-Fi hotspot, micro-ROS agent, gripper controller, 
and provides a direct command line interface to control the physical gripper.

Usage:
    # 1. Start everything (Hotspot, Agent, Gripper Node) in the background:
    python gripper_manager.py setup

    # 2. Open / Close / Test commands:
    python gripper_manager.py open
    python gripper_manager.py close
    python gripper_manager.py test
"""

import argparse
import subprocess
import sys
import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool


class GripperDirectClient(Node):
    """Simple ROS 2 client to communicate with the gripper_grab service."""
    def __init__(self):
        super().__init__("gripper_direct_client")
        self.client = self.create_client(SetBool, "gripper_grab")
        
        # Wait up to 5 seconds for the service to be live
        timeout = 5.0
        self.get_logger().info("Connecting to gripper_grab service...")
        if not self.client.wait_for_service(timeout_sec=timeout):
            self.get_logger().error("Gripper service is not running! Run 'setup' first.")
            sys.exit(1)

    def send_command(self, grab: bool) -> bool:
        req = SetBool.Request()
        req.data = grab
        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response and response.success:
            self.get_logger().info(f"Command Success: {response.message}")
            return True
        self.get_logger().error("Command Failed.")
        return False


def start_system_infrastructure(ssid: str, password: str, interface: str):
    """Launches the Wi-Fi hotspot, micro-ROS agent, and gripper node."""
    print("=" * 80)
    print("STARTING SYSTEM INFRASTRUCTURE")
    print("=" * 80)

    # 1. Start Hotspot
    print(f"[*] Activating Wi-Fi Hotspot on {interface} (SSID: {ssid})...")
    hotspot_cmd = [
        "nmcli", "device", "wifi", "hotspot",
        "ifname", interface, "ssid", ssid, "password", password
    ]
    subprocess.Popen(hotspot_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2.0)

    # 2. Start micro-ROS Agent
    print("[*] Starting micro-ROS Agent (UDP Port 8888)...")
    agent_cmd = [
        "ros2", "run", "micro_ros_agent", "micro_ros_agent", "udp4", "--port", "8888"
    ]
    subprocess.Popen(agent_cmd)
    time.sleep(2.0)

    # 3. Start Gripper ROS Node
    print("[*] Launching manual gripper controller node...")
    gripper_node_cmd = [
        "ros2", "run", "lfd_apples", "lfd_automatic_gripper"
    ]
    # Run in background; it will keep hosting the service
    subprocess.Popen(gripper_node_cmd)
    
    print("\n[✔] System is now fully set up and running in the background!")
    print("Keep this terminal open, or use another terminal to run open/close commands.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down background services...")


def run_gripper_command(action: str):
    """Initializes ROS 2 briefly to send the grab/release command."""
    rclpy.init()
    client = GripperDirectClient()

    if action == "close":
        print("[*] Sending GRAB (Close) request...")
        client.send_command(True)
    elif action == "open":
        print("[*] Sending RELEASE (Open) request...")
        client.send_command(False)
    elif action == "test":
        print("[*] Running cycle test (Close -> Wait 3s -> Open)...")
        if client.send_command(True):
            time.sleep(3.0)
            client.send_command(False)

    client.destroy_node()
    rclpy.shutdown()


def main():
    parser = argparse.ArgumentParser(description="Unified Gripper Manager")
    parser.add_argument(
        "action", 
        choices=["setup", "open", "close", "test"], 
        help="Action to perform. 'setup' spins up background processes. 'open'/'close'/'test' send immediate commands."
    )
    parser.add_argument("--ssid", type=str, default="alejos", help="Hotspot SSID")
    parser.add_argument("--password", type=str, default="harvesting", help="Hotspot password")
    parser.add_argument("--ifname", type=str, default="wlo1", help="Network interface name")
    
    args = parser.parse_args()

    if args.action == "setup":
        start_system_infrastructure(args.ssid, args.password, args.ifname)
    else:
        run_gripper_command(args.action)


if __name__ == "__main__":
    main()