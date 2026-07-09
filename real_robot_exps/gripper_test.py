import sys
import time
import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool

class GripperClient(Node):
    def __init__(self):
        rclpy.init()
        super().__init__("gripper_grab_client")
        self.client = self.create_client(SetBool, "gripper_grab")
        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting...')
            
    def send_request(self, grab: bool):
        req = SetBool.Request()
        req.data = grab
        future = self.client.call_async(req)
        
        # Spin until the service returns a response safely without deadlocking
        rclpy.spin_until_future_complete(self, future)
        return future.result()
    
    def terminate(self):
        self.destroy_node()
        rclpy.shutdown()

def main():
    gc = GripperClient()

    grab = False
    if(len(sys.argv) > 1 and sys.argv[1] == "o"):
        grab = True
    
    if grab:
        # 1. Grab
        print("Sending Grab Request...")
        response = gc.send_request(True)
        if response and response.success:
            print("Grab command accepted!")
        
    else:
        # 2. Let go
        print("Sending Release Request...")
        response = gc.send_request(False)
        if response and response.success:
            print("Release command accepted!")

   

if __name__ == '__main__':
    main()