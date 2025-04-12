#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

class DualArmController(Node):
    def __init__(self):
        super().__init__('dual_arm_controller')

        # Define publishers
        self.left_arm_pub = self.create_publisher(Float64MultiArray, '/target_position', 10)
        self.right_arm_pub = self.create_publisher(Float64MultiArray, '/target_positionR', 10)
        self.left_gripper_pub = self.create_publisher(Float64MultiArray, '/gripper_command', 10)
        self.right_gripper_pub = self.create_publisher(Float64MultiArray, '/gripper_commandR', 10)

        # Store the x-coordinate of the Cyl received in real-time in the coordinate system of the manipulators
        self.left_x = 0.62  
        self.right_x = 0.62

        # Subscribe to the cyl x coordinates published by TF conversion nodes
        self.create_subscription(Float64MultiArray, '/cyl_in_arm_left_x', self.left_x_callback, 10)
        self.create_subscription(Float64MultiArray, '/cyl_in_arm_right_x', self.right_x_callback, 10)

        # Issue the instruction 
        self.send_commands()

    def left_x_callback(self, msg: Float64MultiArray):
        self.left_x = msg.data[0]
        self.get_logger().info(f"Received left_x: {self.left_x:.2f}")

    def right_x_callback(self, msg: Float64MultiArray):
        self.right_x = msg.data[0]
        self.get_logger().info(f"Received right_x: {self.right_x:.2f}")

    def send_commands(self):
        # Using real-time obtained cyl x coordinates, keeping y and z unchanged
        left_target = Float64MultiArray(data=[self.left_x, -0.1, 0.26])
        right_target = Float64MultiArray(data=[self.left_x, 0.15, 0.26])
        left_gripper = Float64MultiArray(data=[-0.45, 0.45])
        right_gripper = Float64MultiArray(data=[-0.45, 0.45])

        self.left_arm_pub.publish(left_target)
        self.right_arm_pub.publish(right_target)
        self.left_gripper_pub.publish(left_gripper)
        self.right_gripper_pub.publish(right_gripper)

        self.get_logger().info(f"Published left target: {left_target.data}")
        self.get_logger().info(f"Published right target: {right_target.data}")

def main(args=None):
    rclpy.init(args=args)
    node = DualArmController()
    
    # Give some time to ensure the completion of the message release
    rclpy.spin(node)
    
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
