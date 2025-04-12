#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Float64MultiArray
import numpy as np
import tf_transformations  
import tf2_ros

class ManualCylArmTransform(Node):
    def __init__(self):
        super().__init__('manual_cyl_arm_transform')

        # Subscribe to Cyl's Odom topic
        self.cyl_sub = self.create_subscription(
            Odometry,
            '/cyl/odom',
            self.cyl_callback,
            10
        )
        # Subscribe to the Odom topic of Blueboat
        self.boat_sub = self.create_subscription(
            Odometry,
            '/blueboat/navigator/odometry',
            self.boat_callback,
            10
        )

        # Used for TF after broadcast conversion
        self.tf_broadcaster = tf2_ros.TransformBroadcaster(self)

        # Publish the x-coordinate of Cyl in the coordinate system of the left and right arms
        self.left_x_pub = self.create_publisher(Float64MultiArray, '/cyl_in_arm_left_x', 10)
        self.right_x_pub = self.create_publisher(Float64MultiArray, '/cyl_in_arm_right_x', 10)

        # Save the latest Odom data of Blueboat
        self.boat_odom = None

    def boat_callback(self, msg):
        self.boat_odom = msg

    def pose_to_mat(self, position, orientation):
        """
        Construct 4x4 homogeneous transformation matrix based on geometriy_msgs/Point and Quaternion
        """
        quat = [orientation.x, orientation.y, orientation.z, orientation.w]
        T = tf_transformations.quaternion_matrix(quat)
        T[0, 3] = position.x
        T[1, 3] = position.y
        T[2, 3] = position.z
        return T

    def mat_to_pose(self, T):
        """
        Extracting translation and quaternions from 4x4 homogeneous matrices
        """
        trans = T[0:3, 3]
        quat = tf_transformations.quaternion_from_matrix(T)
        return trans, quat

    def cyl_callback(self, msg):
        if self.boat_odom is None:
            self.get_logger().warn("No blueboat odom data received yet")
            return

        # Construct the homogeneous transformation matrix of cyl under world_ned 
        T_wcyl = self.pose_to_mat(msg.pose.pose.position, msg.pose.pose.orientation)

        # Construct the homogeneous transformation matrix of the Blueboat under world_ned 
        T_wvehicle = self.pose_to_mat(
            self.boat_odom.pose.pose.position,
            self.boat_odom.pose.pose.orientation
        )

        # Construct fixed transformations from Blueboat to the left and right arms according to the SCN file definition
        # Left arm fixed transformation: translate (0.3, -0.15, 0.1)
        T_va_left = np.eye(4)
        T_va_left[0, 3] = 0.3
        T_va_left[1, 3] = -0.15
        T_va_left[2, 3] = 0.1

        # Right arm fixed transformation: translate (0.3, 0.15, 0.1)
        T_va_right = np.eye(4)
        T_va_right[0, 3] = 0.3
        T_va_right[1, 3] = 0.15
        T_va_right[2, 3] = 0.1

        # Calculate the homogeneous transformation matrix of the left and right arms under world_ned
        T_warm_left = np.dot(T_wvehicle, T_va_left)
        T_warm_right = np.dot(T_wvehicle, T_va_right)

        # Inverse to obtain the transformation from world_ned to the coordinate systems of each arm
        T_arm_left = np.linalg.inv(T_warm_left)
        T_arm_right = np.linalg.inv(T_warm_right)

        # Convert the pose of cyl in world_ned to the coordinate systems of various arms
        T_arm_cyl_left = np.dot(T_arm_left, T_wcyl)
        T_arm_cyl_right = np.dot(T_arm_right, T_wcyl)

        trans_left, quat_left = self.mat_to_pose(T_arm_cyl_left)
        trans_right, quat_right = self.mat_to_pose(T_arm_cyl_right)

        self.get_logger().info(
            "The position of Cyl in the left robot arm coordinate system: x=%.2f, y=%.2f, z=%.2f" %
            (trans_left[0], trans_left[1], trans_left[2])
        )
        self.get_logger().info(
            "The position of Cyl in the right robot arm coordinate system: x=%.2f, y=%.2f, z=%.2f" %
            (trans_right[0], trans_right[1], trans_right[2])
        )

        # Broadcast TF
        t_left = TransformStamped()
        t_left.header.stamp = msg.header.stamp
        t_left.header.frame_id = "arm_left"
        t_left.child_frame_id = "Cyl_in_arm_left"
        t_left.transform.translation.x = trans_left[0]
        t_left.transform.translation.y = trans_left[1]
        t_left.transform.translation.z = trans_left[2]
        t_left.transform.rotation.x = quat_left[0]
        t_left.transform.rotation.y = quat_left[1]
        t_left.transform.rotation.z = quat_left[2]
        t_left.transform.rotation.w = quat_left[3]
        self.tf_broadcaster.sendTransform(t_left)

        t_right = TransformStamped()
        t_right.header.stamp = msg.header.stamp
        t_right.header.frame_id = "arm_right"
        t_right.child_frame_id = "Cyl_in_arm_right"
        t_right.transform.translation.x = trans_right[0]
        t_right.transform.translation.y = trans_right[1]
        t_right.transform.translation.z = trans_right[2]
        t_right.transform.rotation.x = quat_right[0]
        t_right.transform.rotation.y = quat_right[1]
        t_right.transform.rotation.z = quat_right[2]
        t_right.transform.rotation.w = quat_right[3]
        self.tf_broadcaster.sendTransform(t_right)

        # Publish the x-coordinate of Cyl in the coordinate system of the left and right arms
        left_x_msg = Float64MultiArray(data=[trans_left[0]])
        right_x_msg = Float64MultiArray(data=[trans_right[0]])
        self.left_x_pub.publish(left_x_msg)
        self.right_x_pub.publish(right_x_msg)

def main(args=None):
    rclpy.init(args=args)
    node = ManualCylArmTransform()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
