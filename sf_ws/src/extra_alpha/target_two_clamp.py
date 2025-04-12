#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
- The main body of the manipulator contains 4 degrees of freedom (α _ axis_e, α _ axis_d, α _ axis_c, α _ axis_b), and its end position is solved by 4-axis IK
- The end effector consists of two joints (α _ axis_a1, α _ axis_a2), which are controlled separately through the/ripper_command theme
- The base coordinate transformation and joint configuration are consistent with the SCN file
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import numpy as np
from scipy.optimize import minimize
import threading
import time

#################### Math Utility Functions ####################
def rot_x(theta):
    return np.array([
        [1, 0,           0,          0],
        [0, np.cos(theta), -np.sin(theta), 0],
        [0, np.sin(theta),  np.cos(theta), 0],
        [0, 0,           0,          1]
    ])

def rot_y(theta):
    return np.array([
        [ np.cos(theta), 0, np.sin(theta), 0],
        [0,             1, 0,             0],
        [-np.sin(theta), 0, np.cos(theta), 0],
        [0,             0, 0,             1]
    ])

def rot_z(theta):
    return np.array([
        [np.cos(theta), -np.sin(theta), 0, 0],
        [np.sin(theta),  np.cos(theta), 0, 0],
        [0,             0,             1, 0],
        [0,             0,             0, 1]
    ])

def trans(x, y, z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ])

def joint_transform(joint_angle, joint_axis, joint_origin, joint_rpy):
    """
    Calculate the homogeneous transformation matrix of a single joint
    1. Translate to the joint origin joint_origin
    2. Apply fixed rotation joint_rpy
    3. Rotate jointAngle around the joint axis jointAxis
    """
    T = trans(*joint_origin)
    R_fixed = rot_z(joint_rpy[2]) @ rot_y(joint_rpy[1]) @ rot_x(joint_rpy[0])
    T_fixed = T @ R_fixed
    if np.allclose(joint_axis, [1,0,0]):
        T_joint = rot_x(joint_angle)
    elif np.allclose(joint_axis, [0,1,0]):
        T_joint = rot_y(joint_angle)
    elif np.allclose(joint_axis, [0,0,1]):
        T_joint = rot_z(joint_angle)
    else:
        raise ValueError(f"Unsupported joint axis {joint_axis}")
    return T_fixed @ T_joint

#################### Forward Kinematics (4-axis main body) ####################
def forward_kinematics_4(joint_angles):
    """
    Calculate the end position based on four joint angles (α _ axis_e, α _ axis_d, α _ axis_c, α _ axis_b),
    Simultaneously consider the fixed base coordinate transformation from Vehicle to Arm.
    """
    # Base coordinate conversion (fixed conversion from Blueboat to Arm)
    T_base = trans(0.3, -0.15, 0.1) @ rot_x(3.14)
    T = T_base

    # Parameters of 4 joints
    joint_axes = [
        [0, 0, 1],      # α_axis_e
        [0, 1, 0],      # α_axis_d
        [0, 1, 0],      # α_axis_c
        [1, 0, 0],      # α_axis_b
    ]
    joint_origins = [
        [0.120683, -0.021096, -0.025],  # α_axis_e
        [0.023,    0.0,       0.033],    # α_axis_d
        [0.13,    -0.015,     0.08],     # α_axis_c
        [0.031,    0.0165,   -0.0225],    # α_axis_b
    ]
    joint_rpys = [
        [3.14, 0, 0],  # α_axis_e
        [0,    0, 0],  # α_axis_d
        [0,    0, 0],  # α_axis_c
        [0,    0, 0],  # α_axis_b
    ]

    for i in range(4):
        T_joint = joint_transform(joint_angles[i], joint_axes[i], joint_origins[i], joint_rpys[i])
        T = T @ T_joint

    pos = T[:3, 3]
    return pos

def forward_kinematics_4R(joint_angles):
    """
    Calculate the end position based on four joint angles (α _ axis_eR, α _ axis_dR, α _ axis_cR, α _ axis_bR),
    Simultaneously consider the fixed base coordinate transformation from Vehicle to Arm.
    """
    # Base coordinate transformation
    T_base = trans(0.3, 0.15, 0.1) @ rot_x(3.14)
    T = T_base

    # Parameters of 4 joints
    joint_axes = [
        [0, 0, 1],      # α_axis_eR
        [0, 1, 0],      # α_axis_dR
        [0, 1, 0],      # α_axis_cR
        [1, 0, 0],      # α_axis_bR
    ]
    joint_origins = [
        [0.120683, -0.021096, -0.025],  # α_axis_eR
        [0.023,    0.0,       0.033],    # α_axis_dR
        [0.13,    -0.015,     0.08],     # α_axis_cR
        [0.031,    0.0165,   -0.0225],    # α_axis_bR
    ]
    joint_rpys = [
        [3.14, 0, 0],  # α_axis_eR
        [0,    0, 0],  # α_axis_dR
        [0,    0, 0],  # α_axis_cR
        [0,    0, 0],  # α_axis_bR
    ]

    for i in range(4):
        T_joint = joint_transform(joint_angles[i], joint_axes[i], joint_origins[i], joint_rpys[i])
        T = T @ T_joint

    pos = T[:3, 3]
    return pos

#################### IK objective function (4-axis) ####################
def ik_cost_function_4(joint_angles, target_position):
    pos = forward_kinematics_4(joint_angles)
    return np.linalg.norm(pos - target_position)

def ik_cost_function_4R(joint_angles, target_position):
    posR = forward_kinematics_4R(joint_angles)
    return np.linalg.norm(posR - target_position)

#################### 4-axis joint limitation ####################
joint_limits_4 = [
    (-1.0, 1.0),      # α_axis_e
    (-1.58, 1.58),    # α_axis_d
    (-3.15, 3.15),    # α_axis_c
    (0.0, 3.0),       # α_axis_b
]

#################### Inverse Kinematics (4-axis) ####################
def inverse_kinematics_4(target_position, initial_guess):
    result = minimize(
        ik_cost_function_4,
        initial_guess,
        args=(target_position,),
        bounds=joint_limits_4,
        method='SLSQP',
        options={'ftol': 1e-6, 'maxiter': 10000}
    )
    if result.success:
        return result.x
    else:
        print("Optimization failed:", result.message)
        return None

def inverse_kinematics_4R(target_position, initial_guess):
    result = minimize(
        ik_cost_function_4R,
        initial_guess,
        args=(target_position,),
        bounds=joint_limits_4,
        method='SLSQP',
        options={'ftol': 1e-6, 'maxiter': 10000}
    )
    if result.success:
        return result.x
    else:
        print("Right Optimization failed:", result.message)
        return None

#################### Interpolation function ####################
def interpolate_points(start, end, num_points):
    return [start + (end - start) * i / (num_points - 1) for i in range(num_points)]

#################### ROS2 Node ####################
class IKSolver4DofNode(Node):
    def __init__(self):
        super().__init__('ik_solver_4dof')
        self.declare_parameter('joint_offsets', [0.0, 0.0, 0.0, 0.0])
        self.joint_offsets = self.get_parameter('joint_offsets').value
        
        self.joint_sub = self.create_subscription(JointState, '/blueboat/alpha/joint_states', self.joint_states_callback, 10)
        self.joint_pub = self.create_publisher(JointState, '/blueboat/alpha/desired_joint_states', 10)
        
        self.target_sub = self.create_subscription(Float64MultiArray, '/target_position', self.target_position_callback, 10)
        self.target_subR = self.create_subscription(Float64MultiArray, '/target_positionR', self.target_position_callbackR, 10)
        
        self.gripper_sub = self.create_subscription(Float64MultiArray, '/gripper_command', self.gripper_command_callback, 10)
        self.gripper_subR = self.create_subscription(Float64MultiArray, '/gripper_commandR', self.gripper_command_callbackR, 10)
        
        self.joint_names = ['blueboat/alpha_axis_e', 'blueboat/alpha_axis_d', 'blueboat/alpha_axis_c', 'blueboat/alpha_axis_b', 'blueboat/alpha_axis_a1', 'blueboat/alpha_axis_a2']
        self.joint_namesR = ['blueboat/alpha_axis_eR', 'blueboat/alpha_axis_dR', 'blueboat/alpha_axis_cR', 'blueboat/alpha_axis_bR', 'blueboat/alpha_axis_a1R', 'blueboat/alpha_axis_a2R']
        
        self.current_4dof = [0.0] * 4
        self.current_4dofR = [0.0] * 4
        self.gripper_angles = [-0.9, 0.9]
        self.gripper_anglesR = [-0.9, 0.9]
        self.lock = threading.Lock()
    




    def joint_states_callback(self, msg):
        with self.lock:
            positions = dict(zip(msg.name, msg.position))
            self.current_4dof = [positions.get(name, 0.0) for name in self.joint_names[:4]]
            self.current_4dofR = [positions.get(name, 0.0) for name in self.joint_namesR[:4]]

    def publish_joint_states(self):
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = self.joint_names + self.joint_namesR
        js.position = self.current_4dof + self.gripper_angles + self.current_4dofR + self.gripper_anglesR
        self.joint_pub.publish(js)

    def execute_trajectory(self, start, end, guess, is_right=False):
        num_points = 300 # The number of interpolation points
        time_step = 0.07 # Interpolation time interval

        points = interpolate_points(start, end, num_points)

        for pt in points:
            if is_right:
                sol = inverse_kinematics_4R(pt, guess)
                if sol is not None:
                    with self.lock:
                        self.current_4dofR = [s + o for s, o in zip(sol, self.joint_offsets)]
            else:
                sol = inverse_kinematics_4(pt, guess)
                if sol is not None:
                    with self.lock:
                        self.current_4dof = [s + o for s, o in zip(sol, self.joint_offsets)]

            self.publish_joint_states()
            guess = sol
            time.sleep(time_step)

        # Forward kinematics error verification
        with self.lock:
            if is_right:
                final_pos = forward_kinematics_4R(self.current_4dofR)
                error = np.linalg.norm(final_pos - end)
                self.get_logger().info(f"Right Arm - Final FK position: {final_pos}")
                self.get_logger().info(f"Right Arm - Target position: {end}")
                self.get_logger().info(f"Right Arm - Position error: {error:.6f}")
            else:
                final_pos = forward_kinematics_4(self.current_4dof)
                error = np.linalg.norm(final_pos - end)
                self.get_logger().info(f"Left Arm - Final FK position: {final_pos}")
                self.get_logger().info(f"Left Arm - Target position: {end}")
                self.get_logger().info(f"Left Arm - Position error: {error:.6f}")


    def target_position_callback(self, msg):
        threading.Thread(target=self.execute_trajectory, args=(forward_kinematics_4(self.current_4dof), np.array(msg.data), self.current_4dof)).start()

    def target_position_callbackR(self, msg):
        threading.Thread(target=self.execute_trajectory, args=(forward_kinematics_4R(self.current_4dofR), np.array(msg.data), self.current_4dofR, True)).start()

    def gripper_command_callback(self, msg):
        if len(msg.data) != 2:
            self.get_logger().error("Gripper command must have 2 elements, e.g., [angle1, angle2].")
            return
        with self.lock:
            start_angles = self.gripper_angles.copy()
            target_angles = list(msg.data)

        self.update_gripper(start_angles, target_angles, is_right=False)


    def gripper_command_callbackR(self, msg):
        if len(msg.data) != 2:
            self.get_logger().error("Gripper command must have 2 elements, e.g., [angle1, angle2].")
            return
        with self.lock:
            start_angles = self.gripper_anglesR.copy()
            target_angles = list(msg.data)

        self.update_gripper(start_angles, target_angles, is_right=True)

    def update_gripper(self, start_angles, target_angles, is_right=False):
        step_size = 0.0125
        num_steps = max(int(max(abs(s - t) for s, t in zip(start_angles, target_angles)) / step_size), 1)
        for i in range(1, num_steps + 1):
            intermediate_angles = [
                start + (target - start) * i / num_steps 
                for start, target in zip(start_angles, target_angles)
            ]
            with self.lock:
                if is_right:
                    self.gripper_anglesR = intermediate_angles
                else:
                    self.gripper_angles = intermediate_angles
            self.publish_joint_states()
            time.sleep(0.025)


def main(args=None):
    rclpy.init(args=args)
    node = IKSolver4DofNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()