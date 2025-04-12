import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
import math
import time
import os

class PIDController:
    def __init__(self, kp, ki, kd, max_output, min_output):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.min_output = min_output
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, error, dt):
        if dt <= 0.0:
            return 0.0
        
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = max(min(output, self.max_output), self.min_output)
        self.prev_error = error
        return output

class BlueBoatPIDController(Node):
    def __init__(self):
        super().__init__('blueboat_pid_controller')

        self.blueboat_odom_sub = self.create_subscription(
            Odometry, '/blueboat/navigator/odometry', self.odom_callback, 10
        )
        self.cyl_odom_sub = self.create_subscription(
            Odometry, '/cyl/odom', self.cyl_odom_callback, 10
        )
        self.thruster_pub = self.create_publisher(
            Float64MultiArray, '/blueboat/controller/thruster_setpoints_sim', 10
        )

        self.blueboat_pose = None
        self.cyl_pose = None
        #self.state = "align_y"
        self.state = "move_x"

        # PID Controller
        self.angular_pid = PIDController(kp=1.0, ki=0.0, kd=0.5, max_output=0.2, min_output=-0.2)
        self.lateral_pid = PIDController(kp=0.8, ki=0.0, kd=0.2, max_output=0.2, min_output=-0.2)


        # PID control parameters
        self.linear_kP = 0.2
        self.linear_kI = 0.05
        self.linear_kD = 0.01
        self.angular_kP = 0.3
        self.angular_kI = 0.15
        self.angular_kD = 0.05

        # Integral term
        self.linear_integral = 0.0
        self.angular_integral = 0.0
        self.lateral_integral = 0.0  # initialization

        self.previous_linear_error = 0.0
        self.previous_angular_error = 0.0
        self.previous_lateral_error = 0.0  # initialization

        self.previous_time = time.time()
        self.timer = self.create_timer(0.1, self.update_control)

    def odom_callback(self, msg):
        self.blueboat_pose = msg.pose.pose

    def cyl_odom_callback(self, msg):
        self.cyl_pose = msg.pose.pose

    def update_control(self):
        if self.blueboat_pose is None or self.cyl_pose is None:
            return

        # Obtain time interval dt
        current_time = time.time()
        dt = current_time - self.previous_time
        self.previous_time = current_time
        if dt <= 0.0:
            dt = 0.001

        # Obtain cyl position and orientation
        cyl_yaw = self.get_yaw(self.cyl_pose.orientation)
        cyl_x = self.cyl_pose.position.x
        cyl_y = self.cyl_pose.position.y

        # Obtain the position and orientation of the bluecoat
        blue_x = self.blueboat_pose.position.x
        blue_y = self.blueboat_pose.position.y
        blue_yaw = self.get_yaw(self.blueboat_pose.orientation)

        # Calculate y-axis deviation error y
        dx = blue_x - cyl_x
        dy = blue_y - cyl_y
        error_y = -dx * math.sin(cyl_yaw) + dy * math.cos(cyl_yaw)

        # Calculate the expected heading so that the blueboat faces the y-axis of cyl
        desired_heading = self.normalize_angle(cyl_yaw - math.pi / 2)
        heading_error = self.normalize_angle(desired_heading - blue_yaw)

        self.get_logger().info(
            f"State: {self.state}"
        )

        # Align Y-axis
        if self.state == "align_y":
            # Adjust heading
            angular_correction = self.angular_pid.compute(heading_error, dt)

            # If the heading error is large, rotate first
            if abs(heading_error) > math.radians(20):  # Error greater than 20 °, continue rotating
                angular_output = self.angular_pid.compute(heading_error, dt)
                
                if heading_error > 0:
                    left_thrust = -angular_output
                    right_thrust = angular_output
                else:
                    left_thrust = angular_output
                    right_thrust = -angular_output

                # Gradually apply reverse thrust when the error approaches the target
                if abs(heading_error) > math.radians(165):
                    reverse_correction = 0.1 * math.exp(-3 * abs(heading_error) / math.radians(30))  # exponential decay
                    if heading_error > 0:
                        left_thrust -= reverse_correction
                        right_thrust += reverse_correction
                    else:
                        left_thrust += reverse_correction
                        right_thrust -= reverse_correction

                # Stop adjusting when the error is extremely small
                if abs(heading_error) > math.radians(175):
                    self.send_thruster_command(0.0, 0.0,0.0, 0.0)
                    self.get_logger().info("Heading adjustment completed, proceed to the next step")
                    self.create_timer(1.0, self.switch_to_move_y)
                    return

                self.send_thruster_command(left_thrust, right_thrust,0.0, 0.0)
                self.get_logger().info(f"Rotate and adjust heading: Thrusters: L={left_thrust:.2f}, R={right_thrust:.2f}")
            


        elif self.state == "move_y":
            # Move forward to align the Y-axis
            lateral_correction = self.lateral_pid.compute(error_y, dt)

            # Limit maximum movement speed
            max_lat_correction = 0.2  
            lateral_correction = max(min(lateral_correction, max_lat_correction), -max_lat_correction)

            right_thrust = 0.2
            left_thrust = 0.08

            self.send_thruster_command(right_thrust, left_thrust, 0.0, 0.0)
            self.get_logger().info(
                f" Advance along the cyl with y axis: error_y = {error_y:.2f}, Thrusters: L={left_thrust:.2f}, R={right_thrust:.2f}"
            )

            # If the Y-axis error is small enough, stop and proceed to the next stage
            if error_y > 0.1 and error_y < 0.12:
                self.get_logger().info(" Y-axis alignment completed!")
                self.send_thruster_command(0.0, 0.0, 0.0, 0.0)
                self.create_timer(1.0, self.switch_to_align_x)
        

        elif self.state == "align_x":
            # Calculate x-axis deviation error_x
            d = 1.0  # Expected offset distance (BlueBoat should be d meters behind cyl)
            target_x = cyl_x - d * math.cos(cyl_yaw)  # Calculate the x-coordinate of the target
            error_x = blue_x - target_x  # Calculate the current x-axis error

            # Calculate the expected heading so that BlueBoat faces the x-axis of cyl
            desired_heading = self.normalize_angle(cyl_yaw)  # Target orientation
            heading_error_x = self.normalize_angle(desired_heading - blue_yaw)

            self.get_logger().info(
                f"State: {self.state}, heading_err_x = {math.degrees(heading_error_x):.2f}°, error_x = {error_x:.2f}"
            )

            # Adjust heading
            angular_correction = self.angular_pid.compute(heading_error_x, dt)

            # If the heading error is large, rotate first
            if abs(heading_error_x) > math.radians(20):  # Error greater than 20 °, continue rotating
                angular_output = self.angular_pid.compute(heading_error_x, dt)
                
                if heading_error_x > 0:
                    left_thrust = angular_output
                    right_thrust = -angular_output
                else:
                    left_thrust = -angular_output
                    right_thrust = angular_output

                # Gradually apply reverse thrust when the error approaches the target
                if abs(heading_error_x) < math.radians(15) and abs(heading_error_x) > math.radians(5):
                    reverse_correction = 0.1 * math.exp(-3 * abs(heading_error_x) / math.radians(40))  # exponential decay
                    if heading_error_x > 0:
                        left_thrust += reverse_correction
                        right_thrust -= reverse_correction
                    else:
                        left_thrust -= reverse_correction
                        right_thrust += reverse_correction

                self.send_thruster_command(left_thrust, right_thrust, 0.0, 0.0)
                self.get_logger().info(f"Rotate x to adjust heading: Thrusters: L={left_thrust:.2f}, R={right_thrust:.2f}")

            # If the error is less than 5 °, stop
            if abs(heading_error_x) < math.radians(5):  # If the error is less than 5 °, it is considered aligned
                self.send_thruster_command(0.0, 0.0, 0.0, 0.0)  # Send stop command
                self.get_logger().info("X heading adjustment completed, all movements stopped!")
                #self.state = "stop"  # Terminate status to prevent further execution
                self.create_timer(1.0, self.switch_to_move_x)
                return


        elif self.state == "move_x":
            # Calculate the target position
            d = 0.816  # The expected distance between the target and Cyl
            target_x = self.cyl_pose.position.x - d * math.cos(self.get_yaw(self.cyl_pose.orientation))
            target_y = self.cyl_pose.position.y  # The target position on the y-axis remains unchanged
            # calculate error
            error_x = blue_x - target_x
            error_y = -dx * math.sin(cyl_yaw) + dy * math.cos(cyl_yaw)  # Y-axis error
            heading_error_x = self.normalize_angle(cyl_yaw - blue_yaw)  # heading error

            self.get_logger().info(
                f"State: {self.state}, heading_err_x = {math.degrees(heading_error_x):.2f}°, "
                f"error_x = {error_x:.2f}, error_y = {error_y:.2f}"
            )

            # Obtain time interval dt
            current_time = time.time()
            dt = current_time - self.previous_time
            if dt <= 0.0:
                dt = 0.001
            self.previous_time = current_time

            # Forward to correct X-axis error
            self.linear_integral += error_x * dt
            derivative_linear = (error_x - self.previous_linear_error) / dt
            linear_output = (self.linear_kP * error_x +
                            self.linear_kI * self.linear_integral +
                            self.linear_kD * derivative_linear)
            self.previous_linear_error = error_x

            # Set the minimum speed to ensure that the small boat can continue to move forward
            min_linear_speed = 0.08
            if abs(linear_output) < min_linear_speed and abs(error_x) > 0.05:
                linear_output = min_linear_speed * (1 if error_x > 0 else -1)

            # Slow down in advance and apply the brakes in reverse
            # When the error x is small, reduce the forward output in advance and apply a small reverse component
            if 0.05 < error_x < 0.15:
                # Calculate a backpropagation component, where as the error x decreases, the backpropagation component increases
                braking = 0.5 * (0.15 - error_x)  
                linear_output = linear_output - braking
                self.get_logger().info(f"Early deceleration and reverse thrust: braking = {braking:.2f}")

            # Heading correction to maintain angle stability
            self.angular_integral += heading_error_x * dt
            derivative_angular = (heading_error_x - self.previous_angular_error) / dt
            angular_output = (self.angular_kP * heading_error_x +
                            self.angular_kI * self.angular_integral +
                            self.angular_kD * derivative_angular)
            self.previous_angular_error = heading_error_x

            # Set heading dead zone
            if abs(heading_error_x) < math.radians(2):
                angular_output = 0.0

            # Limit the maximum range of heading adjustment
            max_angular = 0.3
            angular_output = max(min(angular_output, max_angular), -max_angular)

            # Horizontal correction to maintain Y-axis alignment
            self.lateral_integral += error_y * dt
            derivative_lateral = (error_y - self.previous_lateral_error) / dt
            lateral_correction = (self.linear_kP * error_y +
                                self.linear_kI * self.lateral_integral +
                                self.linear_kD * derivative_lateral)
            self.previous_lateral_error = error_y

            # The third propeller is used to apply lateral thrust control
            if abs(error_y) > 0.02:  # Only start the side push when the error is greater than 2cm
                side_thruster_output = 0.2 if error_y > 0 else -0.2  # Only positive and negative, constant thrust
            else:
                side_thruster_output = 0.0  # The error is small enough to stop pushing sideways


            # If the error is small, stop pushing sideways
            if abs(error_y) < 0.03:
                lateral_correction = 0.0
                side_thruster_output = 0.0

            # Propulsion output
            left_thrust = -linear_output - angular_output - lateral_correction
            right_thrust = -linear_output + angular_output + lateral_correction

            # Send thrust command, including the third propeller
            #self.send_thruster_command(left_thrust, right_thrust, side_thruster_output)
            self.send_thruster_command(left_thrust, right_thrust, side_thruster_output,0.0)
            self.get_logger().info(
                f"Go forward along the 'cyl' 'x' axis: error_x = {error_x:.2f}, error_y = {error_y:.2f}, "
                f"Thrusters: L={left_thrust:.2f}, R={right_thrust:.2f}, Side={side_thruster_output:.2f}"
            )

            # If the X-axis, Y-axis, and heading are aligned, stop
            if abs(error_x) < 0.05 and abs(error_y) < 0.02 and abs(heading_error_x) < math.radians(2):
                self.get_logger().info("X-axis alignment complete! All motion stopped")
                self.send_thruster_command(0.0, 0.0, 0.0, 0.0)
                self.state = "stop"
                self.get_logger().info(" Mission accomplished, shut down the ROS2 node!")
                self.destroy_node()  # Stop the node
                rclpy.shutdown()  # Close ROS2
                self.get_logger().info("Exit the process completely!")
                os._exit(0)  # Terminate the Python process and close the ROS2 process in the terminal
                subprocess.run(["pkill", "-f", "python3"])  # Forcefully killing Python processes
                #return





    def send_thruster_command(self, left_thrust, right_thrust, side_thrust=0.0, left_left=0.0):
        msg = Float64MultiArray()
        msg.data = [left_thrust, right_thrust, side_thrust,left_left]  # 3 propellers
        self.thruster_pub.publish(msg)
        self.get_logger().info(f"Thrust command:[L={left_thrust:.2f}, R={right_thrust:.2f}, RR={side_thrust:.2f}, LL={side_thrust:.2f}]")

    
    def switch_to_move_y(self):
        """ Switch to the 'move_y' state after 1 second """
        self.state = "move_y"
        #self.get_logger().info("State switch after 1 second: Enter the 'move_y' stage!")
    
    def switch_to_align_x(self):
        """ Switch to the 'align_x' state after 1 second """
        self.state = "align_x"
        #self.get_logger().info("State switch after 1 second: Enter the 'align_x' stage!")
    
    def switch_to_move_x(self):
        """ Switch to the 'move_x' state after 1 second """
        self.state = "move_x"
        #self.get_logger().info("State switch after 1 second: Enter the 'move_x' stage!")


    def get_yaw(self, q):
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                          1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def normalize_angle(self, angle):
        return (angle + math.pi) % (2 * math.pi) - math.pi

def main(args=None):
    rclpy.init(args=args)
    node = BlueBoatPIDController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
