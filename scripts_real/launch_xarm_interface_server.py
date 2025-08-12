import sys
import os

# 添加项目根目录到Python路径
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

import zerorpc
from xarm.wrapper import XArmAPI
import scipy.spatial.transform as st
import numpy as np
from umi.common.logger import Logger

class XArmInterface:
    def __init__(self, ip='192.168.1.237'):
        self.robot = XArmAPI(ip, is_radian=True)
        
        # Basic setup
        self.robot.motion_enable(enable=True)
        self.robot.clean_error()
        self.robot.clean_warn()
        self.robot.set_mode(6)
        self.robot.set_state(0)
        self.robot.set_servo_angle(angle=[0,-40,-30,0,30,0], speed=50, wait=False, is_radian=False)
        import time
        time.sleep(3)  # Wait for initial movement to complete
        self.robot.set_gripper_mode(0)
        self.robot.set_gripper_enable(True)
        self.robot.set_gripper_position(850, wait=True)
        self.robot.set_mode(0)  # 设置为位置控制模式
        self.robot.set_state(0)
        
        # Wait for robot to be ready
        import time
        time.sleep(0.5)
        
        # Initialize gripper
        self.robot.set_gripper_mode(0)
        self.robot.set_gripper_enable(True)
        
        # Set important motion parameters
        self.robot.set_tcp_maxacc(1000)  # TCP max acceleration
        self.robot.set_tcp_jerk(5000)   # TCP jerk
        
        print("xArm initialized and ready to move")

    def get_ee_pose(self):
        pose_result = self.robot.get_position_aa(is_radian=True)
        if isinstance(pose_result, (list, tuple)) and len(pose_result) >= 2 and pose_result[0] == 0:
            pose_data = pose_result[1]
            pos = np.array(pose_data[:3])/1000  # Convert mm to m
            rot_vec = pose_data[3:6]
            return np.concatenate([pos, rot_vec]).tolist()
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    
    def get_joint_positions(self):
        angles_result = self.robot.get_servo_angle(is_radian=True)
        if isinstance(angles_result, (list, tuple)) and len(angles_result) >= 2 and angles_result[0] == 0:
            return angles_result[1]
        return [0.0] * 6
    
    def get_joint_velocities(self):
        return [0.0] * 6
    
    def move_to_joint_positions(self, positions, time_to_go):
        # Clear any warnings or errors first
        self.robot.clean_warn()
        self.robot.clean_error()
        
        # Ensure robot is in correct mode and state
        self.robot.set_mode(0)  # Position control mode
        self.robot.set_state(0)  # Ready to move
        
        # Calculate appropriate speed (degrees per second)
        speed = max(10, min(180, int(90 / time_to_go)))  # Increased max speed
        
        # Execute movement with proper error checking
        ret = self.robot.set_servo_angle(angle=positions, speed=speed, wait=True, is_radian=True)
        
        if ret != 0:
            print(f"Joint movement failed with code: {ret}")
            # Try to recover
            self.robot.clean_error()
            self.robot.clean_warn()
            self.robot.set_state(0)
        
        return ret == 0

    def start_cartesian_impedance(self, Kx, Kxd):
        # xArm doesn't have direct impedance control, set motion parameters instead
        self.robot.set_tcp_maxacc(1000)
        self.robot.set_tcp_jerk(5000)
        
        # Ensure proper state
        self.robot.clean_error()
        self.robot.clean_warn()
        self.robot.set_mode(0)
        self.robot.set_state(0)

    def clear_command_queue(self):
        """Clear the command queue to prevent overflow"""
        import time
        time.sleep(0.1)
        self.robot.clean_error()
        self.robot.clean_warn()
        self.robot.set_state(0)
        
    def get_robot_status(self):
        """Get current robot status for debugging"""
        state = self.robot.get_state()
        error = self.robot.get_err_warn_code()
        return {
            'state': state,
            'error': error
        }

    def update_desired_ee_pose(self, pose):
        # Clear any warnings or errors first
        self.robot.clean_warn()
        self.robot.clean_error()
        
        # Ensure robot is in servo motion mode for cartesian servo control
        self.robot.set_mode(1)  # Servo motion mode (required for set_servo_cartesian_aa)
        self.robot.set_state(0)  # Ready to move
        
        pose = np.asarray(pose)

        Logger.debug(f"Updating desired EE pose: {pose}")
        
        # Execute Cartesian movement with error checking
        # 注意这里要使用轴角而不是欧拉角
        axis_angle_pose = [
            pose[0]*1000,  # Convert m to mm
            pose[1]*1000,  # Convert m to mm
            pose[2]*1000,  # Convert m to mm
            pose[3],       # rx (axis angle in radians)
            pose[4],       # ry (axis angle in radians)
            pose[5]        # rz (axis angle in radians)
        ]
        
        ret = self.robot.set_servo_cartesian_aa(
            axis_angle_pose,
            speed=500,
            mvacc=1000,
            is_radian=True,
            is_tool_coord=False,
            relative=False
        )
    
        # import time
        # print(time.time())
        # print(f"Desired pose: {pose}")
        if ret != 0:
            print(f"Cartesian movement failed with code: {ret}")
            # Try to recover
            self.robot.clean_error()
            self.robot.clean_warn()
            self.robot.set_state(0)
        
        return ret == 0

    def get_gripper_position(self):
        gripper_result = self.robot.get_gripper_position()
        if isinstance(gripper_result, (list, tuple)) and len(gripper_result) >= 2 and gripper_result[0] == 0:
            return gripper_result[1]/10000  # Convert mm to m
        return 0.0

    def set_gripper_position(self, position, speed=20000):
        position = float(position) # Convert from meters to mm
        Logger.info(f"Setting gripper position to: {position} mm with speed {speed}")
        position = max(0, min(850, float(position)))
        self.robot.set_gripper_position(position, speed=speed, wait=False)
        return True

    def test_movement(self):
        """Test basic movement functionality"""
        try:
            # Check robot status first
            status = self.get_robot_status()
            print(f"Robot status: {status}")
            
            # Get current position
            current_joints = self.get_joint_positions()
            print(f"Current joints: {current_joints}")
            
            if not current_joints or all(j == 0.0 for j in current_joints):
                print("Warning: Could not get valid joint positions")
                return False
            
            # Small movement test - move first joint by 5 degrees
            test_joints = current_joints.copy()
            test_joints[0] += 0.087  # 5 degrees in radians
            print(f"Moving to: {test_joints}")
            
            success = self.move_to_joint_positions(test_joints, 2.0)
            
            if success:
                print("Movement test successful")
                # Move back to original position
                import time
                time.sleep(1)
                self.move_to_joint_positions(current_joints, 2.0)
            
            return success
            
        except Exception as e:
            print(f"Movement test failed: {e}")
            # Try to recover
            self.clear_command_queue()
            return False

    def terminate_current_policy(self):
        """Emergency stop and clear all commands"""
        self.robot.emergency_stop()
        import time
        time.sleep(0.5)        
        # Reset to safe state
        self.robot.clean_error()
        self.robot.clean_warn() 
        self.robot.set_state(4)  # Stop state
        
        print("Policy terminated and robot stopped")

s = zerorpc.Server(XArmInterface())
s.bind("tcp://0.0.0.0:4242")
s.run()