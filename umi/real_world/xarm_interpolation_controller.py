import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import scipy.interpolate as si
import scipy.spatial.transform as st
import numpy as np

from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from diffusion_policy.common.precise_sleep import precise_wait
from umi.common.pose_util import pose_to_mat, mat_to_pose
import zerorpc
from umi.common.logger import Logger
class Command(enum.Enum):
    STOP = 0
    SERVOL = 1
    SCHEDULE_WAYPOINT = 2

tx_flangerot90_tip = np.identity(4)
tx_flangerot90_tip[:3, 3] = np.array([0, 0, 0]) # xArm 改装夹爪变长

# tx_flangerot45_flangerot90 = np.identity(4)
# xarm 不需要转
# tx_flangerot45_flangerot90[:3,:3] = st.Rotation.from_euler('x', [np.pi/2]).as_matrix()

tx_flange_flangerot45 = np.identity(4)
# 由于xarm末端执行器坐标系与gopro以及ur机械臂定义不同，需要绕z轴旋转90度，才能使用官方的数据和模型
tx_flange_flangerot45[:3,:3] = st.Rotation.from_euler('z', [np.pi/2]).as_matrix()

tx_flange_tip = tx_flange_flangerot45 @ tx_flangerot90_tip
# Logger.debug(f"Transform from flange to tip: {tx_flange_tip}")
tx_tip_flange = np.linalg.inv(tx_flange_tip)
# Logger.debug(f"Transform from tip to flange: {tx_tip_flange}")

class XArmInterface:
    def __init__(self, ip='127.0.0.1', port=4242):
        self.server = zerorpc.Client(heartbeat=60)
        self.server.connect(f"tcp://{ip}:{port}")

    def get_ee_pose(self):
        """获取末端执行器位姿，返回 [x, y, z, rx, ry, rz] 格式"""
        try:
            flange_pose = np.array(self.server.get_ee_pose())
            tip_pose = mat_to_pose(pose_to_mat(flange_pose) @ tx_flange_tip)
            # Logger.debug(f"Transform from flange to tip: {flange_pose} -> {tip_pose}")
            return tip_pose
        except Exception as e:
            print(f"Failed to get ee pose: {e}")
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    
    def get_joint_positions(self):
        """获取关节位置"""
        try:
            # 只取前6维度 
            # TODO : 如果是xArm7则需要修改为7
            return np.array(self.server.get_joint_positions())[:6]
        except Exception as e:
            print(f"Failed to get joint positions: {e}")
            return np.array([0.0] * 6)
    
    def get_joint_velocities(self):
        """获取关节速度（xArm SDK 不直接支持，返回零向量）"""
        try:
            return np.array(self.server.get_joint_velocities())
        except Exception as e:
            print(f"Failed to get joint velocities: {e}")
            return np.array([0.0] * 6)

    def move_to_joint_positions(self, positions: np.ndarray, time_to_go: float):
        """移动到指定关节位置"""
        try:
            self.server.move_to_joint_positions(positions.tolist(), time_to_go)
        except Exception as e:
            print(f"Failed to move to joint positions: {e}")

    def start_cartesian_impedance(self, Kx: np.ndarray, Kxd: np.ndarray):
        """启动笛卡尔阻抗控制（xArm 可能不直接支持，使用位置控制替代）"""
        try:
            self.server.start_cartesian_impedance(Kx.tolist(), Kxd.tolist())
        except Exception as e:
            print(f"Failed to start cartesian impedance: {e}")
    
    def update_desired_ee_pose(self, pose: np.ndarray):
        """更新期望的末端执行器位姿"""
        try:
            self.server.update_desired_ee_pose(pose.tolist())
        except Exception as e:
            print(f"Failed to update desired ee pose: {e}")

    def terminate_current_policy(self):
        """终止当前策略"""
        try:
            self.server.terminate_current_policy()
        except Exception as e:
            print(f"Failed to terminate current policy: {e}")

    def close(self):
        """关闭连接"""
        try:
            self.server.close()
        except Exception as e:
            print(f"Failed to close connection: {e}")


class XArmInterpolationController(mp.Process):
    """
    xArm 插值控制器
    为了确保以可预测的延迟向机器人发送命令，
    该控制器需要其独立的进程（由于 Python GIL）
    """
    def __init__(self,
        shm_manager: SharedMemoryManager, 
        robot_ip,
        robot_port=4242,
        frequency=125,  # xArm 建议频率
        Kx_scale=1.0,
        Kxd_scale=1.0,
        launch_timeout=3,
        joints_init=None,
        joints_init_duration=None,
        soft_real_time=False,
        verbose=False,
        get_max_k=None,
        receive_latency=0.0
        ):
        """
        robot_ip: xArm 控制器的 IP 地址
        frequency: 控制频率，xArm 建议 125Hz
        soft_real_time: 启用实时调度
        """

        if joints_init is not None:
            joints_init = np.array(joints_init)
            assert joints_init.shape == (6,) 

        super().__init__(name="XArmInterpolationController")
        self.robot_ip = robot_ip
        self.robot_port = robot_port
        self.frequency = frequency
        self.Kx = np.array([100.0, 100.0, 100.0, 10.0, 10.0, 10.0]) * Kx_scale
        self.Kxd = np.array([20.0, 20.0, 20.0, 2.0, 2.0, 2.0]) * Kxd_scale
        self.launch_timeout = launch_timeout
        self.joints_init = joints_init
        self.joints_init_duration = joints_init_duration
        self.soft_real_time = soft_real_time
        self.receive_latency = receive_latency
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 5)

        # 构建输入队列
        example = {
            'cmd': Command.SERVOL.value,
            'target_pose': np.zeros((6,), dtype=np.float64),
            'duration': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=256
        )

        # 构建环形缓冲区
        receive_keys = [
            ('ActualTCPPose', 'get_ee_pose'),
            ('ActualQ', 'get_joint_positions'),
            ('ActualQd','get_joint_velocities'),
        ]
        example = dict()
        for key, func_name in receive_keys:
            if 'joint' in func_name:
                example[key] = np.zeros(6)  # TODO xArm6 是6轴 如果是xarm7则是7轴
            elif 'ee_pose' in func_name:
                example[key] = np.zeros(6)

        example['robot_receive_timestamp'] = time.time()
        example['robot_timestamp'] = time.time()
        ring_buffer = SharedMemoryRingBuffer.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            get_max_k=get_max_k,
            get_time_budget=0.2,
            put_desired_frequency=frequency
        )

        self.ready_event = mp.Event()
        self.input_queue = input_queue
        self.ring_buffer = ring_buffer
        self.receive_keys = receive_keys
            
    # ========= 启动方法 ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[XArmInterpolationController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.STOP.value
        }
        self.input_queue.put(message)
        if wait:
            self.stop_wait()
    
    def start_wait(self):
        self.ready_event.wait(self.launch_timeout)
        assert self.is_alive()
    
    def stop_wait(self):
        self.join()
    
    @property
    def is_ready(self):
        return self.ready_event.is_set()
    
    # ========= 上下文管理器 ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    # ========= 命令方法 ============
    def servoL(self, pose, duration=0.1):
        """
        duration: 到达位姿的期望时间
        """
        assert self.is_alive()
        assert(duration >= (1/self.frequency))
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SERVOL.value,
            'target_pose': pose,
            'duration': duration
        }
        self.input_queue.put(message)
    
    def schedule_waypoint(self, pose, target_time):
        pose = np.array(pose)
        assert pose.shape == (6,)

        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pose': pose,
            'target_time': target_time
        }
        self.input_queue.put(message)

    def clear_queue(self):
        self.input_queue.clear()
    
    # ========= 接收 APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    

    # ========= 进程中的主循环 ============
    def run(self):
        print(f"[XArmInterpolationController] Starting run() method...")
        
        # 启用软实时
        if self.soft_real_time:
            os.sched_setscheduler(
                0, os.SCHED_RR, os.sched_param(20))
            
        # 启动 xArm 接口
        print(f"[XArmInterpolationController] Creating XArmInterface with IP: {self.robot_ip}")
        try:
            robot = XArmInterface(self.robot_ip, self.robot_port)
            print(f"[XArmInterpolationController] XArmInterface created successfully")
        except Exception as e:
            print(f"[XArmInterpolationController] Failed to create XArmInterface: {e}")
            return

        try:
            if self.verbose:
                print(f"[XArmInterpolationController] Connect to robot: {self.robot_ip}")
            
            # 初始化位置
            if self.joints_init is not None:
                print(f"[XArmInterpolationController] Initializing joints: {self.joints_init}")
                duration = self.joints_init_duration if self.joints_init_duration is not None else 3.0
                robot.move_to_joint_positions(
                    positions=np.asarray(self.joints_init),
                    time_to_go=duration
                )
                print(f"[XArmInterpolationController] Joints initialization completed")
 
            # 主循环
            print(f"[XArmInterpolationController] Starting main loop preparation...")
            dt = 1. / self.frequency
            
            print(f"[XArmInterpolationController] Getting current pose...")
            curr_pose = robot.get_ee_pose()
            print(f"[XArmInterpolationController] Current pose: {curr_pose}")

            # 使用单调时间确保控制循环永不倒退
            curr_t = time.monotonic()
            last_waypoint_time = curr_t
            
            print(f"[XArmInterpolationController] Creating pose interpolator...")
            pose_interp = PoseTrajectoryInterpolator(
                times=np.array([curr_t]),
                poses=np.array([curr_pose])
            )
            print(f"[XArmInterpolationController] Pose interpolator created")

            # 启动 xArm 控制模式
            print(f"[XArmInterpolationController] Starting cartesian impedance control...")
            robot.start_cartesian_impedance(
                Kx=self.Kx,
                Kxd=self.Kxd
            )
            print(f"[XArmInterpolationController] Cartesian impedance control started")

            print(f"[XArmInterpolationController] Entering main control loop...")
            t_start = time.monotonic()
            iter_idx = 0
            keep_running = True
            while keep_running:
                # 向机器人发送命令
                t_now = time.monotonic()
                tip_pose = pose_interp(t_now)
                flange_pose = mat_to_pose(pose_to_mat(tip_pose) @ tx_tip_flange)
                # Logger.debug(f"Sending command to robot: {flange_pose}")
                # debug
                # import pdb; pdb.set_trace()
                # 发送命令到机器人
                robot.update_desired_ee_pose(flange_pose)

                # 更新机器人状态
                state = dict()
                for key, func_name in self.receive_keys:
                    state[key] = getattr(robot, func_name)()

                t_recv = time.time()
                state['robot_receive_timestamp'] = t_recv
                state['robot_timestamp'] = t_recv - self.receive_latency
                self.ring_buffer.put(state)

                # 从队列获取命令
                try:
                    commands = self.input_queue.get_k(1)
                    n_cmd = len(commands['cmd'])
                except Empty:
                    n_cmd = 0

                # 执行命令
                for i in range(n_cmd):
                    command = dict()
                    for key, value in commands.items():
                        command[key] = value[i]
                    cmd = command['cmd']

                    if cmd == Command.STOP.value:
                        keep_running = False
                        break
                    elif cmd == Command.SERVOL.value:
                        target_pose = command['target_pose']
                        duration = float(command['duration'])
                        curr_time = t_now + dt
                        t_insert = curr_time + duration
                        pose_interp = pose_interp.drive_to_waypoint(
                            pose=target_pose,
                            time=t_insert,
                            curr_time=curr_time,
                        )
                        last_waypoint_time = t_insert
                        if self.verbose:
                            print("[XArmInterpolationController] New pose target:{} duration:{}s".format(
                                target_pose, duration))
                    elif cmd == Command.SCHEDULE_WAYPOINT.value:
                        target_pose = command['target_pose']
                        target_time = float(command['target_time'])
                        # 将全局时间转换为单调时间
                        target_time = time.monotonic() - time.time() + target_time
                        curr_time = t_now + dt
                        pose_interp = pose_interp.schedule_waypoint(
                            pose=target_pose,
                            time=target_time,
                            curr_time=curr_time,
                            last_waypoint_time=last_waypoint_time
                        )
                        last_waypoint_time = target_time
                    else:
                        keep_running = False
                        break

                # 调节频率
                t_wait_util = t_start + (iter_idx + 1) * dt
                precise_wait(t_wait_util, time_func=time.monotonic)

                # 第一个循环成功，准备接收命令
                if iter_idx == 0:
                    print(f"[XArmInterpolationController] First iteration completed, setting ready event")
                    self.ready_event.set()
                iter_idx += 1

                if self.verbose:
                    print(f"[XArmInterpolationController] Actual frequency {1/(time.monotonic() - t_now)}")

        except Exception as e:
            print(f"[XArmInterpolationController] Error in run() method: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # 强制清理
            print('\n\n\n\nterminate_current_policy\n\n\n\n\n')
            robot.terminate_current_policy()
            robot.close()
            self.ready_event.set()

            if self.verbose:
                print(f"[XArmInterpolationController] Disconnected from robot: {self.robot_ip}") 