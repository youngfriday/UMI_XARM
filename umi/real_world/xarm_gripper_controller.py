import os
import time
import enum
import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
from umi.shared_memory.shared_memory_queue import (
    SharedMemoryQueue, Empty)
from umi.shared_memory.shared_memory_ring_buffer import SharedMemoryRingBuffer
from umi.common.precise_sleep import precise_wait
from umi.common.pose_trajectory_interpolator import PoseTrajectoryInterpolator
import zerorpc


class Command(enum.Enum):
    SHUTDOWN = 0
    SCHEDULE_WAYPOINT = 1
    RESTART_PUT = 2

class XArmGripperController(mp.Process):
    def __init__(self,
            shm_manager: SharedMemoryManager,
            hostname="127.0.0.1",
            port=4242,
            frequency=30,
            move_max_speed=5000,
            get_max_k=None,
            command_queue_size=1024,
            launch_timeout=3,
            receive_latency=0.01,
            use_meters=True,
            verbose=False
            ):
        super().__init__(name="XArmGripperController")
        self.hostname = hostname
        self.port = port
        self.frequency = frequency
        self.move_max_speed = move_max_speed
        self.launch_timeout = launch_timeout
        self.receive_latency = receive_latency
        self.scale = 10000.0 if use_meters else 1.0  
        self.verbose = verbose

        if get_max_k is None:
            get_max_k = int(frequency * 10)
        
        # build input queue
        example = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': 0.0,
            'target_time': 0.0
        }
        input_queue = SharedMemoryQueue.create_from_examples(
            shm_manager=shm_manager,
            examples=example,
            buffer_size=command_queue_size
        )
        
        # build ring buffer
        example = {
            'gripper_position': 0.0,
            'gripper_velocity': 0.0,
            'gripper_force': 0.0,
            'gripper_receive_timestamp': time.time(),
            'gripper_timestamp': time.time()
        }
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

    # ========= launch method ===========
    def start(self, wait=True):
        super().start()
        if wait:
            self.start_wait()
        if self.verbose:
            print(f"[XArmGripperController] Controller process spawned at {self.pid}")

    def stop(self, wait=True):
        message = {
            'cmd': Command.SHUTDOWN.value
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
    
    # ========= context manager ===========
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        
    # ========= command methods ============
    def schedule_waypoint(self, pos: float, target_time: float):
        message = {
            'cmd': Command.SCHEDULE_WAYPOINT.value,
            'target_pos': pos,
            'target_time': target_time
        }
        self.input_queue.put(message)


    def restart_put(self, start_time):
        self.input_queue.put({
            'cmd': Command.RESTART_PUT.value,
            'target_time': start_time
        })
    
    # ========= receive APIs =============
    def get_state(self, k=None, out=None):
        if k is None:
            return self.ring_buffer.get(out=out)
        else:
            return self.ring_buffer.get_last_k(k=k,out=out)
    
    def get_all_state(self):
        return self.ring_buffer.get_all()
    
    # ========= main loop in process ============
    def run(self):
        print(f"[XArmGripperController] run() 方法开始执行，进程ID: {os.getpid()}")
        # start connection
        try:
            client = zerorpc.Client(heartbeat=60, timeout=60)
            client.connect(f"tcp://{self.hostname}:{self.port}")
            
            try:
                # get initial
                result = client.get_gripper_position()
                if isinstance(result, (list, tuple)) and len(result) >= 2:
                    code, curr_pos = result[0], result[1]
                    if code != 0:
                        curr_pos = 0.0
                    else:
                        curr_pos = float(curr_pos)
                else:
                    curr_pos = float(result) if result is not None else 0.0
                
                # curr_pos = 100.0
                curr_t = time.monotonic()
                last_waypoint_time = curr_t
                pose_interp = PoseTrajectoryInterpolator(
                    times=[curr_t],
                    poses=[[curr_pos,0,0,0,0,0]]
                )
                
                keep_running = True
                t_start = time.monotonic()
                iter_idx = 0
                while keep_running:
                    # command gripper
                    t_now = time.monotonic()
                    dt = 1 / self.frequency
                    t_target = t_now
                    target_pos = pose_interp(t_target)[0]
                    
                    # set gripper position
                    gripper_pos = target_pos
                    if self.scale != 1.0:
                        gripper_pos = gripper_pos * self.scale
                    try:
                        client.set_gripper_position(float(gripper_pos), speed=self.move_max_speed)
                    except Exception as e:
                        if self.verbose:
                            print(f"[XArmGripperController] Set position failed: {e}")

                    # get state from robot
                    try:
                        result = client.get_gripper_position()
                        if isinstance(result, (list, tuple)) and len(result) >= 2:
                            code, actual_pos = result[0], result[1]
                            if code == 0:
                                actual_pos = float(actual_pos)
                            else:
                                actual_pos = curr_pos  # use last known position
                        else:
                            actual_pos = float(result) if result is not None else curr_pos
                    except Exception as e:
                        if self.verbose:
                            print(f"[XArmGripperController] Get position failed: {e}")
                        actual_pos = curr_pos
                    
                    # if self.scale != 1.0:
                    #     actual_pos = actual_pos / self.scale
                    
                    state = {
                        'gripper_position': actual_pos,
                        'gripper_velocity': 0.0,
                        'gripper_force': 0.0,
                        'gripper_receive_timestamp': time.time(),
                        'gripper_timestamp': time.time() - self.receive_latency
                    }
                    self.ring_buffer.put(state)

                    # fetch command from queue
                    try:
                        commands = self.input_queue.get_all()
                        n_cmd = len(commands['cmd'])
                    except Empty:
                        n_cmd = 0
                    
                    # execute commands
                    for i in range(n_cmd):
                        command = dict()
                        for key, value in commands.items():
                            command[key] = value[i]
                        cmd = command['cmd']
                        
                        if cmd == Command.SHUTDOWN.value:
                            keep_running = False
                            # stop immediately, ignore later commands
                            break
                        elif cmd == Command.SCHEDULE_WAYPOINT.value:
                            target_pos = command['target_pos']
                            target_time = command['target_time']
                            # translate global time to monotonic time
                            target_time = time.monotonic() - time.time() + target_time
                            curr_time = t_now
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=[target_pos, 0, 0, 0, 0, 0],
                                time=target_time,
                                max_pos_speed=self.move_max_speed,
                                max_rot_speed=self.move_max_speed,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time
                            )
                            last_waypoint_time = target_time
                        elif cmd == Command.RESTART_PUT.value:
                            t_start = command['target_time'] - time.time() + time.monotonic()
                            iter_idx = 1
                        else:
                            keep_running = False
                            break
                        
                    # first loop successful, ready to receive command
                    if iter_idx == 0:
                        self.ready_event.set()
                    iter_idx += 1
                    
                    # regulate frequency
                    dt = 1 / self.frequency
                    t_end = t_start + dt * iter_idx
                    precise_wait(t_end=t_end, time_func=time.monotonic)
                    
            finally:
                try:
                    client.close()
                except:
                    pass
                
        finally:
            self.ready_event.set()
            if self.verbose:
                print(f"[XArmGripperController] Disconnected from robot: {self.hostname}") 