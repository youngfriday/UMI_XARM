"""
Usage:
(umi): python scripts_real/eval_real_umi.py -i data/outputs/2023.10.26/02.25.30_train_diffusion_unet_timm_umi/checkpoints/latest.ckpt -o data_local/cup_test_data

================ Human in control ==============
Robot movement:
Move your SpaceMouse to move the robot EEF (locked in xy plane).
Press SpaceMouse right button to unlock z axis.
Press SpaceMouse left button to enable rotation axes.

Recording control:
Click the opencv window (make sure it's in focus).
Press "C" to start evaluation (hand control over to policy).
Press "Q" to exit program.

================ Policy in control ==============
Make sure you can hit the robot hardware emergency-stop button quickly! 

Recording control:
Press "S" to stop evaluation and gain control back.
"""

# %%
import os
import pathlib
import time
from multiprocessing.managers import SharedMemoryManager
import omegaconf

import av
import click
import cv2
import yaml
import dill
import hydra
import numpy as np
import scipy.spatial.transform as st
import torch
from omegaconf import OmegaConf
import json
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.cv2_util import (
    get_image_transform
)
from umi.common.cv_util import (
    parse_fisheye_intrinsics,
    FisheyeRectConverter
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.common.precise_sleep import precise_wait
from umi.real_world.bimanual_umi_env import BimanualUmiEnv
from umi.real_world.keystroke_counter import (
    KeystrokeCounter, Key, KeyCode
)
from umi.real_world.real_inference_util import (get_real_obs_dict,
                                                get_real_obs_resolution,
                                                get_real_umi_obs_dict,
                                                get_real_umi_action)
from umi.real_world.spacemouse_shared_memory import Spacemouse
from umi.common.pose_util import pose_to_mat, mat_to_pose
from scipy.spatial.transform import Rotation as R

OmegaConf.register_new_resolver("eval", eval, replace=True)

def solve_table_collision(ee_pose, gripper_width, height_threshold):
    finger_thickness = 25.5 / 1000
    keypoints = list()
    for dx in [-1, 1]:
        for dy in [-1, 1]:
            keypoints.append((dx * gripper_width / 2, dy * finger_thickness / 2, 0))
    keypoints = np.asarray(keypoints)
    rot_mat = st.Rotation.from_rotvec(ee_pose[3:6]).as_matrix()
    transformed_keypoints = np.transpose(rot_mat @ np.transpose(keypoints)) + ee_pose[:3]
    delta = max(height_threshold - np.min(transformed_keypoints[:, 2]), 0)
    ee_pose[2] += delta

def solve_sphere_collision(ee_poses, robots_config):
    num_robot = len(robots_config)
    this_that_mat = np.identity(4)
    this_that_mat[:3, 3] = np.array([0, 0.89, 0]) # TODO: very hacky now!!!!

    for this_robot_idx in range(num_robot):
        for that_robot_idx in range(this_robot_idx + 1, num_robot):
            this_ee_mat = pose_to_mat(ee_poses[this_robot_idx][:6])
            this_sphere_mat_local = np.identity(4)
            this_sphere_mat_local[:3, 3] = np.asarray(robots_config[this_robot_idx]['sphere_center'])
            this_sphere_mat_global = this_ee_mat @ this_sphere_mat_local
            this_sphere_center = this_sphere_mat_global[:3, 3]

            that_ee_mat = pose_to_mat(ee_poses[that_robot_idx][:6])
            that_sphere_mat_local = np.identity(4)
            that_sphere_mat_local[:3, 3] = np.asarray(robots_config[that_robot_idx]['sphere_center'])
            that_sphere_mat_global = this_that_mat @ that_ee_mat @ that_sphere_mat_local
            that_sphere_center = that_sphere_mat_global[:3, 3]

            distance = np.linalg.norm(that_sphere_center - this_sphere_center)
            threshold = robots_config[this_robot_idx]['sphere_radius'] + robots_config[that_robot_idx]['sphere_radius']
            # print(that_sphere_center, this_sphere_center)
            if distance < threshold:
                print('avoid collision between two arms')
                half_delta = (threshold - distance) / 2
                normal = (that_sphere_center - this_sphere_center) / distance
                this_sphere_mat_global[:3, 3] -= half_delta * normal
                that_sphere_mat_global[:3, 3] += half_delta * normal
                
                ee_poses[this_robot_idx][:6] = mat_to_pose(this_sphere_mat_global @ np.linalg.inv(this_sphere_mat_local))
                ee_poses[that_robot_idx][:6] = mat_to_pose(np.linalg.inv(this_that_mat) @ that_sphere_mat_global @ np.linalg.inv(that_sphere_mat_local))

@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--output', '-o', required=True, help='Directory to save recording')
@click.option('--robot_config', '-rc', required=True, help='Path to robot_config yaml file')
@click.option('--match_dataset', '-m', default=None, help='Dataset used to overlay and adjust initial condition')
@click.option('--match_episode', '-me', default=None, type=int, help='Match specific episode from the match dataset')
@click.option('--match_camera', '-mc', default=0, type=int)
@click.option('--camera_reorder', '-cr', default='0')
@click.option('--vis_camera_idx', default=0, type=int, help="Which RealSense camera to visualize.")
@click.option('--init_joints', '-j', is_flag=True, default=False, help="Whether to initialize robot joint configuration in the beginning.")
@click.option('--steps_per_inference', '-si', default=6, type=int, help="Action horizon for inference.")
@click.option('--max_duration', '-md', default=2000000, help='Max duration for each epoch in seconds.')
@click.option('--max_timesteps', '-mt', default=500, help='Max steps for each epoch.')
@click.option('--frequency', '-f', default=10, type=float, help="Control frequency in Hz.")
@click.option('--command_latency', '-cl', default=0.01, type=float, help="Latency between receiving SapceMouse command to executing on Robot in Sec.")
@click.option('-nm', '--no_mirror', is_flag=True, default=False)
@click.option('-sf', '--sim_fov', type=float, default=None)
@click.option('-ci', '--camera_intrinsics', type=str, default=None)
@click.option('--mirror_swap', is_flag=True, default=False)
@click.option('--temporal_agg', is_flag=True, default=False)
@click.option('--ensemble_steps', type=int, default=8)
def main(input, output, robot_config, 
    match_dataset, match_episode, match_camera,
    camera_reorder,
    vis_camera_idx, init_joints, 
    steps_per_inference, max_duration, max_timesteps,
    frequency, command_latency, 
    no_mirror, sim_fov, camera_intrinsics, mirror_swap, temporal_agg, ensemble_steps):
    max_gripper_width = 0.09
    gripper_speed = 0.2
    
    # load robot config file
    robot_config_data = yaml.safe_load(open(os.path.expanduser(robot_config), 'r'))
    
    # load left-right robot relative transform
    tx_left_right = np.array(robot_config_data['tx_left_right'])
    tx_robot1_robot0 = tx_left_right
    
    robots_config = robot_config_data['robots']
    grippers_config = robot_config_data['grippers']

    # load checkpoint
    ckpt_path = input
    if not ckpt_path.endswith('.ckpt'):
        ckpt_path = os.path.join(ckpt_path, 'checkpoints', 'latest.ckpt')
    payload = torch.load(open(ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
    cfg = payload['cfg']

    if type(cfg.task.obs_down_sample_steps) == int:
        down_sample_steps = cfg.task.obs_down_sample_steps // cfg.task.action_down_sample_steps
    elif type(cfg.task.obs_down_sample_steps) == omegaconf.listconfig.ListConfig:
        down_sample_steps = [0] + [x // cfg.task.action_down_sample_steps for x in cfg.task.obs_down_sample_steps]
        down_sample_steps = down_sample_steps[::-1]
    
    print("model_name:", cfg.policy.obs_encoder.model_name)
    print("dataset_path:", cfg.task.dataset.dataset_path)

    # setup experiment
    dt = 1/frequency

    obs_res = get_real_obs_resolution(cfg.task.shape_meta)
    # load fisheye converter
    fisheye_converter = None
    if sim_fov is not None:
        assert camera_intrinsics is not None
        opencv_intr_dict = parse_fisheye_intrinsics(
            json.load(open(camera_intrinsics, 'r')))
        fisheye_converter = FisheyeRectConverter(
            **opencv_intr_dict,
            out_size=obs_res,
            out_fov=sim_fov
        )

    print("steps_per_inference:", steps_per_inference)
    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager) as sm, \
            KeystrokeCounter() as key_counter, \
            BimanualUmiEnv(
                output_dir=output,
                robots_config=robots_config,
                grippers_config=grippers_config,
                frequency=frequency,
                obs_image_resolution=obs_res,
                obs_float32=True,
                camera_reorder=[int(x) for x in camera_reorder],
                init_joints=init_joints,
                enable_multi_cam_vis=True,
                # latency
                camera_obs_latency=0.17,

                # downsample
                camera_down_sample_steps=down_sample_steps,
                robot_down_sample_steps=down_sample_steps,
                gripper_down_sample_steps=down_sample_steps,

                # obs
                camera_obs_horizon=cfg.task.shape_meta.obs.camera0_rgb.horizon,
                robot_obs_horizon=cfg.task.shape_meta.obs.robot0_eef_pos.horizon,
                gripper_obs_horizon=cfg.task.shape_meta.obs.robot0_gripper_width.horizon,
                no_mirror=no_mirror,
                fisheye_converter=fisheye_converter,
                mirror_swap=mirror_swap,
                # action
                max_pos_speed=2.0,
                max_rot_speed=6.0,
                shm_manager=shm_manager) as env:
            cv2.setNumThreads(2)
            print("Waiting for camera")
            time.sleep(1.0)

            # load match_dataset
            episode_first_frame_map = dict()
            match_replay_buffer = None
            if match_dataset is not None:
                match_dir = pathlib.Path(match_dataset)
                match_zarr_path = match_dir.joinpath('replay_buffer.zarr')
                match_replay_buffer = ReplayBuffer.create_from_path(str(match_zarr_path), mode='r')
                match_video_dir = match_dir.joinpath('videos')
                for vid_dir in match_video_dir.glob("*/"):
                    episode_idx = int(vid_dir.stem)
                    match_video_path = vid_dir.joinpath(f'{match_camera}.mp4')
                    if match_video_path.exists():
                        img = None
                        with av.open(str(match_video_path)) as container:
                            stream = container.streams.video[0]
                            for frame in container.decode(stream):
                                img = frame.to_ndarray(format='rgb24')
                                break

                        episode_first_frame_map[episode_idx] = img
            print(f"Loaded initial frame for {len(episode_first_frame_map)} episodes")

            # creating model
            # have to be done after fork to prevent 
            # duplicating CUDA context with ffmpeg nvenc
            cls = hydra.utils.get_class(cfg._target_)
            workspace = cls(cfg)
            workspace: BaseWorkspace
            workspace.load_payload(payload, exclude_keys=None, include_keys=None)

            policy = workspace.model
            if cfg.training.use_ema:
                policy = workspace.ema_model
            policy.num_inference_steps = 16 # DDIM inference iterations
            obs_pose_rep = cfg.task.pose_repr.obs_pose_repr
            action_pose_repr = cfg.task.pose_repr.action_pose_repr
            print('obs_pose_rep', obs_pose_rep)
            print('action_pose_repr', action_pose_repr)


            device = torch.device('cuda')
            policy.eval().to(device)

            print("Warming up policy inference")
            obs = env.get_obs()
            episode_start_pose = list()
            for robot_id in range(len(robots_config)):
                pose = np.concatenate([
                    obs[f'robot{robot_id}_eef_pos'],
                    obs[f'robot{robot_id}_eef_rot_axis_angle']
                ], axis=-1)[-1]
                episode_start_pose.append(pose)
            with torch.no_grad():
                policy.reset()
                obs_dict_np = get_real_umi_obs_dict(
                    env_obs=obs, shape_meta=cfg.task.shape_meta, 
                    obs_pose_repr=obs_pose_rep,
                    tx_robot1_robot0=tx_robot1_robot0,
                    episode_start_pose=episode_start_pose)
                obs_dict = dict_apply(obs_dict_np, 
                    lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                result = policy.predict_action(obs_dict)
                action = result['action_pred'][0].detach().to('cpu').numpy()
                assert action.shape[-1] == 10 * len(robots_config)
                action = get_real_umi_action(action, obs, action_pose_repr)
                action_horizon = action.shape[0]
                action_dim = action.shape[-1]
                assert action.shape[-1] == 7 * len(robots_config)
                del result

            print('Ready!')
            if_first_time = True
            while True:
                # ========= human control loop ==========
                print("Human in control!")
                robot_states = env.get_robot_state()
                target_pose = np.stack([rs['ActualTCPPose'] for rs in robot_states])

                gripper_states = env.get_gripper_state()
                gripper_target_pos = np.asarray([gs['gripper_position'] for gs in gripper_states])
                
                control_robot_idx_list = [0]

                t_start = time.monotonic()
                iter_idx = 0
                while True:
                    # calculate timing
                    t_cycle_end = t_start + (iter_idx + 1) * dt
                    t_sample = t_cycle_end - command_latency
                    t_command_target = t_cycle_end + dt

                    # pump obs
                    obs = env.get_obs()

                    # visualize
                    episode_id = env.replay_buffer.n_episodes
                    vis_img = obs[f'camera{match_camera}_rgb'][-1]
                    match_episode_id = episode_id
                    if match_episode is not None:
                        match_episode_id = match_episode
                    if match_episode_id in episode_first_frame_map:
                        match_img = episode_first_frame_map[match_episode_id]
                        ih, iw, _ = match_img.shape
                        oh, ow, _ = vis_img.shape
                        tf = get_image_transform(
                            input_res=(iw, ih), 
                            output_res=(ow, oh), 
                            bgr_to_rgb=False)
                        match_img = tf(match_img).astype(np.float32) / 255
                        vis_img = (vis_img + match_img) / 2
                    obs_left_img = obs['camera0_rgb'][-1]
                    obs_right_img = obs['camera0_rgb'][-1]
                    vis_img = np.concatenate([obs_left_img, obs_right_img, vis_img], axis=1)
                    
                    text = f'Episode: {episode_id}'
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        lineType=cv2.LINE_AA,
                        thickness=3,
                        color=(0,0,0)
                    )
                    cv2.putText(
                        vis_img,
                        text,
                        (10,20),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5,
                        thickness=1,
                        color=(255,255,255)
                    )
                    cv2.imshow('default', vis_img[...,::-1])
                    _ = cv2.pollKey()
                    press_events = key_counter.get_press_events()
                    start_policy = False
                    if init_joints and if_first_time:
                        if_first_time = False
                        break
                    for key_stroke in press_events:
                        if key_stroke == KeyCode(char='q'):
                            # Exit program
                            env.end_episode()
                            exit(0)
                        elif key_stroke == KeyCode(char='c'):
                            # Exit human control loop
                            # hand control over to the policy
                            start_policy = True
                        elif key_stroke == KeyCode(char='e'):
                            # Next episode
                            if match_episode is not None:
                                match_episode = min(match_episode + 1, env.replay_buffer.n_episodes-1)
                        elif key_stroke == KeyCode(char='w'):
                            # Prev episode
                            if match_episode is not None:
                                match_episode = max(match_episode - 1, 0)
                        elif key_stroke == KeyCode(char='m'):
                            # move the robot
                            duration = 3.0
                            ep = match_replay_buffer.get_episode(match_episode_id)

                            for robot_idx in range(1):
                                pos = ep[f'robot{robot_idx}_eef_pos'][0]
                                rot = ep[f'robot{robot_idx}_eef_rot_axis_angle'][0]
                                grip = ep[f'robot{robot_idx}_gripper_width'][0]
                                pose = np.concatenate([pos, rot])
                                env.robots[robot_idx].servoL(pose, duration=duration)
                                env.grippers[robot_idx].schedule_waypoint(grip, target_time=time.time() + duration)
                                target_pose[robot_idx] = pose
                                gripper_target_pos[robot_idx] = grip
                            time.sleep(duration)

                        elif key_stroke == Key.backspace:
                            if click.confirm('Are you sure to drop an episode?'):
                                env.drop_episode()
                                key_counter.clear()
                        elif key_stroke == KeyCode(char='a'):
                            control_robot_idx_list = list(range(target_pose.shape[0]))
                        elif key_stroke == KeyCode(char='1'):
                            control_robot_idx_list = [0]
                        elif key_stroke == KeyCode(char='2'):
                            control_robot_idx_list = [1]

                    if start_policy:
                        break

                    precise_wait(t_sample)
                    # get teleop command
                    sm_state = sm.get_motion_state_transformed()
                    # print(sm_state)
                    dpos = sm_state[:3] * (0.5 / frequency)
                    drot_xyz = sm_state[3:] * (1.5 / frequency)

                    drot = st.Rotation.from_euler('xyz', drot_xyz)
                    for robot_idx in control_robot_idx_list:
                        target_pose[robot_idx, :3] += dpos
                        target_pose[robot_idx, 3:] = (drot * st.Rotation.from_rotvec(
                            target_pose[robot_idx, 3:])).as_rotvec()

                    dpos = 0
                    if sm.is_button_pressed(0):
                        # close gripper
                        dpos = -gripper_speed / frequency
                    if sm.is_button_pressed(1):
                        dpos = gripper_speed / frequency
                    for robot_idx in control_robot_idx_list:
                        gripper_target_pos[robot_idx] = np.clip(gripper_target_pos[robot_idx] + dpos, 0, max_gripper_width)

                    # solve collision with table
                    for robot_idx in control_robot_idx_list:
                        solve_table_collision(
                            ee_pose=target_pose[robot_idx],
                            gripper_width=gripper_target_pos[robot_idx],
                            height_threshold=robots_config[robot_idx]['height_threshold'])
                    
                    # solve collison between two robots
                    solve_sphere_collision(
                        ee_poses=target_pose,
                        robots_config=robots_config
                    )

                    action = np.zeros((7 * target_pose.shape[0],))

                    for robot_idx in range(target_pose.shape[0]):
                        action[7 * robot_idx + 0: 7 * robot_idx + 6] = target_pose[robot_idx]
                        action[7 * robot_idx + 6] = gripper_target_pos[robot_idx]


                    # execute teleop command
                    env.exec_actions(
                        actions=[action], 
                        timestamps=[t_command_target-time.monotonic()+time.time()],
                        compensate_latency=False)
                    precise_wait(t_cycle_end)
                    iter_idx += 1
                
                # ========== policy control loop ==============
                try:
                    # start episode
                    policy.reset()
                    start_delay = 1.0
                    eval_t_start = time.time() + start_delay
                    t_start = time.monotonic() + start_delay
                    env.start_episode(eval_t_start)

                    # get current pose
                    obs = env.get_obs()
                    episode_start_pose = list()
                    for robot_id in range(len(robots_config)):
                        pose = np.concatenate([
                            obs[f'robot{robot_id}_eef_pos'],
                            obs[f'robot{robot_id}_eef_rot_axis_angle']
                        ], axis=-1)[-1]
                        episode_start_pose.append(pose)

                    # wait for 1/30 sec to get the closest frame actually
                    # reduces overall latency
                    frame_latency = 1/60
                    precise_wait(eval_t_start - frame_latency, time_func=time.time)
                    print("Started!")
                    iter_idx = 0
                    inference_idx = steps_per_inference
                    all_time_actions = np.zeros((max_timesteps, max_timesteps + action_horizon, action_dim))

                    while True:
                        # calculate timing
                        t_cycle_end = t_start + (iter_idx + 1) * dt

                        # get obs
                        obs = env.get_obs()
                        obs_timestamps = obs['timestamp']
                        print(f'Obs latency {time.time() - obs_timestamps[-1]}')

                        # run inference
                        with torch.no_grad():
                            s = time.time()
                            obs_dict_np = get_real_umi_obs_dict(
                                env_obs=obs, shape_meta=cfg.task.shape_meta, 
                                obs_pose_repr=obs_pose_rep,
                                tx_robot1_robot0=tx_robot1_robot0,
                                episode_start_pose=episode_start_pose)
                            obs_dict = dict_apply(obs_dict_np, 
                                lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
                            result = policy.predict_action(obs_dict)
                            raw_action = result['action_pred'][0].detach().to('cpu').numpy()
                            action = get_real_umi_action(raw_action, obs, action_pose_repr)  # (16, 7)
                            print('Inference latency:', time.time() - s)
                            all_time_actions[[iter_idx], iter_idx:iter_idx + action_horizon] = action

                        if inference_idx == steps_per_inference:
                            inference_idx = 0

                            if temporal_agg:
                                # temporal ensemble
                                action_seq_for_curr_step = all_time_actions[:, iter_idx:iter_idx + action_horizon]
                                target_pose_list = []
                                for i in range(action_horizon):
                                    actions_for_curr_step = action_seq_for_curr_step[max(0, iter_idx - ensemble_steps + 1):iter_idx + 1, i]
                                    actions_populated = np.all(actions_for_curr_step != 0, axis=1)
                                    actions_for_curr_step = actions_for_curr_step[actions_populated]

                                    k = -0.01
                                    exp_weights = np.exp(k * np.arange(len(actions_for_curr_step)))
                                    exp_weights = exp_weights / exp_weights.sum()
                                    weighted_rotvec = R.from_rotvec(np.array(actions_for_curr_step)[:, 3:6]).mean(weights=exp_weights).as_rotvec()
                                    weighted_action = (actions_for_curr_step * exp_weights[:, np.newaxis]).sum(axis=0, keepdims=True)
                                    weighted_action[0][3:6] = weighted_rotvec
                                    target_pose_list.append(weighted_action)
                                this_target_poses = np.concatenate(target_pose_list, axis=0)
                            else:
                                this_target_poses = action

                            assert this_target_poses.shape[1] == len(robots_config) * 7
                            for target_pose in this_target_poses:
                                for robot_idx in range(len(robots_config)):
                                    solve_table_collision(
                                        ee_pose=target_pose[robot_idx * 7: robot_idx * 7 + 6],
                                        gripper_width=target_pose[robot_idx * 7 + 6],
                                        height_threshold=robots_config[robot_idx]['height_threshold']
                                    )

                                # solve collison between two robots
                                solve_sphere_collision(
                                    ee_poses=target_pose.reshape([len(robots_config), -1]),
                                    robots_config=robots_config
                                )

                            # deal with timing
                            # the same step actions are always the target for
                            action_timestamps = (np.arange(len(action), dtype=np.float64)) * dt + obs_timestamps[-1]
                            action_exec_latency = 0.01
                            curr_time = time.time()
                            is_new = action_timestamps > (curr_time + action_exec_latency)
                            if np.sum(is_new) == 0:
                                # exceeded time budget, still do something
                                this_target_poses = this_target_poses[[-1]]  # (1, 7)
                                # schedule on next available step
                                next_step_idx = int(np.ceil((curr_time - eval_t_start) / dt))
                                action_timestamp = eval_t_start + (next_step_idx) * dt
                                print('Over budget', action_timestamp - curr_time)
                                action_timestamps = np.array([action_timestamp])
                            else:
                                this_target_poses = this_target_poses[is_new]
                                action_timestamps = action_timestamps[is_new]

                            # execute actions
                            env.exec_actions(
                                actions=this_target_poses,
                                timestamps=action_timestamps,
                                compensate_latency=True
                            )
                            print(iter_idx, f"Submitted {len(this_target_poses)} steps of actions.")

                        # visualize
                        episode_id = env.replay_buffer.n_episodes
                        obs_left_img = obs['camera0_rgb'][-1]
                        obs_right_img = obs['camera0_rgb'][-1]
                        vis_img = np.concatenate([obs_left_img, obs_right_img], axis=1)
                        text = 'Episode: {}, Time: {:.1f}'.format(
                            episode_id, time.monotonic() - t_start
                        )
                        cv2.putText(
                            vis_img,
                            text,
                            (10,20),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.5,
                            thickness=1,
                            color=(255,255,255)
                        )
                        cv2.imshow('default', vis_img[...,::-1])

                        _ = cv2.pollKey()
                        press_events = key_counter.get_press_events()
                        stop_episode = False
                        for key_stroke in press_events:
                            if key_stroke == KeyCode(char='s'):
                                # Stop episode
                                # Hand control back to human
                                print('Stopped.')
                                stop_episode = True

                        t_since_start = time.time() - eval_t_start
                        if t_since_start > max_duration:
                            print("Max Duration reached.")
                            stop_episode = True
                        if iter_idx + 1 >= max_timesteps:
                            print("Max Timesteps reached.")
                            stop_episode = True
                        if stop_episode:
                            env.end_episode()
                            break

                        # wait for execution
                        precise_wait(t_cycle_end - frame_latency)
                        iter_idx += 1
                        inference_idx += 1

                except KeyboardInterrupt:
                    print("Interrupted!")
                    # stop robot.
                    env.end_episode()
                
                print("Stopped.")



# %%
if __name__ == '__main__':
    main()