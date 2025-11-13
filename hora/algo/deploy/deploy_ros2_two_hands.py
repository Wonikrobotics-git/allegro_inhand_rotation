#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two-hands deployment script for Allegro hands (left + right)
Based on deploy_ros2.py but extended for bimanual control
"""
import time
import signal

import numpy as np
import torch

# Two hands (left + right)
from hora.algo.deploy.robots.allegro_ros2 import start_allegro_ios, stop_allegro_ios

from hora.algo.models.models import ActorCritic
from hora.algo.models.running_mean_std import RunningMeanStd
from hora.utils.misc import tprint


# =========================================================
# Conversion/rearrangement utility (for single hand)
# =========================================================

def _action_hora2allegro(actions):
    """Convert hora ordering to allegro ordering for a single hand"""
    if isinstance(actions, torch.Tensor):
        if actions.dim() > 1:
            actions = actions.view(-1)
        cmd_act = actions.clone()
        temp = actions[[4, 5, 6, 7]].clone()
        cmd_act[[4, 5, 6, 7]] = actions[[8, 9, 10, 11]]
        cmd_act[[12, 13, 14, 15]] = temp
        cmd_act[[8, 9, 10, 11]] = actions[[12, 13, 14, 15]]
        return cmd_act
    else:
        a = np.asarray(actions).flatten()
        cmd = a.copy()
        temp = a[[4, 5, 6, 7]].copy()
        cmd[[4, 5, 6, 7]] = a[[8, 9, 10, 11]]
        cmd[[12, 13, 14, 15]] = temp
        cmd[[8, 9, 10, 11]] = a[[12, 13, 14, 15]]
        return cmd


def _obs_allegro2hora(o):
    """Convert allegro ordering to hora ordering for a single hand"""
    # allegro: index - middle - ring - thumb
    # hora   : index, thumb, middle, ring
    return np.concatenate([o[0:4], o[12:16], o[4:8], o[8:12]]).astype(np.float64)


def _reorder_imrt2timr(imrt):
    """[ROS1] index-middle-ring-thumb → [ROS2] thumb-index-middle-ring"""
    return np.concatenate([imrt[12:16], imrt[0:12]]).astype(np.float64)


def _reorder_timr2imrt(timr):
    """[ROS2] thumb-index-middle-ring → [ROS1] index-middle-ring-thumb"""
    return np.concatenate([timr[4:16], timr[0:4]]).astype(np.float64)


# =========================================================
# Control agent (Timer-based, two hands)
# =========================================================

class HardwarePlayerTwoHands:
    """
    Two-hands hardware player for bimanual Allegro hand control.
    Uses the same single-hand model independently for each hand.
    """
    def __init__(self, hz: float = 20.0, device: str = "cuda", debug: bool = False):
        torch.set_grad_enabled(False)
        self.hz = float(hz)
        self.device = device
        self.debug = debug
        self._debug_counter = 0

        # Model configuration for single hand (will be used twice)
        obs_shape = (96,)  # Single hand observation

        # Create two separate model instances (one per hand)
        # Note: We need separate config dicts because ActorCritic.__init__ uses pop()
        net_config_right = {
            "actions_num": 16,  # Single hand
            "input_shape": obs_shape,
            "actor_units": [512, 256, 128],
            "priv_mlp_units": [256, 128, 8],
            "priv_info": True,
            "proprio_adapt": True,
            "priv_info_dim": 9,
        }
        net_config_left = {
            "actions_num": 16,  # Single hand
            "input_shape": obs_shape,
            "actor_units": [512, 256, 128],
            "priv_mlp_units": [256, 128, 8],
            "priv_info": True,
            "proprio_adapt": True,
            "priv_info_dim": 9,
        }

        self.model_right = ActorCritic(net_config_right).to(self.device).eval()
        self.model_left = ActorCritic(net_config_left).to(self.device).eval()

        # Create separate normalization for each hand
        self.running_mean_std_right = RunningMeanStd(obs_shape).to(self.device).eval()
        self.running_mean_std_left = RunningMeanStd(obs_shape).to(self.device).eval()
        self.sa_mean_std_right = RunningMeanStd((30, 32)).to(self.device).eval()
        self.sa_mean_std_left = RunningMeanStd((30, 32)).to(self.device).eval()

        # Buffers (separate for each hand)
        self.obs_buf_right = torch.zeros((1, 96), dtype=torch.float32, device=self.device)
        self.obs_buf_left = torch.zeros((1, 96), dtype=torch.float32, device=self.device)
        self.proprio_hist_buf_right = torch.zeros((1, 30, 32), dtype=torch.float32, device=self.device)
        self.proprio_hist_buf_left = torch.zeros((1, 30, 32), dtype=torch.float32, device=self.device)

        # Limits (same for both hands)
        self.allegro_dof_lower = torch.tensor([
            -0.4700, -0.1960, -0.1740, -0.2270,   # Index
             0.2630, -0.1050, -0.1890, -0.1620,   # Thumb
            -0.4700, -0.1960, -0.1740, -0.2270,   # Middle
            -0.4700, -0.1960, -0.1740, -0.2270,   # Ring
        ], dtype=torch.float32, device=self.device)
        self.allegro_dof_upper = torch.tensor([
             0.4700, 1.6100, 1.7090, 1.6180,      # Index
             1.3960, 1.1630, 1.6440, 1.7190,      # Thumb
             0.4700, 1.6100, 1.7090, 1.6180,      # Middle
             0.4700, 1.6100, 1.7090, 1.6180,      # Ring
        ], dtype=torch.float32, device=self.device)

        # Initial poses (same for both hands)
        self.init_pose = [
            0.0627, 1.2923, 0.3383, 0.1088,
            0.0724, 1.1983, 0.1551, 0.1499,
            0.1343, 1.1736, 0.5355, 0.2164,
            1.1202, 1.1374, 0.8535, -0.0852,
        ]

        # State (separate for each hand)
        self.action_scale = 1.0 / 24.0
        self.prev_target_right = torch.zeros((1, 16), dtype=torch.float32, device=self.device)
        self.prev_target_left = torch.zeros((1, 16), dtype=torch.float32, device=self.device)
        self.cur_target_right = torch.zeros((1, 16), dtype=torch.float32, device=self.device)
        self.cur_target_left = torch.zeros((1, 16), dtype=torch.float32, device=self.device)
        self._last_obs_q_right = None
        self._last_obs_q_left = None
        self._skipped_right = 0
        self._skipped_left = 0
        self._last_step_t = None

        # ROS (two hands)
        self.timer = None
        self.allegro_ios = None  # Will be a dict: {'right': io_right, 'left': io_left}

    # ---------- utils ----------
    @staticmethod
    def _unscale(x, lower, upper):
        return (2.0 * x - upper - lower) / (upper - lower)

    def _pre_physics_step_single(self, action, prev_target):
        """Update target for a single hand"""
        target = prev_target + self.action_scale * action
        cur_target = torch.clamp(target, min=self.allegro_dof_lower, max=self.allegro_dof_upper)
        return cur_target

    def _post_physics_step_single(self, obses, cur_target, obs_buf, proprio_hist_buf):
        """
        Update observation buffers for a single hand
        obses: (16,) or (1,16) tensor on self.device
        """
        # 1) Normalize current observation
        cur_obs = self._unscale(
            obses.view(-1), self.allegro_dof_lower, self.allegro_dof_upper
        ).view(1, 16)

        # 2) Roll obs_buf (96 = 32*3)
        src64 = obs_buf[:, 32:96].clone()
        obs_buf[:, 0:64] = src64
        obs_buf[:, 64:80] = cur_obs
        obs_buf[:, 80:96] = cur_target

        # 3) Roll proprio_hist_buf (T=30)
        src_hist = proprio_hist_buf[:, 1:, :].clone()
        proprio_hist_buf[:, 0:-1, :] = src_hist
        proprio_hist_buf[:, -1, :16] = cur_obs
        proprio_hist_buf[:, -1, 16:32] = cur_target

    # ---------- timer callback ----------
    @torch.inference_mode()
    def _control_step(self):
        t0 = time.perf_counter()

        # Process right hand
        # 1) Normalize observations
        obs_norm_right = self.running_mean_std_right(self.obs_buf_right)

        # 2) Inference
        input_dict_right = {
            "obs": obs_norm_right,
            "proprio_hist": self.sa_mean_std_right(self.proprio_hist_buf_right),
        }
        action_right = torch.clamp(self.model_right.act_inference(input_dict_right), -1.0, 1.0)

        # 3) Update target
        self.cur_target_right = self._pre_physics_step_single(action_right, self.prev_target_right)
        self.prev_target_right = self.cur_target_right

        # Process left hand (independent model)
        # 1) Normalize observations
        obs_norm_left = self.running_mean_std_left(self.obs_buf_left)

        # 2) Inference
        input_dict_left = {
            "obs": obs_norm_left,
            "proprio_hist": self.sa_mean_std_left(self.proprio_hist_buf_left),
        }
        action_left = torch.clamp(self.model_left.act_inference(input_dict_left), -1.0, 1.0)

        # 3) Update target
        self.cur_target_left = self._pre_physics_step_single(action_left, self.prev_target_left)
        self.prev_target_left = self.cur_target_left

        # 4) Publish commands to both hands
        # Right hand
        cmd_right = self.cur_target_right.detach().to("cpu").numpy()[0]
        ros1_right = _action_hora2allegro(cmd_right)
        ros2_right = _reorder_imrt2timr(ros1_right)
        self.allegro_ios["right"].command_joint_position(ros2_right)

        # Left hand
        cmd_left = self.cur_target_left.detach().to("cpu").numpy()[0]
        ros1_left = _action_hora2allegro(cmd_left)
        ros2_left = _reorder_imrt2timr(ros1_left)
        self.allegro_ios["left"].command_joint_position(ros2_left)

        # Debug logging (every 100 steps)
        if self.debug and self._debug_counter % 100 == 0:
            print(f"\n[DEBUG] Step {self._debug_counter}")
            print(f"  cmd_right (hora): {cmd_right[:4]}")  # Just thumb for brevity
            print(f"  cmd_left  (hora): {cmd_left[:4]}")
            print(f"  ros2_right[:4]: {ros2_right[:4]}")
            print(f"  ros2_left[:4]:  {ros2_left[:4]}")
        self._debug_counter += 1

        # 5) Non-blocking obs update for both hands
        # Right hand
        q_pos_right = self.allegro_ios["right"].poll_joint_position(wait=False, timeout=0.0)
        if q_pos_right is not None:
            ros1_q_right = _reorder_timr2imrt(q_pos_right)
            hora_q_right = _obs_allegro2hora(ros1_q_right)
            obs_q_right = torch.from_numpy(hora_q_right.astype(np.float32)).to(self.device)
            self._last_obs_q_right = obs_q_right
        else:
            obs_q_right = self._last_obs_q_right
            self._skipped_right += 1

        # Left hand
        q_pos_left = self.allegro_ios["left"].poll_joint_position(wait=False, timeout=0.0)
        if q_pos_left is not None:
            ros1_q_left = _reorder_timr2imrt(q_pos_left)
            hora_q_left = _obs_allegro2hora(ros1_q_left)
            obs_q_left = torch.from_numpy(hora_q_left.astype(np.float32)).to(self.device)
            self._last_obs_q_left = obs_q_left
        else:
            obs_q_left = self._last_obs_q_left
            self._skipped_left += 1

        # Update buffers for each hand
        if obs_q_right is not None:
            self._post_physics_step_single(obs_q_right, self.cur_target_right,
                                          self.obs_buf_right, self.proprio_hist_buf_right)
        if obs_q_left is not None:
            self._post_physics_step_single(obs_q_left, self.cur_target_left,
                                          self.obs_buf_left, self.proprio_hist_buf_left)

        # 6) Light jitter log
        if self._last_step_t is None:
            self._last_step_t = t0
        else:
            dt = t0 - self._last_step_t
            self._last_step_t = t0
            # Log every 5 seconds
            if int(time.time()) % 5 == 0:
                hz_est = 1.0 / max(dt, 1e-6)
                print(f"[timer] {hz_est:.2f} Hz, skipped_right={self._skipped_right}, skipped_left={self._skipped_left}")

    # ---------- deploy ----------
    def deploy(self):
        run_start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        print(f"🧠🧠 Starting Two-Hands HardwarePlayer deployment at {run_start_time}...")

        # Start ROS2 I/O for both hands
        self.allegro_ios = start_allegro_ios(sides=("right", "left"))

        # Warmup - settle hardware for both hands
        warmup = int(self.hz * 4)
        for t in range(warmup):
            tprint(f"setup {t} / {warmup}")
            pose = _reorder_imrt2timr(np.array(self.init_pose, dtype=np.float64))
            self.allegro_ios["right"].command_joint_position(pose)
            self.allegro_ios["left"].command_joint_position(pose)
            time.sleep(1.0 / self.hz)

        # First observation (blocking) - initialization stability
        q_pos_right = self.allegro_ios["right"].poll_joint_position(wait=True, timeout=5.0)
        q_pos_left = self.allegro_ios["left"].poll_joint_position(wait=True, timeout=5.0)

        if q_pos_right is None or q_pos_left is None:
            print("❌ Failed to read joint state from one or both hands.")
            stop_allegro_ios(self.allegro_ios)
            return

        # Convert to hora format for right hand
        ros1_q_right = _reorder_timr2imrt(q_pos_right)
        hora_q_right = _obs_allegro2hora(ros1_q_right)
        obs_q_right = torch.from_numpy(hora_q_right.astype(np.float32)).to(self.device)
        self._last_obs_q_right = obs_q_right

        # Convert to hora format for left hand
        ros1_q_left = _reorder_timr2imrt(q_pos_left)
        hora_q_left = _obs_allegro2hora(ros1_q_left)
        obs_q_left = torch.from_numpy(hora_q_left.astype(np.float32)).to(self.device)
        self._last_obs_q_left = obs_q_left

        # Initialize buffers for right hand
        cur_obs_buf_right = self._unscale(obs_q_right, self.allegro_dof_lower, self.allegro_dof_upper)[None]
        self.prev_target_right = obs_q_right[None]
        for i in range(3):
            self.obs_buf_right[:, i*32:i*32+16] = cur_obs_buf_right
            self.obs_buf_right[:, i*32+16:i*32+32] = self.prev_target_right
        self.proprio_hist_buf_right[:, :, :16] = cur_obs_buf_right
        self.proprio_hist_buf_right[:, :, 16:32] = self.prev_target_right

        # Initialize buffers for left hand
        cur_obs_buf_left = self._unscale(obs_q_left, self.allegro_dof_lower, self.allegro_dof_upper)[None]
        self.prev_target_left = obs_q_left[None]
        for i in range(3):
            self.obs_buf_left[:, i*32:i*32+16] = cur_obs_buf_left
            self.obs_buf_left[:, i*32+16:i*32+32] = self.prev_target_left
        self.proprio_hist_buf_left[:, :, :16] = cur_obs_buf_left
        self.proprio_hist_buf_left[:, :, 16:32] = self.prev_target_left

        # Register timer (we can use either hand's node for timer)
        period = 1.0 / self.hz
        self.timer = self.allegro_ios["right"].create_timer(period, self._control_step)
        print(f"Deployment started (timer-based {self.hz:.1f} Hz, TWO HANDS). Ctrl+C to stop.")

        # Main thread: signal handling + keep alive
        interrupted = False

        def _sigint(_sig, _frm):
            nonlocal interrupted
            interrupted = True
        signal.signal(signal.SIGINT, _sigint)

        try:
            while not interrupted:
                time.sleep(0.2)
        finally:
            try:
                if self.timer is not None:
                    self.timer.cancel()
            except Exception:
                pass
            try:
                self.allegro_ios["right"].go_safe()
                self.allegro_ios["left"].go_safe()
            except Exception:
                pass
            stop_allegro_ios(self.allegro_ios)
            print("🧠🧠 Two-Hands Deployment stopped cleanly.")

            run_end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

            # Calculate elapsed time
            fmt = "%Y-%m-%d %H:%M:%S"
            try:
                t0 = time.mktime(time.strptime(run_start_time, fmt))
                t1 = time.mktime(time.strptime(run_end_time, fmt))
                elapsed = int(round(t1 - t0))
                hrs, rem = divmod(elapsed, 3600)
                mins, secs = divmod(rem, 60)
                print(f"🧠🧠 Total Running Time: {hrs:02d}:{mins:02d}:{secs:02d}")
            except Exception:
                print(f"🔥 Run started at {run_start_time}, ended at {run_end_time}")

    # ---------- checkpoint ----------
    def restore(self, checkpoint_right, checkpoint_left=None):
        """
        Load checkpoints for both hands independently.

        Args:
            checkpoint_right: Path to checkpoint for right hand
            checkpoint_left: Path to checkpoint for left hand (if None, uses same as right)
        """
        # Load right hand checkpoint
        print(f"📦 Loading right hand checkpoint: {checkpoint_right}")
        ckpt_right = torch.load(checkpoint_right, map_location=self.device)
        self.running_mean_std_right.load_state_dict(ckpt_right["running_mean_std"])
        self.model_right.load_state_dict(ckpt_right["model"])
        self.sa_mean_std_right.load_state_dict(ckpt_right["sa_mean_std"])

        # Load left hand checkpoint
        if checkpoint_left is None:
            print(f"📦 Using same checkpoint for left hand")
            checkpoint_left = checkpoint_right

        if checkpoint_left == checkpoint_right:
            # Same checkpoint, just copy the weights
            self.running_mean_std_left.load_state_dict(ckpt_right["running_mean_std"])
            self.model_left.load_state_dict(ckpt_right["model"])
            self.sa_mean_std_left.load_state_dict(ckpt_right["sa_mean_std"])
        else:
            print(f"📦 Loading left hand checkpoint: {checkpoint_left}")
            ckpt_left = torch.load(checkpoint_left, map_location=self.device)
            self.running_mean_std_left.load_state_dict(ckpt_left["running_mean_std"])
            self.model_left.load_state_dict(ckpt_left["model"])
            self.sa_mean_std_left.load_state_dict(ckpt_left["sa_mean_std"])


# =========================================================
# Execution example
# =========================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Two-Hands Allegro Hand Deployment")
    parser.add_argument("--checkpoint-right", type=str, required=True,
                        help="Path to checkpoint file for right hand (.pth)")
    parser.add_argument("--checkpoint-left", type=str, default=None,
                        help="Path to checkpoint file for left hand (.pth). If not specified, uses same as right.")
    parser.add_argument("--hz", type=float, default=20.0,
                        help="Control frequency in Hz (default: 20.0)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use (cuda/cpu). Auto-detects if not specified.")
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug output")

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    # Create agent for two hands
    agent = HardwarePlayerTwoHands(hz=args.hz, device=device, debug=args.debug)

    # Load checkpoints
    agent.restore(args.checkpoint_right, args.checkpoint_left)

    agent.deploy()
