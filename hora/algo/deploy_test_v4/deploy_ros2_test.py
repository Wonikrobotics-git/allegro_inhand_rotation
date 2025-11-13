"""
IsaacGym의 아웃풋을 그대로 실행했을때 그대로 하드웨어에서 동작하는지 테스트
추후 pd gain 튜닝 스크립트로 활용
"""

import os
import time
import numpy as np
import torch

# Allegro I/O (백그라운드 스피너 사용)
from robots.allegro_ros2 import start_allegro_io, stop_allegro_io


class HardwarePlayerTest(object):
    def __init__(self, config, ros2: bool = True):
        self.deploy = self.deploy_ros2

        self.action_scale = 1 / 24
        self.actions_num = 16
        self.device = "cpu"

        # Load actions from file
        self.actions = None
        actions_file = '/home/avery/Documents/isaacgym_allegro_hand/rma/actions_500.npz'
        if os.path.exists(actions_file):
            # try:
            data = np.load(actions_file)
            if 'actions' in data:
                self.actions = data['actions']
                print(f"Successfully loaded {self.actions.shape[0]} actions from {actions_file} using key 'actions'.")
        #         else:
        #             print(f"ERROR: Could not find 'actions' key in {actions_file}")
        #             if data.files:
        #                 print(f"Available keys in the file: {data.files}")
        #             else:
        #                 print("The file seems to be empty or not a valid npz file.")
        #     except Exception as e:
        #         print(f"ERROR: Failed to load actions from {actions_file}: {e}")
        # else:
        #     print(f"ERROR: Actions file not found at {actions_file}")

        # ===== Data Capture =====
        num_actions = self.actions.shape[0] if self.actions is not None else 0
        self.history_command_pos = np.zeros((num_actions, 16), dtype=np.float32)
        self.history_measured_pos = np.zeros((num_actions, 16), dtype=np.float32)
        self._snap_dir = "./snapshots"

        # ===== Allegro joint limits =====
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

        # ===== Poses =====
        self.init_pose = [
            0.0458,  1.0644,  0.4829,  0.4421,       # Index
            -0.1064,  1.3008,  0.1961,  0.1274,      # Middle
            0.0557,  1.1238,  0.4041,  0.3523,       # Ring
            1.1425,  0.9528,  1.1032, -0.1087,       # Thumb
        ]

        self.home_pose = [0.0]*12 + [1.2, 1.0, 0.0, 0.0]

        # ===== Targets (rad) =====
        self.prev_target = torch.zeros((1, 16), dtype=torch.float32)
        self.cur_target  = torch.zeros((1, 16), dtype=torch.float32)

    # ---------- Reorder helpers ----------
    def _action_hora2allegro(self, actions):
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

    def _obs_allegro2hora(self, o):
        # allegro: index - middle - ring - thumb
        # hora  : index, thumb, middle, ring
        return np.concatenate([o[0:4], o[12:16], o[4:8], o[8:12]]).astype(np.float64)

    def _reorder_imrt2timr(self, imrt):
        # [ROS1] index - middle - ring - thumb  ->  [ROS2] thumb - index - middle - ring
        return np.concatenate([imrt[12:16], imrt[0:12]]).astype(np.float64)

    def _reorder_timr2imrt(self, timr):
        # [ROS2] thumb - index - middle - ring  ->  [ROS1] index - middle - ring - thumb
        return np.concatenate([timr[4:16], timr[0:4]]).astype(np.float64)

    # ---------- Save and Plot ----------
    def _save_and_plot_error(self):
        os.makedirs(self._snap_dir, exist_ok=True)

        # ===== Save NPZ =====
        base = os.path.join(self._snap_dir, "test_run_data")
        np.savez(base + ".npz",
                 command_pos=self.history_command_pos,
                 measured_pos=self.history_measured_pos)
        print(f"[saved] {base}.npz")

        # ===== Plot =====
        import matplotlib
        if os.environ.get("DISPLAY", "") == "":
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Plot 1: Command vs Measured
        fig_pos = plt.figure(figsize=(16, 12))
        fig_pos.suptitle('Command vs Measured Position')
        x = np.arange(self.history_command_pos.shape[0])
        for i in range(16):
            ax = fig_pos.add_subplot(4, 4, i + 1)
            ax.plot(x, self.history_command_pos[:, i], label="command (rad)", linewidth=1.0)
            ax.plot(x, self.history_measured_pos[:, i], label="measured (rad)", linewidth=1.0, linestyle='--')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"DOF {i}", fontsize=9)
            if i >= 12: ax.set_xlabel("steps")
            if i == 0: ax.legend(loc="upper right", fontsize=8)
        fig_pos.tight_layout(rect=[0, 0.03, 1, 0.95])
        png_pos = os.path.join(self._snap_dir, "test_run_positions.png")
        fig_pos.savefig(png_pos, dpi=160, bbox_inches="tight")
        print(f"[saved] {png_pos}")

        # Plot 2: Error
        error = np.abs(self.history_command_pos - self.history_measured_pos)
        fig_err = plt.figure(figsize=(16, 12))
        fig_err.suptitle('Position Error (Command - Measured)')
        for i in range(16):
            ax = fig_err.add_subplot(4, 4, i + 1)
            ax.plot(x, error[:, i], label="error (rad)", linewidth=1.0, color='r')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"DOF {i}", fontsize=9)
            if i >= 12: ax.set_xlabel("steps")
            if i == 0: ax.legend(loc="upper right", fontsize=8)
        fig_err.tight_layout(rect=[0, 0.03, 1, 0.95])
        png_err = os.path.join(self._snap_dir, "test_run_error.png")
        fig_err.savefig(png_err, dpi=160, bbox_inches="tight")
        print(f"[saved] {png_err}")

        try:
            plt.show()
        except Exception:
            pass

    # ---------- Deploy ----------
    def deploy_ros2(self):
        if self.actions is None:
            print("No actions loaded, cannot deploy.")
            return

        # 1) Allegro I/O 노드 시작
        self.allegro = start_allegro_io(side='right')

        # 2) 초기 셋업 루프
        warmup = 20 * 4
        for t in range(warmup):
            print(f"setup {t} / {warmup}")
            pose = self._reorder_imrt2timr(np.array(self.init_pose, dtype=np.float64))
            self.allegro.command_joint_position(pose)
            time.sleep(0.05)

        # 3) 첫 관측
        q_pos = self.allegro.poll_joint_position(wait=True, timeout=5.0)
        if q_pos is None:
            print("❌ failed to read joint state.")
            stop_allegro_io(self.allegro)
            return

        ros1_q = self._reorder_timr2imrt(q_pos)
        hora_q = self._obs_allegro2hora(ros1_q)
        obs_q = torch.from_numpy(hora_q.astype(np.float32)).cpu()

        # prev_target은 실측(rad)
        self.prev_target = obs_q[None].clone()

        timestep = 0
        print("Deployment started. Press Ctrl+C to stop.")
        # try:
        for i in range(3):
            for action_np in self.actions:
                loop_start = time.perf_counter()

                action = torch.from_numpy(action_np).to(self.device).unsqueeze(0)

                # control
                target = self.prev_target + self.action_scale * action
                self.cur_target = torch.clip(target, self.allegro_dof_lower, self.allegro_dof_upper)
                self.prev_target = self.cur_target.clone()

                # 명령 전송
                cmd = self.cur_target.cpu().numpy()[0]
                ros1 = self._action_hora2allegro(cmd)
                ros2 = self._reorder_imrt2timr(ros1)
                self.allegro.command_joint_position(ros2)

                # 관측 및 데이터 저장
                q_pos_ros2 = self.allegro.poll_joint_position(wait=True, timeout=0.2)
                if q_pos_ros2 is not None:
                    ros1_q_meas = self._reorder_timr2imrt(q_pos_ros2)
                    hora_q_meas = self._obs_allegro2hora(ros1_q_meas)

                    self.history_command_pos[timestep] = self.cur_target.cpu().numpy()[0]
                    self.history_measured_pos[timestep] = hora_q_meas
                else:
                    # if measurement fails, store NaN
                    self.history_command_pos[timestep] = self.cur_target.cpu().numpy()[0]
                    self.history_measured_pos[timestep] = np.nan

                # Aim for 20Hz loop
                time.sleep(max(0, 0.05 - (time.perf_counter() - loop_start)))

                freq = 1.0 / (time.perf_counter() - loop_start)
                print(f"Hz={freq:.2f}, Step={timestep+1}/{len(self.actions)}")
                timestep += 1

        # except KeyboardInterrupt:
        #     print("KeyboardInterrupt, stopping...")
        # finally:
        #     # 안전 정지
        #     try:
        #         self.allegro.go_safe()
        #     except Exception:
        #         pass
        #     stop_allegro_io(self.allegro)
        #     print("🧠 Deployment stopped cleanly.")
        #     # 저장 및 플로팅
        #     self._save_and_plot_error()

    def restore(self, fn):
        # This method is not used in test script
        pass

if __name__ == '__main__':
    # For direct execution, we can pass a dummy config
    player = HardwarePlayerTest(config={})
    player.deploy()
