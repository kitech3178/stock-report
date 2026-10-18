"""Gymnasium environment: UR5 + RG2 top-down grasping from a wrist depth camera.

One episode
    reset : new object pose on the table, gripper open at `task.start_height`
    step  : a discrete action (move the hand a little, rotate, or 'grasp')
    end   : 'grasp' (close -> lift -> success / fail), or `task.max_steps`

The episode is identical for Q-learning and DQN; only the observation encoding
(`observation` = discrete | features | depth) differs.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from grasp_rl.backends.base import make_backend
from grasp_rl.envs.actions import ActionSet
from grasp_rl.envs.reward import RewardFunction
from grasp_rl.envs.state import StateEncoder
from grasp_rl.perception import ground_truth_features, perceive


class GraspEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, cfg, observation="discrete", backend=None):
        super().__init__()
        self.cfg = cfg
        self.backend = backend or make_backend(cfg)
        self.actions = ActionSet(cfg.actions)
        self.encoder = StateEncoder(cfg, observation)
        self.reward_fn = RewardFunction(cfg)
        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = self.encoder.space()
        self.max_steps = int(cfg.task.max_steps)
        cx, cy = cfg.object.spawn_center
        hx, hy = cfg.task.workspace_half_extent
        self.ws_lo = np.array([cx - hx, cy - hy])
        self.ws_hi = np.array([cx + hx, cy + hy])
        self.max_h = float(cfg.task.max_tcp_height)
        self._rng = np.random.default_rng(cfg.experiment.seed)
        self.last = {}

    # ------------------------------------------------------------------
    def _observe(self):
        pos, yaw = self.backend.tcp_pose()
        depth = self.backend.depth_image()
        T_cam = self.backend.T_world_cam()
        perceived = perceive(depth, self.backend.camera, T_cam, pos, yaw, self.cfg.camera)
        gt = ground_truth_features(self.backend.object_state(), pos, yaw)
        f = gt if self.cfg.state.source == "ground_truth" else perceived
        obs = self.encoder.encode(f, depth, pos[2], yaw)
        shaping_f = gt if self.cfg.reward.shaping.use_ground_truth else perceived
        self.last = dict(depth=depth, pos=pos, yaw=yaw, features=f, perceived=perceived,
                         gt=gt, phi=self.reward_fn.potential(shaping_f))
        return obs

    def _info(self, **extra):
        f = self.last["features"]
        info = {"step": self.steps, "tcp": self.last["pos"].copy(), "yaw": self.last["yaw"],
                "features": f, "gt": self.last["gt"], "success": False}
        if self.encoder.mode != "discrete":
            info["state_index"] = self.encoder.discrete_index(f)
        info.update(extra)
        return info

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.backend.reset(self._rng)
        self.steps = 0
        obs = self._observe()
        return obs, self._info()

    def step(self, action):
        name = self.actions.names[int(action)]
        phi_prev = self.last["phi"]
        events, grasp = [], None
        terminated = False
        if name == "grasp":
            grasp = self.backend.grasp()
            if grasp.success:
                events.append("grasp_success")
                terminated = True
            else:
                events.append("grasp_fail")
                terminated = bool(self.cfg.reward.terminate_on_grasp_fail)
        else:
            pos, yaw = self.backend.tcp_pose()
            tgt, tyaw = self.actions.target(name, pos, yaw)
            if np.any(tgt[:2] < self.ws_lo) or np.any(tgt[:2] > self.ws_hi) or tgt[2] > self.max_h:
                events.append("out_of_workspace")
            else:
                res = self.backend.move_tcp(tgt, tyaw)
                if res.status.startswith("collision"):
                    events += ["collision", res.status]
                    terminated = bool(self.cfg.reward.terminate_on_collision)
                elif res.status == "ik_fail":
                    events.append("ik_fail")
        self.steps += 1
        obs = self._observe()
        truncated = (not terminated) and self.steps >= self.max_steps
        reward, comp = self.reward_fn(events, phi_prev, self.last["phi"], terminated)
        info = self._info(action_name=name, events=events, reward_terms=comp,
                          success="grasp_success" in events)
        if grasp is not None:
            info["grasp_reason"] = grasp.reason
            info["grasp_info"] = grasp.info
        return obs, float(reward), terminated, truncated, info

    def close(self):
        self.backend.close()
