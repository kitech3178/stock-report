"""Simulator backend interface.

The MDP (actions, state, reward) lives in `grasp_rl.envs` and only talks to a
backend through this interface, so the same agent / experiment runs either on
the built-in physics model or inside Isaac Sim.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from grasp_rl.physics.depth_camera import DepthCamera
from grasp_rl.physics.rg2_gripper import RG2Gripper
from grasp_rl.physics.ur5_kinematics import UR5Kinematics, down_rotation


@dataclass
class MoveResult:
    status: str          # ok | ik_fail | collision_table | collision_object


class Backend(ABC):
    def __init__(self, cfg):
        self.cfg = cfg
        self.kin = UR5Kinematics(cfg.robot, cfg.gripper.tcp_offset)
        self.gripper = RG2Gripper(cfg.gripper)
        self.camera = DepthCamera(cfg.camera)
        self.rng = np.random.default_rng(cfg.experiment.seed)

    # -- helpers shared by all backends ---------------------------------
    def home_tcp(self, rng):
        t = self.cfg.task
        cx, cy = self.cfg.object.spawn_center
        n = float(t.start_xy_noise)
        pos = np.array([cx + rng.uniform(-n, n), cy + rng.uniform(-n, n), t.start_height])
        return pos, 0.0

    def solve_ik(self, pos, yaw, q_init):
        return self.kin.ik(pos, down_rotation(yaw), q_init)

    def T_world_cam(self):
        return self.camera.world_pose(self.kin.tcp(self.joint_positions()))

    def tcp_pose(self):
        return self.kin.tcp_down_pose(self.joint_positions())

    # -- to implement ----------------------------------------------------
    @abstractmethod
    def reset(self, rng):
        """Place a new object, move the robot home with the gripper open."""

    @abstractmethod
    def joint_positions(self) -> np.ndarray:
        """Current 6 arm joint angles."""

    @abstractmethod
    def move_tcp(self, pos, yaw) -> MoveResult:
        """Move the TCP (pointing down) to `pos` with closing-axis angle `yaw`."""

    @abstractmethod
    def grasp(self):
        """Close the gripper, lift, and return a GraspResult."""

    @abstractmethod
    def depth_image(self) -> np.ndarray:
        """(H, W) float32 depth in metres from the wrist camera, 0 = invalid."""

    @abstractmethod
    def object_state(self):
        """Ground-truth TableObject (pose, size, mass, friction)."""

    def close(self):
        pass


def make_backend(cfg) -> Backend:
    if cfg.backend == "simple":
        from grasp_rl.backends.simple_backend import SimpleBackend
        return SimpleBackend(cfg)
    if cfg.backend == "isaac":
        from grasp_rl.backends.isaac_backend import IsaacBackend
        return IsaacBackend(cfg)
    raise ValueError(f"unknown backend {cfg.backend!r} (simple | isaac)")
