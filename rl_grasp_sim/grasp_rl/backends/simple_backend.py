"""Built-in physics backend: UR5 kinematics + RG2 contact/friction model +
ray-cast depth camera, all in numpy (~1 ms per step)."""

from __future__ import annotations

import numpy as np

from grasp_rl.backends.base import Backend, MoveResult
from grasp_rl.physics.objects import sample_object


class SimpleBackend(Backend):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.q = np.array(cfg.robot.ik_seed, dtype=float)
        self.obj = None
        self.noise_std = float(cfg.robot.execution_noise_std)

    def reset(self, rng):
        self.rng = rng
        self.obj = sample_object(self.cfg.object, rng)
        pos, yaw = self.home_tcp(rng)
        q, ok = self.solve_ik(pos, yaw, self.cfg.robot.ik_seed)
        if not ok:
            raise RuntimeError("home pose is not reachable; check task.start_height / spawn_center")
        self.q = q

    def joint_positions(self):
        return self.q

    def move_tcp(self, pos, yaw):
        pos = np.array(pos, dtype=float)
        if self.noise_std > 0:
            pos = pos + self.rng.normal(0.0, self.noise_std, 3) * np.array([1, 1, 0.5])
        hit = self.gripper.check_collision(self.obj, pos, yaw)
        if hit:
            return MoveResult(f"collision_{hit}")
        q, ok = self.solve_ik(pos, yaw, self.q)
        if not ok:
            return MoveResult("ik_fail")
        self.q = q
        return MoveResult("ok")

    def grasp(self):
        pos, yaw = self.tcp_pose()
        return self.gripper.grasp_and_lift(self.obj, pos, yaw, self.cfg.task)

    def depth_image(self):
        return self.camera.render(self.T_world_cam(), [self.obj], self.rng)

    def object_state(self):
        return self.obj
