"""Baselines: a hand-written controller (uses ground-truth features) and random."""

import numpy as np


class ScriptedAgent:
    """Rotate -> centre -> descend -> grasp. Shows the task is solvable and gives
    an upper reference line for the learning curves."""

    def __init__(self, cfg, env):
        self.env = env
        self.names = env.actions.names
        self.target_depth = float(cfg.task.target_grasp_depth)
        self.tol_xy = 0.5 * float(cfg.actions.step_xy) + 0.001
        self.tol_yaw = np.deg2rad(0.5 * float(cfg.actions.step_yaw_deg) + 1.0)
        self.dz = float(cfg.actions.step_z)

    def _a(self, name):
        return self.names.index(name)

    def act(self, obs, explore=False):
        f = self.env.last["gt"]
        yaw = self.env.last["yaw"]
        if abs(f.yaw) > self.tol_yaw and f.dz > 0:
            want = "yaw+" if f.yaw > 0 else "yaw-"
            lim = self.env.actions.yaw_limit - 1e-3
            if (want == "yaw+" and yaw >= lim) or (want == "yaw-" and yaw <= -lim):
                want = "yaw-" if want == "yaw+" else "yaw+"   # go the other way round
            return self._a(want)
        if abs(f.dx) > self.tol_xy and f.dz > 0:
            return self._a("x+" if f.dx > 0 else "x-")
        if abs(f.dy) > self.tol_xy and f.dz > 0:
            return self._a("y+" if f.dy > 0 else "y-")
        if f.dz > -self.target_depth + self.dz / 2:
            return self._a("z-")
        return self._a("grasp")

    def observe(self, *a):
        return {}

    def end_episode(self):
        pass


class RandomAgent:
    def __init__(self, n_actions, seed=0):
        self.n = n_actions
        self.rng = np.random.default_rng(seed)

    def act(self, obs, explore=True):
        return int(self.rng.integers(self.n))

    def observe(self, *a):
        return {}

    def end_episode(self):
        pass
