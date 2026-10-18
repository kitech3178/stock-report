"""STATE definition: how a perceived scene becomes the agent's observation.

    discrete : bin the grasp features -> one integer (Q-table row)
    features : continuous, normalised feature vector (DQN with an MLP)
    depth    : the raw depth image + proprioception (DQN with a CNN)
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces


class StateEncoder:
    def __init__(self, cfg, mode):
        self.mode = mode
        b = cfg.state.bins
        self.edges = [np.asarray(b.dx), np.asarray(b.dy), np.asarray(b.dz),
                      np.deg2rad(np.asarray(b.yaw))]
        self.dims = [len(e) + 1 for e in self.edges]
        self.n_discrete = int(np.prod(self.dims)) + 1           # +1: object not visible
        self.scale = np.asarray(cfg.state.feature_scale, dtype=np.float32)
        self.far = float(cfg.camera.far)
        self.H, self.W = int(cfg.camera.height), int(cfg.camera.width)
        self.height_scale = float(cfg.task.max_tcp_height)

    # ------------------------------------------------------------------
    def space(self):
        if self.mode == "discrete":
            return spaces.Discrete(self.n_discrete)
        if self.mode == "features":
            return spaces.Box(-np.inf, np.inf, shape=(len(self.scale),), dtype=np.float32)
        if self.mode == "depth":
            return spaces.Dict({
                "depth": spaces.Box(0.0, 1.0, shape=(1, self.H, self.W), dtype=np.float32),
                "proprio": spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            })
        raise ValueError(f"unknown observation mode {self.mode!r}")

    def discrete_index(self, f):
        if not f.visible:
            return self.n_discrete - 1
        idx = [int(np.digitize(v, e)) for v, e in zip((f.dx, f.dy, f.dz, f.yaw), self.edges)]
        return int(np.ravel_multi_index(idx, self.dims))

    def decode(self, index):
        """Bin indices (dx, dy, dz, yaw) of a discrete state, None if not visible."""
        if index == self.n_discrete - 1:
            return None
        return np.unravel_index(index, self.dims)

    def encode(self, f, depth, tip_z, yaw):
        if self.mode == "discrete":
            return self.discrete_index(f)
        if self.mode == "features":
            return f.vector(tip_z) / self.scale
        return {
            "depth": (np.clip(depth, 0.0, self.far) / self.far)[None].astype(np.float32),
            "proprio": np.array([tip_z / self.height_scale, np.sin(yaw), np.cos(yaw)],
                                dtype=np.float32),
        }
