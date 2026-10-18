"""ACTION definition: discrete Cartesian moves of the gripper + 'grasp'."""

from __future__ import annotations

import numpy as np

KNOWN = {"x+", "x-", "y+", "y-", "z+", "z-", "yaw+", "yaw-", "grasp"}


class ActionSet:
    def __init__(self, acfg):
        self.names = list(acfg["set"])
        unknown = set(self.names) - KNOWN
        if unknown:
            raise ValueError(f"unknown actions {unknown}; choose from {sorted(KNOWN)}")
        if "grasp" not in self.names:
            raise ValueError("the action set must contain 'grasp'")
        self.frame = acfg["frame"]
        self.dxy = float(acfg["step_xy"])
        self.dz = float(acfg["step_z"])
        self.dyaw = np.deg2rad(float(acfg["step_yaw_deg"]))
        self.yaw_limit = np.deg2rad(float(acfg["yaw_limit_deg"]))

    def __len__(self):
        return len(self.names)

    def target(self, name, pos, yaw):
        """New (pos, yaw) for a motion action, given the current TCP pose."""
        pos = np.array(pos, dtype=float)
        if self.frame == "tool":
            u = np.array([np.cos(yaw), np.sin(yaw), 0.0])
            v = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
        else:
            u, v = np.array([1.0, 0, 0]), np.array([0, 1.0, 0])
        step = {
            "x+": self.dxy * u, "x-": -self.dxy * u,
            "y+": self.dxy * v, "y-": -self.dxy * v,
            "z+": np.array([0, 0, self.dz]), "z-": np.array([0, 0, -self.dz]),
        }
        if name in step:
            return pos + step[name], yaw
        if name == "yaw+":
            return pos, min(yaw + self.dyaw, self.yaw_limit)
        if name == "yaw-":
            return pos, max(yaw - self.dyaw, -self.yaw_limit)
        raise ValueError(name)
