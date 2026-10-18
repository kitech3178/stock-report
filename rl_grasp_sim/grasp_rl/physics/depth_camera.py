"""Wrist-mounted pinhole depth camera (RealSense D405-like) rendered by ray casting.

Camera frame follows the ROS optical convention: x right, y down, z forward
(the viewing direction). Depth values are z-depth (distance to the image
plane) in metres, 0 = invalid, exactly like RealSense / Isaac Sim
`distance_to_image_plane`.
"""

from __future__ import annotations

import numpy as np

from grasp_rl.physics.ur5_kinematics import make_T, rot_x


class DepthCamera:
    def __init__(self, ccfg):
        self.cfg = ccfg
        self.W = int(ccfg["width"])
        self.H = int(ccfg["height"])
        hfov = np.deg2rad(float(ccfg["hfov_deg"]))
        self.fx = self.W / 2.0 / np.tan(hfov / 2.0)
        self.fy = self.fx
        self.cx = (self.W - 1) / 2.0
        self.cy = (self.H - 1) / 2.0
        self.near = float(ccfg["near"])
        self.far = float(ccfg["far"])
        u, v = np.meshgrid(np.arange(self.W), np.arange(self.H))
        # direction with unit z component -> ray parameter t == z-depth
        self.rays_cam = np.stack([(u - self.cx) / self.fx, (v - self.cy) / self.fy,
                                  np.ones_like(u, dtype=float)], -1).reshape(-1, 3)
        self.T_tcp_cam = make_T(rot_x(np.deg2rad(float(ccfg["tilt_deg"]))),
                                np.asarray(ccfg["mount_xyz"], dtype=float))
        n = ccfg["noise"]
        self.noise_base = float(n["base_std"])
        self.noise_quad = float(n["quad_coeff"])
        self.dropout = float(n["dropout"])
        self.quant = float(n["quantization"])

    @property
    def K(self):
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1.0]])

    def world_pose(self, T_world_tcp):
        return T_world_tcp @ self.T_tcp_cam

    def render(self, T_world_cam, objects, rng=None, table_z=0.0, noise=True):
        """Ray cast the table plane and objects. Returns (H, W) float32 depth."""
        R, o = T_world_cam[:3, :3], T_world_cam[:3, 3]
        dirs = self.rays_cam @ R.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (table_z - o[2]) / dirs[:, 2]
        t = np.where(t > 0, t, np.inf)
        for obj in objects:
            t = np.minimum(t, obj.ray_cast(o, dirs))
        depth = t                                    # z-depth thanks to unit-z rays
        if noise and rng is not None:
            sigma = self.noise_base + self.noise_quad * depth ** 2
            depth = depth + rng.normal(size=depth.shape) * np.where(np.isfinite(sigma), sigma, 0)
            if self.dropout > 0:
                depth = np.where(rng.random(depth.shape) < self.dropout, 0.0, depth)
        if self.quant > 0:
            depth = np.round(depth / self.quant) * self.quant
        valid = np.isfinite(depth) & (depth >= self.near) & (depth <= self.far)
        return np.where(valid, depth, 0.0).reshape(self.H, self.W).astype(np.float32)

    def backproject(self, depth, T_world_cam):
        """Valid pixels -> (N, 3) world points and their (row, col) indices."""
        flat = depth.reshape(-1)
        idx = np.nonzero(flat > 0)[0]
        pts_cam = self.rays_cam[idx] * flat[idx, None]
        pts = pts_cam @ T_world_cam[:3, :3].T + T_world_cam[:3, 3]
        return pts, np.stack(np.unravel_index(idx, depth.shape), -1)
