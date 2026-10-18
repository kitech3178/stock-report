"""Depth image -> grasp features (the agent's view of the world).

Features are expressed in the gripper's horizontal frame
(u = finger closing axis, v = perpendicular), so they do not depend on where
the object is on the table, only on where it is relative to the hand:

    dx, dy   object centre offset from the TCP           [m]
    dz       fingertip height above the object top        [m]
    yaw      angle from closing axis to the object's narrow side, (-90, 90] deg
    visible  object segmented in the image
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from grasp_rl.physics.objects import wrap_half_pi


@dataclass
class Features:
    visible: bool
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    yaw: float = 0.0          # rad
    n_pixels: int = 0

    def vector(self, tip_z):
        """Continuous vector used by DQN (features mode)."""
        return np.array([self.dx, self.dy, self.dz, np.sin(2 * self.yaw), np.cos(2 * self.yaw),
                         float(self.visible), tip_z], dtype=np.float32)


def _to_hand(xy, tcp_xy, yaw):
    u = np.array([np.cos(yaw), np.sin(yaw)])
    v = np.array([-np.sin(yaw), np.cos(yaw)])
    d = np.asarray(xy) - np.asarray(tcp_xy)
    return d @ u, d @ v


def ground_truth_features(obj, tcp_pos, tcp_yaw) -> Features:
    dx, dy = _to_hand([obj.x, obj.y], tcp_pos[:2], tcp_yaw)
    yaw = 0.0 if obj.shape == "cylinder" else wrap_half_pi(obj.short_axis_angle() - tcp_yaw)
    return Features(True, float(dx), float(dy), float(tcp_pos[2] - obj.top_z), float(yaw))


def perceive(depth, camera, T_world_cam, tcp_pos, tcp_yaw, ccfg, table_z=0.0) -> Features:
    """Segment the object from the table using the known camera pose
    (robot forward kinematics + hand-eye calibration) and estimate features."""
    pts, _ = camera.backproject(depth, T_world_cam)
    obj_pts = pts[pts[:, 2] > table_z + float(ccfg["seg_threshold"])]
    if len(obj_pts) < int(ccfg["min_pixels"]):
        return Features(False, n_pixels=len(obj_pts))
    top_z = float(np.percentile(obj_pts[:, 2], 90))
    top = obj_pts[obj_pts[:, 2] > top_z - 0.008]
    if len(top) < 3:
        top = obj_pts
    hx, hy = _to_hand(top[:, :2], tcp_pos[:2], tcp_yaw)
    dx, dy = float(np.mean(hx)), float(np.mean(hy))
    yaw = 0.0
    if len(top) >= 5:
        cov = np.cov(np.stack([hx, hy]))
        w, V = np.linalg.eigh(cov)                 # ascending eigenvalues
        if w[1] > 1.4 * max(w[0], 1e-12):          # elongated -> orientation is defined
            minor = V[:, 0]                        # narrow side direction (hand frame)
            yaw = float(wrap_half_pi(np.arctan2(minor[1], minor[0])))
    return Features(True, dx, dy, float(tcp_pos[2] - top_z), yaw, len(obj_pts))
