"""UR5 forward / inverse kinematics from standard DH parameters.

Frame convention
----------------
* DH frame i:  T_i = Rz(q_i) · Tz(d_i) · Tx(a_i) · Rx(alpha_i)
* Flange frame = DH frame 6 (z axis points out of the tool flange).
* TCP frame    = flange · Tz(tcp_offset)  (centre between the RG2 fingertips).
  With the gripper pointing down the TCP z axis is world -z; the TCP x axis is
  the finger closing axis.

The same functions are used by the simple physics backend, by the Isaac Sim
backend (to turn Cartesian RL actions into joint targets) and by the URDF
generator, so all three agree on the robot geometry.
"""

from __future__ import annotations

import numpy as np


def rot_x(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def make_T(R, p):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def down_rotation(yaw):
    """TCP orientation pointing straight down with closing axis at angle `yaw`."""
    return rot_z(yaw) @ rot_x(np.pi)


def rotation_error(R_target, R_current):
    """Axis-angle vector of R_target · R_current^T (world frame)."""
    R = R_target @ R_current.T
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    s = np.sin(angle)
    if s < 1e-6:  # angle close to pi
        w, V = np.linalg.eigh((R + np.eye(3)) / 2.0)
        axis = V[:, np.argmax(w)]
        return axis * angle
    return v / (2.0 * s) * angle


class UR5Kinematics:
    def __init__(self, robot_cfg, tcp_offset: float):
        dh = robot_cfg["dh"]
        self.d = np.asarray(dh["d"], dtype=float)
        self.a = np.asarray(dh["a"], dtype=float)
        self.alpha = np.asarray(dh["alpha"], dtype=float)
        self.limits = np.asarray(robot_cfg["joint_limits"], dtype=float)
        self.base = make_T(np.eye(3), np.asarray(robot_cfg["base_position"], dtype=float))
        self.tcp_offset = float(tcp_offset)
        self.T_flange_tcp = make_T(np.eye(3), [0.0, 0.0, self.tcp_offset])
        self.max_iters = int(robot_cfg.get("ik_max_iters", 60))
        self.tol_p = float(robot_cfg.get("ik_tol_pos", 5e-4))
        self.tol_r = float(robot_cfg.get("ik_tol_rot", 2e-3))
        # fixed part of each DH transform: Tz(d)·Tx(a)·Rx(alpha)
        # = translation (a, 0, d) followed by the rotation Rx(alpha)
        self._fixed = [make_T(rot_x(self.alpha[i]), [self.a[i], 0.0, self.d[i]])
                       for i in range(6)]

    # ------------------------------------------------------------------ FK
    def frames(self, q):
        """List of 4x4 world transforms: [base, frame1, ..., frame6(flange)]."""
        T = self.base.copy()
        out = [T]
        for i in range(6):
            c, s = np.cos(q[i]), np.sin(q[i])
            Rz = np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])
            T = T @ Rz @ self._fixed[i]
            out.append(T)
        return out

    def flange(self, q):
        return self.frames(q)[-1]

    def tcp(self, q):
        return self.flange(q) @ self.T_flange_tcp

    # ------------------------------------------------------------ Jacobian
    def jacobian_tcp(self, q):
        """6x6 geometric Jacobian of the TCP (linear; angular) in world frame."""
        F = self.frames(q)
        p_tcp = (F[-1] @ self.T_flange_tcp)[:3, 3]
        J = np.zeros((6, 6))
        for i in range(6):
            z = F[i][:3, 2]          # joint i+1 rotates about z of frame i
            p = F[i][:3, 3]
            J[:3, i] = np.cross(z, p_tcp - p)
            J[3:, i] = z
        return J

    # ------------------------------------------------------------------ IK
    def ik(self, p_target, R_target, q_init, damping=0.02):
        """Damped least squares IK for the TCP. Returns (q, success)."""
        q = np.array(q_init, dtype=float)
        for _ in range(self.max_iters):
            T = self.tcp(q)
            e_p = np.asarray(p_target) - T[:3, 3]
            e_r = rotation_error(R_target, T[:3, :3])
            if np.linalg.norm(e_p) < self.tol_p and np.linalg.norm(e_r) < self.tol_r:
                ok = np.all(q >= self.limits[:, 0]) and np.all(q <= self.limits[:, 1])
                return q, bool(ok)
            J = self.jacobian_tcp(q)
            e = np.concatenate([e_p, e_r])
            JJt = J @ J.T + (damping ** 2) * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            n = np.linalg.norm(dq)
            if n > 0.5:
                dq *= 0.5 / n
            q = q + dq
        return q, False

    def tcp_down_pose(self, q):
        """(position, yaw) of the TCP assuming it points down."""
        T = self.tcp(q)
        x_axis = T[:3, 0]
        return T[:3, 3].copy(), float(np.arctan2(x_axis[1], x_axis[0]))
