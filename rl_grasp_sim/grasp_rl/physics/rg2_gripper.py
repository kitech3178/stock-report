"""OnRobot RG2 parallel gripper: collision geometry, closing and lift physics.

The model is quasi-static for the squeeze and a 1-D rigid-body simulation for
the lift:

* Coulomb friction at the two finger pads. The object stays in the hand while
      m (g + a_lift) <= 2 · μs · F_n                       (linear slip)
  and while the gravity torque about the grasp line is resisted by torsional
  friction of the two contact patches (radius ρ):
      m (g + a_lift) · r_com <= 2 · (2/3) · μs · F_n · ρ     (rotational slip)
* During the lift the gripper follows a trapezoidal velocity profile. When the
  static limit is exceeded the object slides with kinetic friction μk; it is
  dropped once the accumulated slip exceeds the pad contact length.
* The parallel jaws centre the object along the closing axis and rotate it
  into alignment when the yaw error is below `self_align_max_deg`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from grasp_rl.physics.objects import TableObject, wrap_half_pi

G = 9.81
TABLE_CLEARANCE = 0.002


@dataclass
class GraspResult:
    success: bool
    reason: str
    info: dict = field(default_factory=dict)


class RG2Gripper:
    def __init__(self, gcfg):
        self.cfg = gcfg
        self.stroke = float(gcfg["stroke"])
        self.open_width = min(float(gcfg["open_width"]), self.stroke)
        self.force = float(np.clip(gcfg["force"], 3.0, 40.0))
        self.t = float(gcfg["finger_thickness"])
        self.w = float(gcfg["finger_width"])
        self.pad = float(gcfg["pad_length"])
        self.finger_len = float(gcfg["finger_length"])
        self.body = np.asarray(gcfg["body_footprint"], dtype=float)
        self.mu_pad = float(gcfg["friction"])
        self.rho = float(gcfg["contact_patch_radius"])
        self.min_contact = float(gcfg["min_contact_length"])
        self.self_align = np.deg2rad(float(gcfg["self_align_max_deg"]))
        vmin, vmax = gcfg["speed_range"]
        # RG2: higher force setting -> slower closing
        self.speed = vmax - (self.force - 3.0) / 37.0 * (vmax - vmin)

    # ------------------------------------------------------------ geometry
    def finger_rects(self, tcp_xy, yaw, width):
        """Two finger footprints (center, yaw, half extents) for opening `width`."""
        u = np.array([np.cos(yaw), np.sin(yaw)])
        off = width / 2.0 + self.t / 2.0
        half = np.array([self.t / 2.0, self.w / 2.0])
        return [(tcp_xy + u * off, yaw, half), (tcp_xy - u * off, yaw, half)]

    def check_collision(self, obj: TableObject, tcp_pos, yaw, width=None):
        """None if the pose is free, else 'table' or 'object'."""
        width = self.open_width if width is None else width
        tip_z = tcp_pos[2]
        if tip_z < TABLE_CLEARANCE:
            return "table"
        if tip_z < obj.top_z:
            for c, yw, half in self.finger_rects(tcp_pos[:2], yaw, width):
                if obj.overlaps_rect(c, yw, half):
                    return "object"
        if tip_z + self.finger_len < obj.top_z:
            if obj.overlaps_rect(tcp_pos[:2], yaw, self.body / 2.0):
                return "object"
        return None

    # ------------------------------------------------------- grasp physics
    def grasp_and_lift(self, obj: TableObject, tcp_pos, yaw, task_cfg) -> GraspResult:
        tip_z = float(tcp_pos[2])
        # combined pad/object friction (PhysX default combine mode = average)
        mu_s = 0.5 * (obj.mu_s + self.mu_pad)
        mu_k = mu_s * (obj.mu_k / obj.mu_s)
        info = {"mu_s": mu_s, "force": self.force, "mass": obj.mass}

        # 1) vertical contact overlap of the pads with the object side
        contact_len = min(obj.top_z, tip_z + self.pad) - max(tip_z, 0.0)
        info["contact_length"] = contact_len
        if contact_len < self.min_contact:
            return GraspResult(False, "no_contact_height", info)

        # 2) object pose in the gripper frame (u: closing axis, v: perpendicular)
        u = np.array([np.cos(yaw), np.sin(yaw)])
        v = np.array([-np.sin(yaw), np.cos(yaw)])
        d = np.array([obj.x, obj.y]) - np.asarray(tcp_pos[:2])
        dx, dy = float(d @ u), float(d @ v)
        info.update(dx=dx, dy=dy)

        if obj.shape == "cylinder":
            width, other, delta = 2 * obj.radius, 2 * obj.radius, 0.0
            if abs(dy) > obj.radius - 0.002:
                return GraspResult(False, "miss", info)
            r_com = 0.0              # round object rolls/slides onto the jaw line
        else:
            cands = []
            for k, ang in ((0, obj.yaw), (1, obj.yaw + np.pi / 2)):
                cands.append((abs(wrap_half_pi(ang - yaw)), float(obj.size[k]),
                              float(obj.size[1 - k])))
            fitting = [c for c in cands if c[1] <= self.open_width - 0.002]
            if not fitting:
                return GraspResult(False, "too_wide", info)
            delta, width, other = min(fitting)
            if delta > self.self_align:
                info["yaw_error_deg"] = np.rad2deg(delta)
                return GraspResult(False, "misaligned", info)
            if abs(dy) > other / 2 + self.w / 2 - 0.004:
                return GraspResult(False, "miss", info)
            r_com = abs(dy)          # flat faces: no centring perpendicular to the jaws
        if abs(dx) + width / 2 > self.open_width / 2 + 1e-4:
            return GraspResult(False, "miss", info)
        info.update(width=width, yaw_error_deg=float(np.rad2deg(delta)), r_com=r_com,
                    close_time=(self.open_width - width) / self.speed)

        # 3) friction capacities
        F_lin = 2.0 * mu_s * self.force
        T_rot = 2.0 * (2.0 / 3.0) * mu_s * self.force * self.rho
        info.update(friction_capacity=F_lin, torque_capacity=T_rot)

        # 4) lift: trapezoidal profile, 1 ms integration
        h = float(task_cfg["lift_height"])
        vmax = float(task_cfg["lift_max_speed"])
        amax = float(task_cfg["lift_accel"])
        dt = 1e-3
        zg = vg = 0.0          # gripper height / velocity (relative to start)
        zo = vo = 0.0          # object
        slip = 0.0
        stuck = True
        t = 0.0
        max_need = 0.0
        while zg < h - 1e-6 and t < 10.0:
            # trapezoidal velocity profile
            dist_left = h - zg
            v_brake = np.sqrt(2 * amax * dist_left)
            v_cmd = min(vmax, v_brake, vg + amax * dt)
            ag = (v_cmd - vg) / dt
            need = obj.mass * (G + ag)
            max_need = max(max_need, need)
            if obj.mass * (G + ag) * r_com > T_rot:
                info.update(max_required_force=max_need, lift_time=t)
                return GraspResult(False, "rotational_slip", info)
            if stuck and need <= F_lin:
                ao = ag
            else:
                stuck = False
                # kinetic friction drags the object up, gravity pulls it down
                ao = (2.0 * mu_k * self.force * np.sign(vg - vo + 1e-12)) / obj.mass - G
                if vo + ao * dt >= vg + ag * dt and need <= F_lin:
                    stuck = True
                    ao = ag
            vg += ag * dt
            vo += ao * dt
            zg += vg * dt
            zo += vo * dt
            if zo < 0:            # resting on the table
                zo, vo = 0.0, max(vo, 0.0)
            slip = zg - zo
            if slip > contact_len:
                info.update(max_required_force=max_need, slip=slip, lift_time=t)
                return GraspResult(False, "dropped", info)
            t += dt
        info.update(max_required_force=max_need, slip=slip, lift_time=t,
                    object_lift=zo)
        if zo < 0.8 * h:
            return GraspResult(False, "dropped", info)
        return GraspResult(True, "success", info)
