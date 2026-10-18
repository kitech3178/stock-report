"""NVIDIA Isaac Sim backend (PhysX articulation + RTX depth camera).

Runs the *same* MDP as the simple backend:
  * the robot is the URDF generated from the config (grasp_rl/urdf.py),
  * RL actions -> TCP target -> the shared UR5 IK -> joint position targets,
  * the RG2 fingers are force-limited prismatic drives (max effort = grip force),
  * the wrist camera is an Isaac `Camera` attached to the TCP link with the
    same intrinsics / mounting as `physics/depth_camera.py`,
  * contacts, friction, slip and dropping come from PhysX instead of the
    analytic model.

Launch with Isaac Sim's python, e.g.
    ~/isaacsim/python.sh scripts/train.py --algo qlearning --backend isaac
Tested API names: Isaac Sim 4.x / 5.x (`isaacsim.*`) with a fallback to the
2023.1 `omni.isaac.*` modules. Asset paths and module names move between
releases; see README section "Isaac Sim" if an import fails.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from grasp_rl.backends.base import Backend, MoveResult
from grasp_rl.config import PROJECT_ROOT
from grasp_rl.physics.objects import sample_object
from grasp_rl.physics.rg2_gripper import GraspResult
from grasp_rl.physics.ur5_kinematics import rot_x
from grasp_rl.urdf import ARM_JOINTS, FINGER_JOINTS, write_urdf


# ----------------------------------------------------------------- compat
def _launch(headless):
    try:
        from isaacsim import SimulationApp
    except ImportError:
        from omni.isaac.kit import SimulationApp
    return SimulationApp({"headless": bool(headless)})


def _modules():
    """Import the Isaac core API (must run *after* SimulationApp is created)."""
    m = SimpleNamespace()
    try:
        from isaacsim.core.api import World
        from isaacsim.core.api.materials import PhysicsMaterial
        from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder, FixedCuboid
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        try:
            from isaacsim.core.prims import SingleArticulation as Articulation
        except ImportError:
            from isaacsim.core.prims import Articulation
    except ImportError:
        from omni.isaac.core import World
        from omni.isaac.core.articulations import Articulation
        from omni.isaac.core.materials import PhysicsMaterial
        from omni.isaac.core.objects import DynamicCuboid, DynamicCylinder, FixedCuboid
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.types import ArticulationAction
        from omni.isaac.sensor import Camera
    m.World, m.Articulation, m.PhysicsMaterial = World, Articulation, PhysicsMaterial
    m.DynamicCuboid, m.DynamicCylinder, m.FixedCuboid = DynamicCuboid, DynamicCylinder, FixedCuboid
    m.ArticulationAction, m.Camera, m.enable_extension = ArticulationAction, Camera, enable_extension
    return m


def _set(obj, name, value):
    """Import configs expose either attributes or set_<name>() depending on version."""
    setter = getattr(obj, f"set_{name}", None)
    if callable(setter):
        setter(value)
    else:
        setattr(obj, name, value)


def _import_urdf(m, urdf_path):
    for ext in ("isaacsim.asset.importer.urdf", "omni.importer.urdf"):
        try:
            m.enable_extension(ext)
            break
        except Exception:  # noqa: BLE001 - extension name differs per release
            continue
    import omni.kit.commands

    _, icfg = omni.kit.commands.execute("URDFCreateImportConfig")
    for k, v in (("merge_fixed_joints", False), ("convex_decomp", False), ("fix_base", True),
                 ("make_default_prim", False), ("self_collision", False),
                 ("create_physics_scene", False), ("import_inertia_tensor", True),
                 ("distance_scale", 1.0)):
        try:
            _set(icfg, k, v)
        except Exception:  # noqa: BLE001
            pass
    _, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile", urdf_path=str(urdf_path), import_config=icfg,
        get_articulation_root=True)
    return prim_path


def _find_prim(root_path, name):
    import omni.usd
    from pxr import Usd

    stage = omni.usd.get_context().get_stage()
    top = "/" + root_path.strip("/").split("/")[0]
    for prim in Usd.PrimRange(stage.GetPrimAtPath(top)):
        if prim.GetName() == name:
            return str(prim.GetPath())
    raise RuntimeError(f"link '{name}' not found under {top}")


def _quat_wxyz(R):
    w = np.sqrt(max(0.0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    x = np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    y = np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    z = np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    x = np.copysign(x, R[2, 1] - R[1, 2])
    y = np.copysign(y, R[0, 2] - R[2, 0])
    z = np.copysign(z, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def _yaw_from_quat(q):
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


# ----------------------------------------------------------------- backend
class IsaacBackend(Backend):
    def __init__(self, cfg):
        super().__init__(cfg)
        ic = cfg.isaac
        self.ic = ic
        self.app = _launch(ic.headless)
        m = self.m = _modules()

        urdf = Path(ic.urdf_path)
        urdf = urdf if urdf.is_absolute() else PROJECT_ROOT / urdf
        if ic.regenerate_urdf or not urdf.exists():
            urdf.parent.mkdir(parents=True, exist_ok=True)
            write_urdf(cfg, urdf)

        self.world = m.World(stage_units_in_meters=1.0, physics_dt=float(ic.physics_dt),
                             rendering_dt=float(ic.rendering_dt))
        self.world.scene.add_default_ground_plane(z_position=-0.75)
        cx, cy = cfg.object.spawn_center
        self.world.scene.add(m.FixedCuboid(
            prim_path="/World/table", name="table", position=np.array([cx * 0.6, cy, -0.025]),
            scale=np.array([1.4, 1.2, 0.05]), color=np.array([0.55, 0.45, 0.35])))

        root = _import_urdf(m, urdf)
        self.robot = self.world.scene.add(m.Articulation(prim_path=root, name="ur5_rg2"))

        o = cfg.object
        size = np.array(o.size, dtype=float)
        kw = dict(prim_path="/World/target", name="target", mass=float(o.mass),
                  position=np.array([cx, cy, size[2] / 2 + 0.002]), color=np.array([0.9, 0.3, 0.2]))
        if o.shape == "cylinder":
            self.obj_prim = self.world.scene.add(m.DynamicCylinder(
                radius=size[0] / 2, height=size[2], **kw))
        else:
            self.obj_prim = self.world.scene.add(m.DynamicCuboid(size=1.0, scale=size, **kw))
        self.obj_mat = m.PhysicsMaterial(prim_path="/World/materials/object",
                                         static_friction=float(o.friction),
                                         dynamic_friction=float(o.friction) * float(o.kinetic_ratio),
                                         restitution=0.0)
        self.obj_prim.apply_physics_material(self.obj_mat)

        # wrist camera attached to the TCP link, ROS optical axes
        tcp_path = _find_prim(root, "tcp")
        self._bind_finger_material(root)
        c = cfg.camera
        self.cam = m.Camera(prim_path=f"{tcp_path}/wrist_depth_camera",
                            resolution=(int(c.width), int(c.height)))
        self.world.reset()
        self.cam.initialize()
        self.cam.add_distance_to_image_plane_to_frame()
        self.cam.set_local_pose(translation=np.array(c.mount_xyz, dtype=float),
                                orientation=_quat_wxyz(rot_x(np.deg2rad(float(c.tilt_deg)))),
                                camera_axes="ros")
        self.cam.set_clipping_range(float(c.near) * 0.5, float(c.far) * 2)
        f = 1.0
        ap = 2 * f * np.tan(np.deg2rad(float(c.hfov_deg)) / 2)
        self.cam.set_focal_length(f)
        self.cam.set_horizontal_aperture(ap)
        self.cam.set_vertical_aperture(ap * int(c.height) / int(c.width))

        self.arm_idx = np.array([self.robot.get_dof_index(n) for n in ARM_JOINTS])
        self.fin_idx = np.array([self.robot.get_dof_index(n) for n in FINGER_JOINTS])
        n = self.robot.num_dof
        kps, kds = np.zeros(n), np.zeros(n)
        kps[self.arm_idx], kds[self.arm_idx] = ic.arm_stiffness, ic.arm_damping
        kps[self.fin_idx], kds[self.fin_idx] = ic.finger_stiffness, ic.finger_damping
        ctrl = self.robot.get_articulation_controller()
        ctrl.set_gains(kps=kps, kds=kds)
        try:
            ctrl.set_max_efforts(np.full(2, self.gripper.force), joint_indices=self.fin_idx)
        except Exception:  # noqa: BLE001
            print("[isaac] could not set finger max effort; grip force is not limited")
        self.obj = None
        self._q_cmd = None

    def _bind_finger_material(self, root):
        from pxr import UsdShade

        mat = self.m.PhysicsMaterial(prim_path="/World/materials/finger_pad",
                                     static_friction=float(self.cfg.gripper.friction),
                                     dynamic_friction=float(self.cfg.gripper.friction) * 0.8,
                                     restitution=0.0)
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        for name in ("finger_left", "finger_right"):
            prim = stage.GetPrimAtPath(_find_prim(root, name))
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                UsdShade.Material(mat.prim), UsdShade.Tokens.strongerThanDescendants, "physics")

    # -------------------------------------------------------------- sim utils
    def _step(self, n=1, render=False):
        for _ in range(int(n)):
            self.world.step(render=render)

    def _apply(self, q_arm=None, finger=None):
        q = np.array(self.robot.get_joint_positions(), dtype=float)
        tgt = q.copy() if self._q_cmd is None else self._q_cmd.copy()
        if q_arm is not None:
            tgt[self.arm_idx] = q_arm
        if finger is not None:
            tgt[self.fin_idx] = finger
        self._q_cmd = tgt
        self.robot.apply_action(self.m.ArticulationAction(joint_positions=tgt))

    def _wait_arm(self, q_arm, max_steps):
        for _ in range(int(max_steps)):
            self._step()
            err = np.abs(self.robot.get_joint_positions()[self.arm_idx] - q_arm).max()
            if err < float(self.ic.joint_tol):
                return True
        return False

    def _read_object(self):
        pos, quat = self.obj_prim.get_world_pose()
        o = copy.copy(self.obj)
        o.x, o.y = float(pos[0]), float(pos[1])
        o.yaw = _yaw_from_quat(np.asarray(quat))
        return o, np.asarray(pos, dtype=float)

    # ------------------------------------------------------------ interface
    def reset(self, rng):
        self.rng = rng
        self.obj = sample_object(self.cfg.object, rng)
        o = self.obj
        self.obj_prim.set_world_pose(position=np.array([o.x, o.y, o.height / 2 + 0.001]),
                                     orientation=np.array([np.cos(o.yaw / 2), 0, 0, np.sin(o.yaw / 2)]))
        self.obj_prim.set_linear_velocity(np.zeros(3))
        self.obj_prim.set_angular_velocity(np.zeros(3))
        try:
            self.obj_prim.set_mass(o.mass)
            self.obj_mat.set_static_friction(o.mu_s)
            self.obj_mat.set_dynamic_friction(o.mu_k)
        except Exception:  # noqa: BLE001
            pass
        pos, yaw = self.home_tcp(rng)
        q_arm, ok = self.solve_ik(pos, yaw, self.cfg.robot.ik_seed)
        if not ok:
            raise RuntimeError("home pose is not reachable")
        q = np.zeros(self.robot.num_dof)
        q[self.arm_idx] = q_arm
        q[self.fin_idx] = self.gripper.open_width / 2
        self.robot.set_joint_positions(q)
        self.robot.set_joint_velocities(np.zeros_like(q))
        self._q_cmd = None
        self._apply(q_arm, np.full(2, self.gripper.open_width / 2))
        self._step(30)

    def joint_positions(self):
        return np.array(self.robot.get_joint_positions(), dtype=float)[self.arm_idx]

    def move_tcp(self, pos, yaw):
        pos = np.array(pos, dtype=float)
        std = float(self.cfg.robot.execution_noise_std)
        if std > 0:
            pos = pos + self.rng.normal(0.0, std, 3) * np.array([1, 1, 0.5])
        obj, before = self._read_object()
        hit = self.gripper.check_collision(obj, pos, yaw)
        if hit:                                   # same rule as the simple backend
            return MoveResult(f"collision_{hit}")
        q, ok = self.solve_ik(pos, yaw, self.joint_positions())
        if not ok:
            return MoveResult("ik_fail")
        self._apply(q_arm=q)
        self._wait_arm(q, self.ic.move_settle_steps)
        _, after = self._read_object()
        if np.linalg.norm(after[:2] - before[:2]) > float(self.ic.knock_threshold):
            return MoveResult("collision_object")
        return MoveResult("ok")

    def grasp(self):
        g = self.gripper
        obj0, p0 = self._read_object()
        # close: target fully closed, the drive force limit acts as the RG2 force setting
        self._apply(finger=np.zeros(2))
        close_steps = (g.open_width / g.speed + 0.3) / float(self.ic.physics_dt)
        self._step(close_steps)
        width = float(np.sum(self.robot.get_joint_positions()[self.fin_idx]))
        info = {"width": width, "force": g.force, "mass": obj0.mass, "mu_s": obj0.mu_s}
        if width < 0.003:
            self._apply(finger=np.full(2, g.open_width / 2))
            self._step(60)
            return GraspResult(False, "miss", info)
        # lift along a straight line with the IK, one waypoint per physics step
        pos, yaw = self.tcp_pose()
        h = float(self.cfg.task.lift_height)
        n = int(h / float(self.cfg.task.lift_max_speed) / float(self.ic.physics_dt))
        q = self.joint_positions()
        for k in range(1, n + 1):
            q, _ = self.solve_ik(pos + np.array([0, 0, h * k / n]), yaw, q)
            self._apply(q_arm=q)
            self._step()
        self._step(30)
        _, p1 = self._read_object()
        lift = float(p1[2] - p0[2])
        info["object_lift"] = lift
        if lift >= 0.8 * h:
            return GraspResult(True, "success", info)
        self._apply(finger=np.full(2, g.open_width / 2))
        self._step(60)
        return GraspResult(False, "dropped" if lift > 0.01 else "no_lift", info)

    def depth_image(self):
        for _ in range(int(self.ic.render_steps_before_depth)):
            self.world.render()
        d = self.cam.get_depth()
        c = self.cfg.camera
        if d is None or np.size(d) == 0:
            return np.zeros((int(c.height), int(c.width)), np.float32)
        d = np.nan_to_num(np.asarray(d, dtype=np.float64), nan=0.0, posinf=0.0)
        n = c.noise
        sigma = float(n.base_std) + float(n.quad_coeff) * d ** 2
        d = d + self.rng.normal(size=d.shape) * sigma
        d = np.where(self.rng.random(d.shape) < float(n.dropout), 0.0, d)
        if float(n.quantization) > 0:
            d = np.round(d / float(n.quantization)) * float(n.quantization)
        d = np.where((d >= float(c.near)) & (d <= float(c.far)), d, 0.0)
        return d.astype(np.float32)

    def object_state(self):
        return self._read_object()[0]

    def close(self):
        self.app.close()
