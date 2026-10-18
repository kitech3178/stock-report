"""Sanity tests:  python -m pytest tests   (or  python tests/test_core.py)"""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from grasp_rl.agents import make_agent  # noqa: E402
from grasp_rl.config import load_config  # noqa: E402
from grasp_rl.envs import GraspEnv  # noqa: E402
from grasp_rl.physics.objects import TableObject  # noqa: E402
from grasp_rl.physics.rg2_gripper import RG2Gripper  # noqa: E402
from grasp_rl.physics.ur5_kinematics import UR5Kinematics, down_rotation, make_T, rot_x, rot_z  # noqa: E402
from grasp_rl.urdf import generate_urdf  # noqa: E402

CFG = load_config()


def _box(x=0.45, y=0.0, yaw=0.0, mass=0.3, mu=0.6):
    return TableObject("box", np.array([0.04, 0.07, 0.05]), x, y, yaw, mass, mu, 0.8 * mu)


def test_fk_ik_roundtrip():
    kin = UR5Kinematics(CFG.robot, CFG.gripper.tcp_offset)
    rng = np.random.default_rng(0)
    for _ in range(20):
        p = np.array([0.45, 0.0, 0.1]) + rng.uniform(-0.12, 0.12, 3) * [1, 1, 0.5]
        yaw = rng.uniform(-1.5, 1.5)
        q, ok = kin.ik(p, down_rotation(yaw), CFG.robot.ik_seed)
        assert ok
        T = kin.tcp(q)
        assert np.allclose(T[:3, 3], p, atol=1e-3)
        assert np.allclose(T[:3, :3], down_rotation(yaw), atol=5e-3)


def _rpy(r, p, y):
    return rot_z(y) @ (np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]]) @ rot_x(r))


def test_urdf_matches_dh():
    """Forward kinematics through the generated URDF == DH forward kinematics."""
    root = ET.fromstring(generate_urdf(CFG))
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    kin = UR5Kinematics(CFG.robot, CFG.gripper.tcp_offset)
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = rng.uniform(-3, 3, 6)
        link, T = "tcp", np.eye(4)
        while link in joints:
            j = joints[link]
            o = j.find("origin")
            Tj = make_T(_rpy(*map(float, o.get("rpy").split())), list(map(float, o.get("xyz").split())))
            if j.get("type") == "revolute":
                i = int(j.get("name").split("_")[1]) - 1
                Tj = Tj @ make_T(rot_z(q[i]), [0, 0, 0])
            T = Tj @ T
            link = j.find("parent").get("link")
        assert np.allclose(T, kin.tcp(q), atol=1e-6)


def test_camera_sees_object():
    env = GraspEnv(CFG, "discrete")
    env.reset(seed=3)
    f = env.last["perceived"]
    gt = env.last["gt"]
    assert f.visible
    assert abs(f.dx - gt.dx) < 0.01 and abs(f.dy - gt.dy) < 0.01
    assert abs(f.dz - gt.dz) < 0.01


def test_grasp_physics():
    g = RG2Gripper(CFG.gripper)
    task = CFG.task
    tcp = np.array([0.45, 0.0, 0.025])
    assert g.grasp_and_lift(_box(), tcp, 0.0, task).success
    assert g.grasp_and_lift(_box(yaw=np.deg2rad(45)), tcp, 0.0, task).reason == "misaligned"
    assert g.grasp_and_lift(_box(), tcp + [0, 0, 0.05], 0.0, task).reason == "no_contact_height"
    assert g.grasp_and_lift(_box(), tcp + [0, 0.06, 0], 0.0, task).reason == "miss"
    weak = RG2Gripper({**CFG.gripper, "force": 3.0})
    assert weak.grasp_and_lift(_box(mass=1.5), tcp, 0.0, task).reason in ("dropped", "rotational_slip")
    # a grasp far from the centre of mass twists out of the fingers
    assert weak.grasp_and_lift(_box(mass=0.5), tcp + [0, 0.03, 0], 0.0, task).reason == "rotational_slip"
    # collision: an open finger landing on the object
    assert g.check_collision(_box(), np.array([0.45 + 0.05, 0.0, 0.03]), 0.0) == "object"
    assert g.check_collision(_box(), np.array([0.45, 0.0, 0.0]), 0.0) == "table"


def test_env_modes_and_scripted_policy():
    for mode in ("discrete", "features", "depth"):
        env = GraspEnv(CFG, mode)
        obs, _ = env.reset(seed=0)
        assert env.observation_space.contains(obs), mode
    env = GraspEnv(CFG, "discrete")
    agent = make_agent("scripted", CFG, env)
    wins = 0
    for ep in range(10):
        obs, info = env.reset(seed=ep)
        while True:
            obs, r, te, tr, info = env.step(agent.act(obs))
            if te or tr:
                break
        wins += info["success"]
    assert wins >= 9


def test_q_learning_update():
    cfg = load_config(None, ["qlearning.alpha=0.5", "qlearning.gamma=0.9"])
    agent = make_agent("qlearning", cfg, type("E", (), {
        "observation_space": type("S", (), {"n": 3})(), "action_space": type("A", (), {"n": 2})()})())
    agent.Q[1] = [1.0, 2.0]
    agent.observe(0, 0, 1.0, 1, False)
    assert np.isclose(agent.Q[0, 0], 0.5 * (1.0 + 0.9 * 2.0))
    agent.observe(0, 1, 1.0, 1, True)                       # terminal: no bootstrap
    assert np.isclose(agent.Q[0, 1], 0.5)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
