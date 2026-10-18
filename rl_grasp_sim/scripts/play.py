"""Drive the robot by hand to understand the MDP (state, action, reward).

    python scripts/play.py              # keys: the action names, e.g.  z-  x+  yaw+  grasp
    python scripts/play.py --frames     # also save a PNG per step to results/play/
"""

from pathlib import Path

import numpy as np
from _common import ROOT, base_parser, config_from_args
from PIL import Image

from grasp_rl.envs import GraspEnv
from grasp_rl.render import render_frame


def show(env, obs, info):
    f, gt = info["features"], info["gt"]
    bins = env.encoder.decode(env.encoder.discrete_index(f))
    print(f"  state index {obs} bins(dx,dy,dz,yaw)={None if bins is None else tuple(int(b) for b in bins)}")
    print(f"  perceived: visible={f.visible} dx={f.dx * 100:+.1f}cm dy={f.dy * 100:+.1f}cm "
          f"dz={f.dz * 100:+.1f}cm yaw={np.rad2deg(f.yaw):+.0f}°")
    print(f"  truth    : dx={gt.dx * 100:+.1f}cm dy={gt.dy * 100:+.1f}cm "
          f"dz={gt.dz * 100:+.1f}cm yaw={np.rad2deg(gt.yaw):+.0f}°")


def main():
    p = base_parser(__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--frames", action="store_true")
    args = p.parse_args()
    cfg = config_from_args(args)
    env = GraspEnv(cfg, "discrete")
    out = ROOT / "results" / "play"
    out.mkdir(parents=True, exist_ok=True)
    obs, info = env.reset(seed=args.seed)
    print("actions:", " ".join(env.actions.names), "| r = reset, q = quit")
    show(env, obs, info)
    ret = 0.0
    while True:
        cmd = input("action> ").strip()
        if cmd == "q":
            break
        if cmd == "r":
            obs, info = env.reset()
            ret = 0.0
            show(env, obs, info)
            continue
        if cmd not in env.actions.names:
            print("  unknown action")
            continue
        obs, r, te, tr, info = env.step(env.actions.names.index(cmd))
        ret += r
        print(f"  reward {r:+.3f} {info['reward_terms']}  events={info['events']}  return {ret:+.2f}")
        if "grasp_info" in info:
            gi = {k: round(v, 4) if isinstance(v, float) else v for k, v in info["grasp_info"].items()}
            print(f"  grasp: {info['grasp_reason']}  {gi}")
        show(env, obs, info)
        if args.frames:
            Image.fromarray(render_frame(env, f"step {info['step']}", cmd, r, info)).save(
                Path(out) / f"step_{info['step']:03d}.png")
        if te or tr:
            print(f"episode over: success={info['success']} (terminated={te}, truncated={tr}); 'r' to reset")


if __name__ == "__main__":
    main()
