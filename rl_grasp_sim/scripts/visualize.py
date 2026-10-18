"""Roll out a policy and save a GIF + the final frame.

    python scripts/visualize.py --policy scripted
    python scripts/visualize.py --algo qlearning --model results/compare/qlearning_seed0/q_table.npz
    python scripts/visualize.py --algo dqn --model results/compare/dqn_seed0/dqn.pt --seed 3
"""

from pathlib import Path

import numpy as np
from _common import ROOT, base_parser, config_from_args
from PIL import Image

from grasp_rl.agents import make_agent, observation_mode
from grasp_rl.envs import GraspEnv
from grasp_rl.render import render_frame


def main():
    p = base_parser(__doc__)
    p.add_argument("--algo", choices=["qlearning", "dqn"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--policy", choices=["scripted", "random"], default=None)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--out", default=str(ROOT / "results" / "videos"))
    args = p.parse_args()
    cfg = config_from_args(args)
    algo = args.algo or args.policy or "scripted"
    env = GraspEnv(cfg, observation_mode(algo, cfg))
    agent = make_agent(algo, cfg, env)
    if args.model:
        agent.load(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for k in range(args.episodes):
        seed = args.seed + k
        obs, info = env.reset(seed=seed)
        frames = [render_frame(env, f"{algo}  seed {seed}  step 0")]
        while True:
            a = agent.act(obs, explore=False)
            obs, r, te, tr, info = env.step(a)
            frames.append(render_frame(env, f"{algo}  seed {seed}  step {info['step']}",
                                       env.actions.names[a], r, info))
            if te or tr:
                break
        frames += [frames[-1]] * 4
        imgs = [Image.fromarray(f) for f in frames]
        gif = out / f"{algo}_seed{seed}.gif"
        imgs[0].save(gif, save_all=True, append_images=imgs[1:], duration=350, loop=0)
        imgs[-1].save(out / f"{algo}_seed{seed}_final.png")
        print(f"seed {seed}: success={info['success']} steps={info['step']} -> {gif}")
    env.close()


if __name__ == "__main__":
    np.set_printoptions(precision=3)
    main()
