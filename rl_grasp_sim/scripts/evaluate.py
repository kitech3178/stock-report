"""Evaluate a saved policy (greedy) on either backend - e.g. train fast on the
simple physics model, then test the same Q-table / network inside Isaac Sim.

    python scripts/evaluate.py --algo qlearning --model results/compare/qlearning_seed0/q_table.npz
    ~/isaacsim/python.sh scripts/evaluate.py --backend isaac --algo qlearning --model ...
    python scripts/evaluate.py --policy scripted --episodes 200
"""

from collections import Counter

import numpy as np
from _common import base_parser, config_from_args

from grasp_rl.agents import make_agent, observation_mode
from grasp_rl.envs import GraspEnv
from grasp_rl.train_loop import run_episode


def main():
    p = base_parser(__doc__)
    p.add_argument("--algo", choices=["qlearning", "dqn"], default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--policy", choices=["scripted", "random"], default=None)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=100000)
    args = p.parse_args()
    cfg = config_from_args(args)
    algo = args.algo or args.policy or "scripted"
    env = GraspEnv(cfg, observation_mode(algo, cfg))
    agent = make_agent(algo, cfg, env)
    if args.model:
        agent.load(args.model)
    res = [run_episode(env, agent, explore=False, learn=False, seed=args.seed + i)
           for i in range(args.episodes)]
    print(f"{algo} on backend={cfg.backend}: {args.episodes} episodes")
    print(f"  success rate : {np.mean([r['success'] for r in res]):.3f}")
    print(f"  mean length  : {np.mean([r['length'] for r in res]):.1f}")
    print(f"  task return  : {np.mean([r['task_return'] for r in res]):.2f}")
    print(f"  outcomes     : {dict(Counter(r['outcome'] for r in res))}")
    env.close()


if __name__ == "__main__":
    main()
