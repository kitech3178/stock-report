"""Train one algorithm.

    python scripts/train.py --algo qlearning
    python scripts/train.py --algo dqn --set dqn.episodes=800 --set dqn.observation=features
    ./python.sh scripts/train.py --algo dqn --backend isaac        (inside Isaac Sim)
"""

from _common import base_parser, config_from_args

from grasp_rl.train_loop import train


def main():
    p = base_parser(__doc__)
    p.add_argument("--algo", choices=["qlearning", "dqn"], required=True)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--out", default=None, help="output directory")
    args = p.parse_args()
    cfg = config_from_args(args)
    seed = args.seed if args.seed is not None else int(cfg.experiment.seed)
    _, summary, out = train(cfg, args.algo, seed=seed, out_dir=args.out, episodes=args.episodes)
    print(f"\nresults -> {out}")
    print(summary)


if __name__ == "__main__":
    main()
