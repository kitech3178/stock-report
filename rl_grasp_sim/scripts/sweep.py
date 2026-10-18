"""Parameter sweep: train one algorithm for several values of one config key.

    python scripts/sweep.py --algo qlearning --param qlearning.alpha --values 0.05 0.2 0.5
    python scripts/sweep.py --algo qlearning --param reward.shaping.enabled --values true false
    python scripts/sweep.py --algo dqn --param dqn.double_dqn --values true false --seeds 0 1
    python scripts/sweep.py --algo qlearning --param gripper.force --values 5 20 40

Writes results/sweep_<param>/{comparison.png, summary.md}.
"""

from _common import ROOT, base_parser, config_from_args

from grasp_rl.analysis import make_report
from grasp_rl.config import apply_override
from grasp_rl.train_loop import train


def main():
    p = base_parser(__doc__)
    p.add_argument("--algo", choices=["qlearning", "dqn"], required=True)
    p.add_argument("--param", required=True, help="dotted config key, e.g. qlearning.alpha")
    p.add_argument("--values", nargs="+", required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--episodes", type=int, default=None)
    p.add_argument("--name", default=None)
    args = p.parse_args()
    name = args.name or f"sweep_{args.algo}_{args.param.replace('.', '_')}"
    root = ROOT / "results" / name
    for v in args.values:
        cfg = config_from_args(args)
        apply_override(cfg, f"{args.param}={v}")
        label = f"{args.param}={v}"
        for seed in args.seeds:
            train(cfg, args.algo, seed=seed, episodes=args.episodes, label=label,
                  out_dir=root / f"{args.param}={v}" / f"seed{seed}")
    png, md = make_report(root, title=f"{args.algo}: sweep over {args.param}")
    print(f"\nplots   -> {png}\nsummary -> {md}")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
