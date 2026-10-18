"""Q-learning vs DQN on the same environment, several seeds, one report.

    python scripts/compare.py --seeds 0 1 2
    python scripts/compare.py --algos qlearning dqn --seeds 0 --name quick \
        --set qlearning.episodes=2000 --set dqn.episodes=800
    python scripts/compare.py --replot results/compare      (only redraw plots)

Writes results/<name>/{comparison.png, summary.md} plus one folder per run.
"""

from _common import ROOT, base_parser, config_from_args

from grasp_rl.analysis import make_report
from grasp_rl.train_loop import train


def main():
    p = base_parser(__doc__)
    p.add_argument("--algos", nargs="+", default=["qlearning", "dqn"],
                   help="any of: qlearning dqn dqn_features (DQN on the feature vector)")
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--name", default="compare")
    p.add_argument("--replot", default=None, help="existing result folder to re-plot")
    args = p.parse_args()
    if args.replot:
        print(make_report(args.replot))
        return
    cfg = config_from_args(args)
    root = ROOT / cfg.experiment.output_dir / args.name
    for algo in args.algos:
        run_algo, label = algo, algo
        run_cfg = cfg
        if algo == "dqn_features":
            run_algo = "dqn"
            run_cfg = config_from_args(args)
            run_cfg.dqn.observation = "features"
        elif algo == "dqn":
            label = f"dqn ({cfg.dqn.observation})"
        for seed in args.seeds:
            train(run_cfg, run_algo, seed=seed, out_dir=root / f"{algo}_seed{seed}", label=label)
    png, md = make_report(root)
    print(f"\nplots   -> {png}\nsummary -> {md}")
    print(md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
