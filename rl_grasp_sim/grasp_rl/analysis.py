"""Aggregate training logs into comparison plots and a Markdown summary."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

COLORS = ["#2a6fdb", "#e4572e", "#17a398", "#f3a712", "#7d5ba6", "#666666"]


def _read_csv(path):
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    out = defaultdict(list)
    for r in rows:
        for k, v in r.items():
            try:
                out[k].append(float(v))
            except (TypeError, ValueError):
                out[k].append(v)
    return {k: (np.array(v) if k != "outcome" else v) for k, v in out.items()}


def load_runs(root):
    runs = []
    for s in sorted(Path(root).rglob("summary.json")):
        d = s.parent
        if not (d / "episodes.csv").exists():
            continue
        summary = json.loads(s.read_text())
        runs.append({"dir": d, "summary": summary,
                     "group": summary.get("label") or summary["algo"],
                     "ep": _read_csv(d / "episodes.csv"), "ev": _read_csv(d / "eval.csv")})
    return runs


def moving_average(x, w):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return x
    w = max(1, min(w, len(x)))
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def _band(ax, xs, ys, label, color):
    """Mean ± std over seeds; curves are interpolated onto a common x grid."""
    x_max = min(x[-1] for x in xs)
    grid = np.linspace(min(x[0] for x in xs), x_max, 300)
    Y = np.stack([np.interp(grid, x, y) for x, y in zip(xs, ys)])
    m, s = Y.mean(0), Y.std(0)
    ax.plot(grid, m, color=color, lw=2, label=f"{label} (n={len(xs)})")
    if len(xs) > 1:
        ax.fill_between(grid, m - s, m + s, color=color, alpha=0.18, lw=0)


def _groups(runs):
    g = defaultdict(list)
    for r in runs:
        g[r["group"]].append(r)
    return dict(g)


def episodes_to_reach(ev, threshold):
    idx = np.nonzero(ev["success_rate"] >= threshold)[0]
    return None if len(idx) == 0 else int(ev["episode"][idx[0]])


def make_report(root, window=100, threshold=0.8, title=None):
    root = Path(root)
    runs = load_runs(root)
    if not runs:
        raise FileNotFoundError(f"no runs (summary.json) under {root}")
    groups = _groups(runs)

    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    panels = [
        (axes[0, 0], "episode", lambda r: moving_average(r["ep"]["success"], window),
         "ep", f"Training success rate (moving avg {window})", "episode"),
        (axes[0, 1], "env_steps", lambda r: moving_average(r["ep"]["success"], window),
         "ep", "Training success rate vs environment steps", "environment steps"),
        (axes[0, 2], "episode", lambda r: r["ev"]["success_rate"],
         "ev", "Greedy evaluation success rate", "episode"),
        (axes[1, 0], "episode", lambda r: moving_average(r["ep"]["task_return"], window),
         "ep", "Task return (without shaping)", "episode"),
        (axes[1, 1], "episode", lambda r: moving_average(r["ep"]["length"], window),
         "ep", "Episode length (steps)", "episode"),
    ]
    for ax, xkey, yfun, src, ttl, xl in panels:
        for i, (name, rs) in enumerate(groups.items()):
            _band(ax, [r[src][xkey] for r in rs], [yfun(r) for r in rs], name,
                  COLORS[i % len(COLORS)])
        ax.set_title(ttl)
        ax.set_xlabel(xl)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    for ax in (axes[0, 0], axes[0, 1], axes[0, 2]):
        ax.set_ylim(-0.02, 1.02)

    # outcome breakdown over the last 20 % of training episodes
    ax = axes[1, 2]
    outcomes = sorted({o for r in runs for o in r["ep"]["outcome"]})
    width = 0.8 / max(1, len(groups))
    for i, (name, rs) in enumerate(groups.items()):
        cnt = Counter()
        total = 0
        for r in rs:
            o = r["ep"]["outcome"]
            tail = o[int(0.8 * len(o)):]
            cnt.update(tail)
            total += len(tail)
        vals = [cnt[o] / max(total, 1) for o in outcomes]
        ax.bar(np.arange(len(outcomes)) + i * width, vals, width, label=name,
               color=COLORS[i % len(COLORS)])
    ax.set_xticks(np.arange(len(outcomes)) + width * (len(groups) - 1) / 2)
    ax.set_xticklabels(outcomes, rotation=30, ha="right", fontsize=8)
    ax.set_title("Episode outcomes (last 20% of training)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.suptitle(title or f"UR5 + RG2 grasping — {root.name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(root / "comparison.png", dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------ summary.md
    lines = [f"# Results: {root.name}", "",
             f"| group | seeds | episodes | env steps | final eval success | best eval success | "
             f"episodes to {int(threshold * 100)}% | train success (last {window}) | wall time [s] "
             f"| parameters |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name, rs in groups.items():
        final = np.array([r["summary"]["final_eval"]["success_rate"] for r in rs])
        best = np.array([r["ev"]["success_rate"].max() for r in rs])
        reach = [episodes_to_reach(r["ev"], threshold) for r in rs]
        reach_s = ", ".join("—" if x is None else str(x) for x in reach)
        last = np.array([np.mean(r["ep"]["success"][-window:]) for r in rs])
        steps = np.mean([r["summary"]["env_steps"] for r in rs])
        wall = np.mean([r["summary"]["wall_time_s"] for r in rs])
        params = rs[0]["summary"]["parameters"]
        lines.append(f"| {name} | {len(rs)} | {rs[0]['summary']['episodes']} | {steps:,.0f} | "
                     f"{final.mean():.2f} ± {final.std():.2f} | {best.mean():.2f} | {reach_s} | "
                     f"{last.mean():.2f} | {wall:,.0f} | {params:,} |")
    lines += ["", "![comparison](comparison.png)", ""]
    for name, rs in groups.items():
        extra = {k: v for k, v in rs[0]["summary"].items()
                 if k in ("states_visited", "table_size", "sa_visited", "updates", "buffer")}
        if extra:
            lines.append(f"- **{name}** (seed {rs[0]['summary']['seed']}): {extra}")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root / "comparison.png", root / "summary.md"
