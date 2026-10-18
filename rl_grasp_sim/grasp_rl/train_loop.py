"""Training / evaluation loop shared by every algorithm."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from grasp_rl.agents import make_agent, observation_mode
from grasp_rl.config import PROJECT_ROOT, save_config
from grasp_rl.envs import GraspEnv

EPISODE_FIELDS = ["episode", "env_steps", "return", "task_return", "length", "success",
                  "outcome", "epsilon", "td_or_loss", "wall_time"]
EVAL_FIELDS = ["episode", "env_steps", "success_rate", "mean_return", "mean_task_return",
               "mean_length", "wall_time"]


def run_episode(env, agent, explore=True, learn=True, seed=None, recorder=None):
    obs, info = env.reset(seed=seed)
    ret = task_ret = 0.0
    stats = []
    while True:
        a = agent.act(obs, explore=explore)
        obs2, r, terminated, truncated, info = env.step(a)
        if learn:
            out = agent.observe(obs, a, r, obs2, terminated)
            v = out.get("loss", out.get("td_error"))
            if v is not None:
                stats.append(v)
        if recorder is not None:
            recorder(env, a, r, info)
        ret += r
        task_ret += r - info["reward_terms"].get("shaping", 0.0)
        obs = obs2
        if terminated or truncated:
            break
    outcome = info.get("grasp_reason", "timeout") if not info["success"] else "success"
    return {"return": ret, "task_return": task_ret, "length": info["step"],
            "success": int(info["success"]), "outcome": outcome,
            "td_or_loss": float(np.mean(stats)) if stats else float("nan")}


def evaluate(env, agent, n, seed_offset):
    saved = env._rng                                  # keep the training sequence intact
    res = [run_episode(env, agent, explore=False, learn=False, seed=seed_offset + i)
           for i in range(n)]
    env._rng = saved
    return {"success_rate": float(np.mean([r["success"] for r in res])),
            "mean_return": float(np.mean([r["return"] for r in res])),
            "mean_task_return": float(np.mean([r["task_return"] for r in res])),
            "mean_length": float(np.mean([r["length"] for r in res]))}


def train(cfg, algo, seed=0, out_dir=None, episodes=None, env=None, verbose=True, label=None):
    """Train `algo` ('qlearning' | 'dqn') and write logs to `out_dir`."""
    acfg = cfg[algo]
    episodes = int(episodes or acfg.episodes)
    env = env or GraspEnv(cfg, observation_mode(algo, cfg))
    agent = make_agent(algo, cfg, env, seed)
    out = Path(out_dir or PROJECT_ROOT / cfg.experiment.output_dir / cfg.experiment.name
               / f"{algo}_seed{seed}")
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config.yaml")

    f_ep = open(out / "episodes.csv", "w", newline="")
    f_ev = open(out / "eval.csv", "w", newline="")
    w_ep = csv.DictWriter(f_ep, EPISODE_FIELDS)
    w_ev = csv.DictWriter(f_ev, EVAL_FIELDS)
    w_ep.writeheader()
    w_ev.writeheader()

    t0 = time.time()
    env_steps = 0
    recent = []
    ev_every, ev_n = int(cfg.eval.every), int(cfg.eval.episodes)
    for ep in range(1, episodes + 1):
        eps = agent.epsilon
        r = run_episode(env, agent, explore=True, learn=True, seed=seed * 10_000_000 + ep)
        agent.end_episode()
        env_steps += r["length"]
        row = {"episode": ep, "env_steps": env_steps, "epsilon": round(eps, 4),
               "wall_time": round(time.time() - t0, 2), **r}
        w_ep.writerow(row)
        recent.append(r)
        if ep % ev_every == 0 or ep == episodes:
            ev = evaluate(env, agent, ev_n, int(cfg.eval.seed_offset))
            w_ev.writerow({"episode": ep, "env_steps": env_steps,
                           "wall_time": round(time.time() - t0, 2), **ev})
            f_ev.flush()
            f_ep.flush()
        if verbose and (ep % int(cfg.log.print_every) == 0 or ep == episodes):
            last = recent[-int(cfg.log.print_every):]
            msg = (f"[{algo} s{seed}] ep {ep:5d}/{episodes}  steps {env_steps:7d}  "
                   f"ε {eps:.3f}  return {np.mean([x['return'] for x in last]):7.2f}  "
                   f"success {np.mean([x['success'] for x in last]):.2f}  "
                   f"len {np.mean([x['length'] for x in last]):5.1f}")
            if ep % ev_every == 0 or ep == episodes:
                msg += f"  | eval success {ev['success_rate']:.2f}"
            print(msg, flush=True)
    f_ep.close()
    f_ev.close()

    summary = {"algo": algo, "label": label, "seed": seed, "episodes": episodes, "env_steps": env_steps,
               "wall_time_s": round(time.time() - t0, 1),
               "parameters": agent.num_parameters(), "final_eval": ev, **agent.stats()}
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    if cfg.log.save_model:
        agent.save(str(out / ("q_table.npz" if algo == "qlearning" else "dqn.pt")))
    return agent, summary, out
