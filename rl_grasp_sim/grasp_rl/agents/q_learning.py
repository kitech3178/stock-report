"""Tabular Q-learning (Watkins, 1989).

    Q(s,a) <- Q(s,a) + α [ r + γ · max_a' Q(s',a') · (1 − terminated) − Q(s,a) ]

A time-limit truncation is *not* a terminal state, so we still bootstrap there.
"""

from __future__ import annotations

import numpy as np

from grasp_rl.agents.schedules import epsilon_at


class QLearningAgent:
    def __init__(self, qcfg, n_states, n_actions, seed=0):
        self.cfg = qcfg
        self.nS, self.nA = n_states, n_actions
        self.Q = np.full((n_states, n_actions), float(qcfg.q_init))
        self.N = np.zeros((n_states, n_actions), dtype=np.int64)
        self.alpha = float(qcfg.alpha)
        self.gamma = float(qcfg.gamma)
        self.rng = np.random.default_rng(seed)
        self.episode = 0
        self.epsilon = float(qcfg.epsilon_start)

    def act(self, s, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.nA))
        q = self.Q[s]
        return int(self.rng.choice(np.flatnonzero(q == q.max())))   # random tie-break

    def step_size(self, s, a):
        if self.cfg.alpha_decay == "visit":
            return max(float(self.cfg.alpha_min), 1.0 / self.N[s, a] ** float(self.cfg.alpha_omega))
        return self.alpha

    def observe(self, s, a, r, s2, terminated):
        self.N[s, a] += 1
        target = r if terminated else r + self.gamma * self.Q[s2].max()
        td = target - self.Q[s, a]
        self.Q[s, a] += self.step_size(s, a) * td
        return {"td_error": abs(td)}

    def end_episode(self):
        self.episode += 1
        c = self.cfg
        self.epsilon = epsilon_at(self.episode, c.epsilon_start, c.epsilon_end,
                                  c.epsilon_decay, c.epsilon_decay_episodes)

    def stats(self):
        visited = int((self.N.sum(1) > 0).sum())
        return {"states_visited": visited, "table_size": self.Q.size,
                "sa_visited": int((self.N > 0).sum())}

    def num_parameters(self):
        return int(self.Q.size)

    def save(self, path):
        np.savez_compressed(path, Q=self.Q, N=self.N, epsilon=self.epsilon, episode=self.episode)

    def load(self, path):
        d = np.load(path)
        self.Q, self.N = d["Q"], d["N"]
        self.epsilon, self.episode = float(d["epsilon"]), int(d["episode"])
