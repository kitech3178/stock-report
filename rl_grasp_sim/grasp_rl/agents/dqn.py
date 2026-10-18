"""Deep Q-Network (Mnih et al., 2015) with optional Double DQN (van Hasselt et al., 2016)
and n-step returns (Sutton & Barto ch. 7; used in Rainbow, Hessel et al., 2018).

    G = r_t + γ r_{t+1} + ... + γ^{n-1} r_{t+n-1}
    y = G + γ^n · (1 − terminated) · Q_target(s_{t+n}, argmax_a Q_online(s_{t+n}, a))   (double)
    y = G + γ^n · (1 − terminated) · max_a Q_target(s_{t+n}, a)                         (vanilla)
    loss = Huber(Q_online(s_t, a_t) − y)

Observation 'depth'    -> CNN over the wrist depth image (+ proprioception)
Observation 'features' -> MLP over the perceived grasp features
"""

from __future__ import annotations

from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from grasp_rl.agents.schedules import epsilon_at


def _mlp(sizes):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class DepthQNet(nn.Module):
    def __init__(self, img_shape, proprio_dim, n_actions, channels, hidden):
        super().__init__()
        c0, c1, c2 = channels
        self.conv = nn.Sequential(
            nn.Conv2d(img_shape[0], c0, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(c0, c1, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flat = self.conv(torch.zeros(1, *img_shape)).shape[1]
        self.head = _mlp([n_flat + proprio_dim, *hidden, n_actions])

    def forward(self, depth, proprio):
        return self.head(torch.cat([self.conv(depth), proprio], dim=1))


class FeatureQNet(nn.Module):
    def __init__(self, in_dim, n_actions, hidden):
        super().__init__()
        self.net = _mlp([in_dim, *hidden, n_actions])

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Ring buffer. Depth images are stored as uint8 to save memory."""

    def __init__(self, size, obs_space, rng):
        self.size, self.rng = int(size), rng
        self.ptr = self.n = 0
        self.dict_obs = hasattr(obs_space, "spaces")
        if self.dict_obs:
            img = obs_space["depth"].shape
            pro = obs_space["proprio"].shape
            self.img = np.zeros((2, self.size, *img), dtype=np.uint8)
            self.pro = np.zeros((2, self.size, *pro), dtype=np.float32)
        else:
            self.vec = np.zeros((2, self.size, *obs_space.shape), dtype=np.float32)
        self.a = np.zeros(self.size, dtype=np.int64)
        self.r = np.zeros(self.size, dtype=np.float32)
        self.d = np.zeros(self.size, dtype=np.float32)
        self.g = np.zeros(self.size, dtype=np.float32)    # discount γ^k for the bootstrap

    def _put(self, k, obs):
        if self.dict_obs:
            self.img[k, self.ptr] = np.round(obs["depth"] * 255).astype(np.uint8)
            self.pro[k, self.ptr] = obs["proprio"]
        else:
            self.vec[k, self.ptr] = obs

    def add(self, s, a, r, s2, done, discount):
        self._put(0, s)
        self._put(1, s2)
        self.a[self.ptr], self.r[self.ptr], self.d[self.ptr] = a, r, float(done)
        self.g[self.ptr] = discount
        self.ptr = (self.ptr + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch, device):
        idx = self.rng.integers(0, self.n, size=batch)

        def obs(k):
            if self.dict_obs:
                return (torch.as_tensor(self.img[k, idx], device=device).float() / 255.0,
                        torch.as_tensor(self.pro[k, idx], device=device))
            return (torch.as_tensor(self.vec[k, idx], device=device),)

        return (obs(0), torch.as_tensor(self.a[idx], device=device),
                torch.as_tensor(self.r[idx], device=device), obs(1),
                torch.as_tensor(self.d[idx], device=device),
                torch.as_tensor(self.g[idx], device=device))


class DQNAgent:
    def __init__(self, dcfg, obs_space, n_actions, seed=0):
        self.cfg = dcfg
        self.nA = n_actions
        self.rng = np.random.default_rng(seed)
        torch.manual_seed(seed)
        dev = dcfg.device
        self.device = torch.device("cuda" if dev == "auto" and torch.cuda.is_available()
                                   else ("cpu" if dev == "auto" else dev))
        self.dict_obs = hasattr(obs_space, "spaces")

        def build():
            if self.dict_obs:
                return DepthQNet(obs_space["depth"].shape, obs_space["proprio"].shape[0],
                                 n_actions, dcfg.conv_channels, dcfg.hidden)
            return FeatureQNet(obs_space.shape[0], n_actions, dcfg.hidden)

        self.q = build().to(self.device)
        self.q_target = build().to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=float(dcfg.lr))
        self.buffer = ReplayBuffer(dcfg.buffer_size, obs_space, self.rng)
        self.gamma = float(dcfg.gamma)
        self.n_step = int(dcfg.get("n_step", 1))
        self.pending = deque()                 # last n transitions of the episode
        self.total_steps = 0
        self.updates = 0
        self.episode = 0
        self.epsilon = float(dcfg.epsilon_start)

    # ------------------------------------------------------------------
    def _tensor_obs(self, obs):
        if self.dict_obs:
            return (torch.as_tensor(obs["depth"], device=self.device)[None],
                    torch.as_tensor(obs["proprio"], device=self.device)[None])
        return (torch.as_tensor(obs, device=self.device)[None],)

    def q_values(self, obs):
        with torch.no_grad():
            return self.q(*self._tensor_obs(obs))[0].cpu().numpy()

    def act(self, obs, explore=True):
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.nA))
        return int(np.argmax(self.q_values(obs)))

    def _emit(self, terminated):
        """Turn the oldest pending transition into an n-step transition."""
        G = 0.0
        for k, (_, _, r, _) in enumerate(self.pending):
            G += self.gamma ** k * r
        s0, a0 = self.pending[0][0], self.pending[0][1]
        s_last = self.pending[-1][3]
        self.buffer.add(s0, a0, G, s_last, terminated, self.gamma ** len(self.pending))
        self.pending.popleft()

    def observe(self, s, a, r, s2, terminated):
        self.pending.append((s, a, r, s2))
        if terminated:
            while self.pending:
                self._emit(True)
        elif len(self.pending) >= self.n_step:
            self._emit(False)
        self.total_steps += 1
        c = self.cfg
        out = {}
        if self.total_steps >= c.learning_starts and self.total_steps % c.train_freq == 0:
            losses = [self._update() for _ in range(int(c.gradient_steps))]
            out["loss"] = float(np.mean(losses))
        if c.target_update == "hard" and self.total_steps % c.target_update_interval == 0:
            self.q_target.load_state_dict(self.q.state_dict())
        return out

    def _update(self):
        c = self.cfg
        s, a, r, s2, d, disc = self.buffer.sample(int(c.batch_size), self.device)
        q = self.q(*s).gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            if c.double_dqn:
                a2 = self.q(*s2).argmax(1, keepdim=True)
                q2 = self.q_target(*s2).gather(1, a2).squeeze(1)
            else:
                q2 = self.q_target(*s2).max(1).values
            y = r + disc * (1.0 - d) * q2
        loss = F.smooth_l1_loss(q, y) if c.huber_loss else F.mse_loss(q, y)
        self.opt.zero_grad()
        loss.backward()
        if c.grad_clip:
            nn.utils.clip_grad_norm_(self.q.parameters(), float(c.grad_clip))
        self.opt.step()
        self.updates += 1
        if c.target_update == "soft":
            with torch.no_grad():
                for p, pt in zip(self.q.parameters(), self.q_target.parameters()):
                    pt.mul_(1 - c.tau).add_(c.tau * p)
        return loss.item()

    def end_episode(self):
        while self.pending:                    # time-limit truncation: still bootstrap
            self._emit(False)
        self.episode += 1
        c = self.cfg
        self.epsilon = epsilon_at(self.episode, c.epsilon_start, c.epsilon_end,
                                  c.epsilon_decay, c.epsilon_decay_episodes)

    def stats(self):
        return {"buffer": self.buffer.n, "updates": self.updates}

    def num_parameters(self):
        return int(sum(p.numel() for p in self.q.parameters()))

    def save(self, path):
        torch.save({"q": self.q.state_dict(), "epsilon": float(self.epsilon),
                    "episode": int(self.episode)}, path)

    def load(self, path):
        d = torch.load(path, map_location=self.device, weights_only=True)
        self.q.load_state_dict(d["q"])
        self.q_target.load_state_dict(d["q"])
        self.epsilon, self.episode = d["epsilon"], d["episode"]
