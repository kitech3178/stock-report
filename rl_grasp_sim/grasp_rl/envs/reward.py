"""REWARD definition: sparse task outcome + penalties + optional potential-based
shaping  F(s, s') = γ·Φ(s') − Φ(s)  (Ng et al., 1999: keeps the optimal policy)."""

from __future__ import annotations

import numpy as np


class RewardFunction:
    def __init__(self, cfg):
        r = cfg.reward
        self.r = r
        self.sh = r.shaping
        self.target_depth = float(cfg.task.target_grasp_depth)

    def potential(self, f):
        """Φ(s): larger (closer to 0) when the hand is in a good grasp pose."""
        if f is None or not f.visible:
            return None
        dist_xy = np.hypot(f.dx, f.dy)
        dz_err = abs(f.dz + self.target_depth)         # want dz = -target_depth
        return -(self.sh.w_xy * dist_xy + self.sh.w_z * dz_err + self.sh.w_yaw * abs(f.yaw))

    def __call__(self, events, phi_prev, phi_next, terminated):
        """Returns (total reward, dict of components) for one transition."""
        comp = {"step": float(self.r.step)}
        for key in ("collision", "ik_fail", "out_of_workspace"):
            if key in events:
                comp[key] = float(self.r[key])
        if "grasp_success" in events:
            comp["success"] = float(self.r.success)
        if "grasp_fail" in events:
            comp["grasp_fail"] = float(self.r.grasp_fail)
        if self.sh.enabled and phi_prev is not None:
            # 'zero': Φ(terminal)=0 keeps the optimal policy unchanged in theory, but
            # every terminal transition then gets a bonus −Φ(s), so grasping right
            # away looks attractive early in training. 'keep' rewards ending in a
            # good pose instead.
            zero = terminated and self.sh.get("terminal_potential", "keep") == "zero"
            nxt = 0.0 if zero else phi_next
            if nxt is not None:
                comp["shaping"] = float(self.sh.scale * (self.sh.gamma * nxt - phi_prev))
        return sum(comp.values()), comp
