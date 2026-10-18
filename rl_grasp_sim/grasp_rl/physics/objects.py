"""Rigid objects resting on the table (z = 0): geometry, ray casting, footprints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def wrap_half_pi(a):
    """Wrap an angle to (-pi/2, pi/2] (parallel-jaw symmetry: 180 deg periodic)."""
    return (a + np.pi / 2) % np.pi - np.pi / 2


@dataclass
class TableObject:
    shape: str                 # "box" | "cylinder"
    size: np.ndarray           # box (sx, sy, sz); cylinder (diameter, diameter, height)
    x: float
    y: float
    yaw: float
    mass: float
    mu_s: float                # finger-object static friction
    mu_k: float                # finger-object kinetic friction

    # ----------------------------------------------------------- basics
    @property
    def height(self):
        return float(self.size[2])

    @property
    def top_z(self):
        return self.height

    @property
    def radius(self):
        return float(self.size[0]) / 2.0

    def center(self):
        return np.array([self.x, self.y, self.height / 2.0])

    def short_axis_angle(self):
        """Direction the closing axis should point to squeeze the narrow side."""
        if self.shape == "cylinder":
            return 0.0
        return self.yaw if self.size[0] <= self.size[1] else self.yaw + np.pi / 2

    def _to_local_xy(self, xy):
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        d = np.asarray(xy, dtype=float) - np.array([self.x, self.y])
        return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1]], -1)

    # ------------------------------------------------------- footprint
    def overlaps_rect(self, center_xy, yaw, half_extents) -> bool:
        """Does the object footprint intersect an oriented rectangle?"""
        half = np.asarray(half_extents, dtype=float)
        if self.shape == "cylinder":
            c, s = np.cos(yaw), np.sin(yaw)
            d = np.array([self.x, self.y]) - np.asarray(center_xy)
            local = np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])
            closest = np.clip(local, -half, half)
            return bool(np.linalg.norm(local - closest) < self.radius)
        return _obb_overlap(np.array([self.x, self.y]), self.yaw, self.size[:2] / 2.0,
                            np.asarray(center_xy, dtype=float), yaw, half)

    # ------------------------------------------------------- ray casting
    def ray_cast(self, origin, dirs):
        """Distance t (in units of |dir|) to the first hit, np.inf for a miss.

        origin: (3,), dirs: (N, 3) world directions (need not be unit length).
        """
        o_l = np.empty(3)
        o_l[:2] = self._to_local_xy(origin[:2])
        o_l[2] = origin[2]
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        d_l = np.stack([c * dirs[:, 0] + s * dirs[:, 1], -s * dirs[:, 0] + c * dirs[:, 1],
                        dirs[:, 2]], -1)
        if self.shape == "cylinder":
            return _ray_cylinder(o_l, d_l, self.radius, self.height)
        lo = np.array([-self.size[0] / 2, -self.size[1] / 2, 0.0])
        hi = np.array([self.size[0] / 2, self.size[1] / 2, self.height])
        return _ray_aabb(o_l, d_l, lo, hi)


def _obb_overlap(c1, yaw1, h1, c2, yaw2, h2) -> bool:
    """Separating axis test for two 2D oriented boxes."""
    axes = []
    for yaw in (yaw1, yaw2):
        axes.append(np.array([np.cos(yaw), np.sin(yaw)]))
        axes.append(np.array([-np.sin(yaw), np.cos(yaw)]))
    a1, b1 = axes[0], axes[1]
    a2, b2 = axes[2], axes[3]
    d = c2 - c1
    for ax in axes:
        r1 = h1[0] * abs(a1 @ ax) + h1[1] * abs(b1 @ ax)
        r2 = h2[0] * abs(a2 @ ax) + h2[1] * abs(b2 @ ax)
        if abs(d @ ax) > r1 + r2:
            return False
    return True


def _ray_aabb(o, d, lo, hi):
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1 = (lo - o) * inv
        t2 = (hi - o) * inv
    tmin = np.nanmax(np.minimum(t1, t2), axis=1)
    tmax = np.nanmin(np.maximum(t1, t2), axis=1)
    hit = (tmax >= tmin) & (tmax > 0)
    t = np.where(tmin > 0, tmin, tmax)
    return np.where(hit, t, np.inf)


def _ray_cylinder(o, d, r, h):
    t_best = np.full(d.shape[0], np.inf)
    # side wall
    a = d[:, 0] ** 2 + d[:, 1] ** 2
    b = 2 * (o[0] * d[:, 0] + o[1] * d[:, 1])
    c = o[0] ** 2 + o[1] ** 2 - r * r
    disc = b * b - 4 * a * c
    ok = (disc >= 0) & (a > 1e-12)
    sq = np.sqrt(np.where(ok, disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
            z = o[2] + t * d[:, 2]
            good = ok & (t > 0) & (z >= 0) & (z <= h)
            t_best = np.where(good & (t < t_best), t, t_best)
        # top cap
        t = (h - o[2]) / d[:, 2]
    px = o[0] + t * d[:, 0]
    py = o[1] + t * d[:, 1]
    good = (t > 0) & (px ** 2 + py ** 2 <= r * r)
    return np.where(good & (t < t_best), t, t_best)


def sample_object(obj_cfg, rng) -> TableObject:
    size = np.array(obj_cfg["size"], dtype=float)
    jit = float(obj_cfg.get("size_jitter") or 0.0)
    if jit > 0:
        size = size * (1.0 + rng.uniform(-jit, jit, size=3))
    shape = obj_cfg["shape"]
    if shape == "cylinder":
        size[1] = size[0]
    cx, cy = obj_cfg["spawn_center"]
    hx, hy = obj_cfg["spawn_half_extent"]
    yaw_lo, yaw_hi = np.deg2rad(obj_cfg["yaw_range_deg"])
    mass = obj_cfg["mass"] if not obj_cfg.get("mass_range") else rng.uniform(*obj_cfg["mass_range"])
    mu = obj_cfg["friction"] if not obj_cfg.get("friction_range") else rng.uniform(*obj_cfg["friction_range"])
    return TableObject(shape=shape, size=size,
                       x=cx + rng.uniform(-hx, hx), y=cy + rng.uniform(-hy, hy),
                       yaw=rng.uniform(yaw_lo, yaw_hi), mass=float(mass),
                       mu_s=float(mu), mu_k=float(mu) * float(obj_cfg["kinetic_ratio"]))
