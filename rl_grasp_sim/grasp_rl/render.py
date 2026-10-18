"""Matplotlib rendering of one environment step (3-D arm, top view, depth image)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402


def _rect(center, yaw, half):
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    pts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * half
    return pts @ R.T + center


def _box_edges(obj):
    if obj.shape == "cylinder":
        th = np.linspace(0, 2 * np.pi, 25)
        ring = np.stack([obj.x + obj.radius * np.cos(th), obj.y + obj.radius * np.sin(th)], -1)
        return [np.c_[ring, np.zeros(25)], np.c_[ring, np.full(25, obj.height)]]
    base = _rect(np.array([obj.x, obj.y]), obj.yaw, obj.size[:2] / 2)
    lo = np.c_[base, np.zeros(4)]
    hi = np.c_[base, np.full(4, obj.height)]
    edges = [np.vstack([lo, lo[:1]]), np.vstack([hi, hi[:1]])]
    edges += [np.vstack([lo[i], hi[i]]) for i in range(4)]
    return edges


def render_frame(env, title="", action=None, reward=None, info=None):
    b = env.backend
    obj = b.object_state()
    q = b.joint_positions()
    frames = b.kin.frames(q)
    T_tcp = b.kin.tcp(q)
    pos, yaw = b.tcp_pose()
    depth = env.last["depth"]
    g = b.gripper

    fig = plt.figure(figsize=(14, 4.6))
    # --- 3-D view
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    pts = np.array([F[:3, 3] for F in frames] + [T_tcp[:3, 3]])
    ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], "-o", color="#4a6fa5", lw=4, ms=4)
    u = np.array([np.cos(yaw), np.sin(yaw), 0])
    for s in (1, -1):
        p = pos + s * u * g.open_width / 2
        ax.plot([p[0], p[0]], [p[1], p[1]], [p[2], p[2] + g.finger_len], color="k", lw=3)
    for e in _box_edges(obj):
        ax.plot(e[:, 0], e[:, 1], e[:, 2], color="#e4572e")
    T_cam = b.T_world_cam()
    c = T_cam[:3, 3]
    z = T_cam[:3, 2] * 0.08
    ax.quiver(c[0], c[1], c[2], z[0], z[1], z[2], color="#17a398")
    ax.set_xlim(-0.1, 0.7)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(0, 0.8)
    ax.set_title("UR5 + RG2 (3-D)")
    ax.view_init(25, -60)

    # --- top view
    ax = fig.add_subplot(1, 3, 2)
    cx, cy = env.cfg.object.spawn_center
    if obj.shape == "cylinder":
        ax.add_patch(plt.Circle((obj.x, obj.y), obj.radius, color="#e4572e", alpha=0.7))
    else:
        ax.add_patch(Polygon(_rect(np.array([obj.x, obj.y]), obj.yaw, obj.size[:2] / 2),
                             color="#e4572e", alpha=0.7))
    for cen, yw, half in g.finger_rects(pos[:2], yaw, g.open_width):
        ax.add_patch(Polygon(_rect(cen, yw, half), color="k"))
    ax.plot(*pos[:2], "+", color="#2a6fdb", ms=12, mew=2)
    ax.plot([pos[0], pos[0] + 0.04 * np.cos(yaw)], [pos[1], pos[1] + 0.04 * np.sin(yaw)],
            color="#2a6fdb")
    hx, hy = env.cfg.task.workspace_half_extent
    ax.add_patch(plt.Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy, fill=False, ls="--", color="grey"))
    ax.set_xlim(cx - hx - 0.02, cx + hx + 0.02)
    ax.set_ylim(cy - hy - 0.02, cy + hy + 0.02)
    ax.set_aspect("equal")
    ax.set_title(f"top view   tip height {pos[2] * 100:.1f} cm")
    ax.grid(alpha=0.3)

    # --- depth image
    ax = fig.add_subplot(1, 3, 3)
    im = ax.imshow(np.where(depth > 0, depth, np.nan), cmap="viridis_r",
                   vmin=0.05, vmax=env.cfg.camera.far)
    fig.colorbar(im, ax=ax, fraction=0.035, label="depth [m]")
    f = env.last["perceived"]
    txt = (f"visible={f.visible}  dx={f.dx * 100:+.1f}cm dy={f.dy * 100:+.1f}cm\n"
           f"dz={f.dz * 100:+.1f}cm  yaw={np.rad2deg(f.yaw):+.0f}°")
    ax.set_title("wrist depth camera (perceived)\n" + txt, fontsize=9)
    ax.axis("off")

    sub = title
    if action is not None:
        sub += f"   action={action}"
    if reward is not None:
        sub += f"   r={reward:+.2f}"
    if info and info.get("grasp_reason"):
        sub += f"   grasp: {info['grasp_reason']}"
    fig.suptitle(sub)
    fig.tight_layout()
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img
