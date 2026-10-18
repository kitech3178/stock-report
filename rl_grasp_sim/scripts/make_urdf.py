"""Write assets/urdf/ur5_rg2.urdf from the current config (used by Isaac Sim).

    python scripts/make_urdf.py --set gripper.tcp_offset=0.21
"""

from _common import ROOT, base_parser, config_from_args

from grasp_rl.urdf import write_urdf


def main():
    p = base_parser(__doc__)
    p.add_argument("--out", default=str(ROOT / "assets" / "urdf" / "ur5_rg2.urdf"))
    cfg = config_from_args(p.parse_args())
    print("wrote", write_urdf(cfg, p.parse_args().out))


if __name__ == "__main__":
    main()
