"""Shared CLI helpers: --config / --set handling and import path setup."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grasp_rl.config import load_config  # noqa: E402


def base_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=None, help="extra YAML merged over configs/default.yaml")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE",
                   help="override a config value, e.g. --set reward.success=20 (repeatable)")
    p.add_argument("--backend", choices=["simple", "isaac"], default=None)
    return p


def config_from_args(args):
    overrides = list(args.overrides)
    if args.backend:
        overrides.append(f"backend={args.backend}")
    return load_config(args.config, overrides)
