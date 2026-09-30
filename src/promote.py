"""Promote the `staging` model version to `production` (with safety checks)."""

from __future__ import annotations

import argparse
import sys

from src import registry
from src.config import load_config, tracking_uri


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--force", action="store_true", help="skip validation and regression checks")
    args = p.parse_args()
    cfg = load_config()
    client = registry.connect(tracking_uri())
    try:
        res = registry.promote(client, cfg, force=args.force)
    except registry.PromotionError as exc:
        print(f"PROMOTION REFUSED: {exc}", file=sys.stderr)
        sys.exit(1)
    msg = f"Promoted version {res['promoted']} to production"
    if res["archived"]:
        msg += f"; archived previous production version {res['archived']}"
    print(msg)


if __name__ == "__main__":
    main()
