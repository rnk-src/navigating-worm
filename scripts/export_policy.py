"""Write `policy.json` next to a trained policy, for the browser demo.

Usage: uv run python scripts/export_policy.py results/policy_terrain
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from worm import npolicy
from worm.training import load_policy


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    folder = Path(argv[1])
    out = folder / "policy.json"
    out.write_text(json.dumps(npolicy.export(load_policy(folder)), separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
