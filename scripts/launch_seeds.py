"""Start, watch and stop detached training runs: `launch`, `status` or `stop`."""

from __future__ import annotations

import sys

from worm.launch import main

if __name__ == "__main__":
    sys.exit(main())
