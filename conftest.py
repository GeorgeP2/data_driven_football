"""Make every ``<project>/src`` importable during tests.

Each top-level project folder has a package with a unique name, so they can
all share one ``sys.path`` without collisions.
"""

import sys
from pathlib import Path

for src in sorted(Path(__file__).parent.glob("*/src")):
    if src.parent.name != "src":
        sys.path.insert(0, str(src))
