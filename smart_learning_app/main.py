"""Stable desktop entry point while the accepted UI is migrated into product code."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTOTYPE = ROOT / "ui_prototype"


def main() -> None:
    # The UI prototype is an accepted presentation layer. Business integration
    # remains in smart_learning_app and must not import the prototype mock service.
    sys.path.insert(0, str(PROTOTYPE))
    from main import main as launch_ui

    launch_ui()


if __name__ == "__main__":
    main()

