from __future__ import annotations

import sys

from src.gui.app_interface import AppInterface


def main() -> int:
    app = AppInterface()
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
