from __future__ import annotations

import sys
from typing import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] == "gui":
        if args and args[0] == "gui":
            args = args[1:]
        if args:
            raise SystemExit("The GUI command does not accept positional arguments.")
        try:
            from .app import main as gui_main
        except ImportError as exc:
            if exc.name and exc.name.startswith("PyQt6"):
                raise SystemExit(
                    'GUI dependencies are not installed. Install `coopinfer[gui]`.'
                ) from exc
            raise
        gui_main()
        return 0

    if args[0] == "solve":
        args = args[1:]
    elif args[0] in {"-h", "--help"}:
        print(
            "usage: coopinfer [gui] | coopinfer solve CONFIG [CONFIG ...] [options]\n"
            "\n"
            "With no command, CoopInfer launches the GUI. A config path as the\n"
            "first argument is shorthand for the headless `solve` command."
        )
        return 0

    from .cli import main as cli_main

    return cli_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
