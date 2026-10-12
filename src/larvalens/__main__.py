"""Command line interface: python -m larvalens <command>."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence


def _version(_: argparse.Namespace) -> None:
    from larvalens import __version__

    print(__version__)


COMMANDS: dict[str, tuple[str, Callable[[argparse.Namespace], None]]] = {
    "version": ("Print the package version.", _version),
}


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with one subcommand per pipeline step."""
    parser = argparse.ArgumentParser(prog="larvalens", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, (help_text, func) in COMMANDS.items():
        sub.add_parser(name, help=help_text).set_defaults(func=func)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Run the CLI."""
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
