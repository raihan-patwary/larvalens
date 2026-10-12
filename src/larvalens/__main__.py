"""Command line interface: python -m larvalens <command>."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence


def _version(_: argparse.Namespace) -> None:
    from larvalens import __version__

    print(__version__)


def _data(_: argparse.Namespace) -> None:
    from larvalens.pipeline import build_data

    build_data()


def _figures(_: argparse.Namespace) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from larvalens.pipeline import build_figures

    for path in build_figures():
        print(path.name)


def _weights(_: argparse.Namespace) -> None:
    from larvalens.pipeline import run_weight_models

    print(run_weight_models().to_string())


COMMANDS: dict[str, tuple[str, Callable[[argparse.Namespace], None]]] = {
    "version": ("Print the package version.", _version),
    "data": ("Download, verify, clean, audit and split the data.", _data),
    "weights": ("Fit the weight-from-shape models on the measurements.", _weights),
    "figures": ("Rebuild every figure from saved results.", _figures),
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
