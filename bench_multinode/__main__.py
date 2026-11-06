"""Independent Ray-backed benchmark and simulator comparison entry point."""

from __future__ import annotations

import argparse


def main() -> int:
    parser = argparse.ArgumentParser(prog="bench_multinode")
    sub = parser.add_subparsers(dest="cmd", required=True)
    from .core.runner_multinode import register_args as register_run
    from .core.validate_multinode import register_args as register_validate

    register_run(sub.add_parser("run", help="Replay a workload on an existing Ray cluster"))
    register_validate(sub.add_parser("validate", help="Compare a completed run with simulation"))
    args = parser.parse_args()
    if args.cmd == "run":
        from .core.runner_multinode import run
    else:
        from .core.validate_multinode import run
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
