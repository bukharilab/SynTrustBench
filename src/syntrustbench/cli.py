"""Command-line interface for SynTrustBench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ._version import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="syntrustbench",
        description=(
            "Evaluate structured synthetic clinical data across fidelity, utility, "
            "privacy, equity, and robustness."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subcommands = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = subcommands.add_parser(
        "evaluate", help="Run the complete executable tabular protocol."
    )
    for name, help_text in (
        ("real-train", "Real data used to train the generator."),
        ("real-test", "Held-out real data not used to train the generator."),
        ("synthetic", "Synthetic data to evaluate."),
        ("config", "Versioned YAML configuration."),
        ("output", "Directory for benchmark outputs."),
    ):
        evaluate_parser.add_argument(f"--{name}", required=True, help=help_text)
    evaluate_parser.add_argument(
        "--synthetic-replicate",
        action="append",
        default=[],
        help=(
            "Optional additional synthetic CSV generated under another generator seed. "
            "Repeat this option for multiple replicates."
        ),
    )
    evaluate_parser.add_argument(
        "--quiet", action="store_true", help="Print only the machine-readable final summary."
    )

    validate_parser = subcommands.add_parser(
        "validate", help="Validate a submission without calculating benchmark metrics."
    )
    for name, help_text in (
        ("real-train", "Real data used to train the generator."),
        ("real-test", "Held-out real data."),
        ("synthetic", "Synthetic data to validate."),
        ("config", "Versioned YAML configuration."),
    ):
        validate_parser.add_argument(f"--{name}", required=True, help=help_text)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "validate":
            from .validation import load_submission

            submission = load_submission(
                args.real_train, args.real_test, args.synthetic, args.config
            )
            payload = {
                "status": "valid",
                "dataset": submission.config.name,
                "rows": {
                    "real_train": len(submission.real_train),
                    "real_test": len(submission.real_test),
                    "synthetic": len(submission.synthetic),
                },
                "warnings": submission.warnings,
            }
            print(json.dumps(payload, indent=2))
            return 0

        from .engine import evaluate

        result = evaluate(
            args.real_train,
            args.real_test,
            args.synthetic,
            args.config,
            args.output,
            synthetic_replicates=args.synthetic_replicate,
            progress=(
                None
                if args.quiet
                else lambda message: print(f"[SynTrustBench] {message}", file=sys.stderr)
            ),
        )
        payload = {
            "benchmark_gate": result.benchmark_gate,
            "dimension_status": result.summary["dimension_status"],
            "failure_flags": result.summary["failure_flags"],
            "output_dir": str(Path(args.output).resolve()),
        }
        if not args.quiet:
            print(
                "SynTrustBench completed. Pass/Conditional/Fail values are provisional "
                "benchmark decisions, not certifications.",
                file=sys.stderr,
            )
        print(json.dumps(payload, indent=2))
        return 0
    except Exception as exc:  # CLI boundary: concise error, non-zero status
        print(f"syntrustbench: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
