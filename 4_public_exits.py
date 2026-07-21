import argparse
import importlib

import pnadc_superpc.transitions as transitions_module


def main():
    parser = argparse.ArgumentParser(description="Analyze public-sector exits in the PNADC panel.")
    parser.add_argument("--year", type=int, default=None, help="Single year (shorthand for --years).")
    parser.add_argument("--years", nargs="+", type=int, default=None, help="One or more years to analyze together.")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--use-sample-inputs", action="store_true")
    args = parser.parse_args()

    years = args.years if args.years is not None else ([args.year] if args.year is not None else None)
    if not years:
        parser.error("Specify --year YYYY or --years YYYY [YYYY ...]")

    importlib.reload(transitions_module)

    result = transitions_module.run_public_exit_analysis(
        years=years,
        sample_frac=args.sample_frac,
        seed=args.seed,
        tag=args.tag,
        use_sample_inputs=args.use_sample_inputs,
    )
    print(f"Years analyzed: {result['years']}")
    print(f"Rows loaded: {result['rows']:,}")
    print(f"Public-origin next-quarter transitions: {result['public_transitions']:,}")
    print(f"Public -> out of labor force: {result['public_to_outlf']:,}")
    print(f"Output tag: {result['tag']}")


if __name__ == "__main__":
    main()
