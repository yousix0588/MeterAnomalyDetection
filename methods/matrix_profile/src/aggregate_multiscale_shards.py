import argparse

from anomalies.multiscale.aggregate import aggregate_shards


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate Spartan Matrix Profile shards")
    parser.add_argument("--shards-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--expected-shards", type=int, default=100)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    paths = aggregate_shards(args.shards_dir, args.output_dir, args.expected_shards)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
