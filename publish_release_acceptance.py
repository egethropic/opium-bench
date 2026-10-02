#!/usr/bin/env python3
"""Publish a separate release-acceptance evidence snapshot without model loading."""
import argparse
import json
from pathlib import Path

from acceptance_publication import publish


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-dir', type=Path, action='append', required=True,
        help='One immutable driver output directory; repeat to retain all attempts')
    parser.add_argument('--data-dir', type=Path, required=True, help='Managed lab data containing referenced runs/calibrations/research jobs')
    parser.add_argument('--output', type=Path, required=True, help='New publication directory; never overwritten')
    parser.add_argument('--plan', type=Path, default=Path(__file__).parent/'protocols/acceptance/release-v1.json')
    args = parser.parse_args(argv)
    print(json.dumps(publish(args.stage_dir, data_dir=args.data_dir, output_dir=args.output, plan_path=args.plan), indent=2))


if __name__ == '__main__': main()
