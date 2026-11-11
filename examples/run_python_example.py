"""Train the existing marker workflow on user-supplied, prepared expression."""
import argparse
from pathlib import Path

from api import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('expression', 'metadata', 'markers', 'output'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--class-column', default='class')
    args = parser.parse_args()
    if (args.output / 'leaderboard.csv').exists():
        parser.error('Output is an existing trained run; use the prediction example')
    result = run(args.expression, output_dir=args.output, meta_path=args.metadata,
        classification_column=args.class_column, markers_path=args.markers,
        raw_count=False, log_transform=False, feature_selection_enabled=False, run_de=False,
        **{'batch_correction.algorithm': 'none'})
    print(result.out_dir)


if __name__ == '__main__':
    main()
