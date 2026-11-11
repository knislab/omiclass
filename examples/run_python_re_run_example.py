"""Apply an actual completed core run; an HTML-only folder is not a model."""
import argparse
from pathlib import Path

from api import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('expression', 'metadata', 'run-dir'):
        parser.add_argument('--'+name, required=True, type=Path)
    args = parser.parse_args()
    for name in ('leaderboard.csv', 'config.freeze.yaml'):
        if not (args.run_dir / name).is_file():
            parser.error('Missing saved-run prerequisite: '+name)
    if (args.run_dir / 'foundation.json').is_file():
        parser.error('Foundation runs require api.predict with a separate output directory')
    result = run(args.expression, output_dir=args.run_dir, meta_path=args.metadata)
    print(result.out_dir)


if __name__ == '__main__':
    main()
