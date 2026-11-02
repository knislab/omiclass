#!/usr/bin/env python3
"""Convenience entry point for external BulkRNABert SVM/MLP experiments."""
import sys
from final_results_runs.test_bulkrnabert.standalone.run import main

if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, FileExistsError, KeyError, ImportError) as exc:
        print(f'BulkRNABert standalone: {exc}', file=sys.stderr)
        raise SystemExit(2)
