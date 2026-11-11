"""Validate table identifiers before pandas can rename duplicate CSV headers."""
import csv
from pathlib import Path

import pandas as pd

from omiclass.errors.errors import DataShapeError


def validate_identifiers(values, description, *, allow_duplicates=False, error_type=DataShapeError):
    if isinstance(values, pd.MultiIndex):
        raise error_type(f'{description} must be a single identifier axis')
    names = []
    for value in values:
        if pd.isna(value) or not str(value).strip() or str(value) != str(value).strip():
            raise error_type(f'{description} must be nonempty and trimmed')
        names.append(str(value))
    if not names:
        raise error_type(f'{description} must not be empty')
    if not allow_duplicates and len(names) != len(set(names)):
        raise error_type(f'Duplicate {description}')


def validate_delimited_header(path, *, error_type=DataShapeError):
    path = Path(path)
    if path.suffix.lower() not in {'.csv', '.tsv'}:
        return
    with path.open(newline='', encoding='utf-8-sig') as stream:
        header = next(csv.reader(stream, delimiter='\t' if path.suffix.lower() == '.tsv' else ','), [])
    validate_identifiers(header[1:], 'column/sample identifiers', error_type=error_type)


def validate_frame_identifiers(frame, *, allow_duplicate_features=False, error_type=DataShapeError):
    validate_identifiers(frame.index, 'row/feature identifiers',
                         allow_duplicates=allow_duplicate_features, error_type=error_type)
    validate_identifiers(frame.columns, 'column/sample identifiers', error_type=error_type)
