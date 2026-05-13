# Example data

This folder contains small example files used by the example scripts.

## Files

| File | Description |
|---|---|
| `raw_counts.csv` | Gene expression matrix. Rows are genes/features and columns are samples. |
| `patients_meta.csv` | Metadata table containing sample annotations, class labels, and batch labels. |
| `raw_counts_with_external.csv` | The gene expression matrix with demo external dtaset. |
| `patients_meta_with_external.csv` | full Metadata table with demo external dtaset. |
| `gene_lengths.csv` | Gene expression matrix. Rows are genes/features and columns are samples. |

## Expected format

### Expression matrix

The expression matrix should have genes/features as rows and samples as columns.

Example:

| gene_id | sample_1 | sample_2 | sample_3 |
|---|---|---|---|
| GENE1 | 10 | 20 | 15 |
| GENE2 | 5 | 0 | 8 |
| GENE3 | 12 | 7 | 3 |


### Metadata file

The metadata file should contain one row per sample and include the columns used by the run, such as the classification column and batch column.

Example:
| sample_id | class | batch | age | sex |
|---|---|---|---|---|
| sample_1 | A | cohort1 | 45 | M |
| sample_2 | B | cohort1 | 72 | M |
| sample_3 | A | cohort2 | 53 | F |

### Holdout file

The holdout file should contain one sample ID per line.

Example:

```text
sample_1
sample_3
```

## Notes

Column names in the example scripts must match the column names in the metadata file.

For example, if the script uses:

```python
"class_col": "ighv_mut",
"batch_column": "cohort",
```

then `patients_meta.csv` must contain columns named `ighv_mut` and `cohort`.