"""Promoted original corrected-BERT policies; deliberately unchanged for Step 2."""
import numpy as np
from omiclass.foundation.io import write_json

class IncompatibleInput(ValueError):
    pass


def restore_bulkrnabert_zero_genes(before, corrected):
    """Restore only proven-zero vocabulary rows, on this call's retained samples.

    ``before`` is the fixed-vocabulary log10(TPM+1) matrix supplied to Limma.
    Check every retained column before downstream partition/external slicing.
    """
    if (not before.index.is_unique or not corrected.index.is_unique
            or not before.columns.is_unique or not corrected.columns.is_unique):
        raise IncompatibleInput('BulkRNABert restoration requires unique gene and sample IDs')
    if corrected.empty or len(corrected.columns.difference(before.columns)):
        raise IncompatibleInput('BulkRNABert correction has empty or unknown retained samples/genes')
    if len(corrected.index.difference(before.index)):
        raise IncompatibleInput('BulkRNABert correction contains genes outside the input vocabulary')
    missing = before.index[~before.index.isin(corrected.index)]
    if len(missing):
        values = before.loc[missing, corrected.columns].to_numpy(dtype=float)
        safe = (np.isfinite(values) & (values == 0.0)).all(axis=1)
        if not safe.all():
            unsafe = missing[~safe].tolist()
            raise IncompatibleInput(
                'Cannot restore BulkRNABert genes: nonzero or nonfinite values before '
                f'correction on retained samples: {unsafe[:10]}')
    return corrected.reindex(before.index, fill_value=0.0), missing.tolist()


def clip_bulkrnabert_corrected(frame, directory):
    """Floor corrected log expression at zero before fixed-bin tokenization."""
    values = frame.to_numpy()
    if not np.isfinite(values).all():
        raise IncompatibleInput('BulkRNABert corrected input contains nonfinite values')
    negative = values < 0
    clipped = frame.clip(lower=0.0)
    write_json(directory / 'clipping.json', dict(
        policy='corrected_log10_tpm_floor_zero_v1', lower_bound=0.0,
        clipped_entries=int(negative.sum()), total_entries=int(values.size),
        clipped_fraction=float(negative.mean()), minimum_before=float(values.min()),
        minimum_after=float(clipped.to_numpy().min())))
    return clipped
