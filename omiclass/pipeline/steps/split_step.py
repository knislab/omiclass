"""Own one immutable labelled population and train/holdout split per run."""
from pathlib import Path
import pandas as pd

from omiclass.utils.io import load_labels
from omiclass.utils.pipeline import make_or_load_split
from omiclass.utils.splits import apply_split, save_split, load_split


class SplitStep:
    name = 'sample_partitions'

    def should_run(self, ctx):
        return ctx.opts.mode != 'clean_and_predict'

    def run(self, ctx):
        frame = ctx.cache['full_expr']
        y = load_labels(ctx.cfg.project.meta_path, ctx.cfg.project.cluster_column, cfg=ctx.cfg)
        eligible = frame.columns[frame.columns.isin(y.dropna().index)]
        X, y = frame.loc[:, eligible].T, y.loc[eligible]
        codes = sorted(y.unique())
        mapping = {value: i for i, value in enumerate(codes)}
        names = ctx.cfg.project.classes_names
        if names and all(0 <= int(value) < len(names) for value in codes):
            ctx.cfg.project.classes_names = [names[int(value)] for value in codes]
        elif not names:
            ctx.cfg.project.classes_names = [str(value) for value in codes]
        y = y.map(mapping).astype(int)
        bc = ctx.cfg.batch_correction
        reference = bc.algorithm in {'frozen_quantile', 'reference_ruv', 'fsva_exact'}
        explicit = getattr(ctx.opts, 'split_json', None)
        if reference and bc.training_samples_path is not None and not explicit and not ctx.opts.reuse_split:
            from omiclass.batch.reference_pipeline import read_ids
            train = read_ids(bc.training_samples_path)
            split = dict(train_index=train, test_index=[s for s in eligible if s not in set(train)],
                         method='declared_reference_training')
            apply_split(X, y, split)
            path = Path(ctx.split_out) / 'split.json'
            save_split(split, path)
        else:
            *_, path, _created = make_or_load_split(X, y, ctx.split_out, ctx.cfg,
                ctx.opts.reuse_split, split_json=explicit)
            split = load_split(path)
        if reference and bc.training_samples_path is not None:
            from omiclass.batch.reference_pipeline import read_ids
            if set(read_ids(bc.training_samples_path)) != set(split['train_index']):
                raise ValueError('Reference fitting IDs must equal the classifier training partition')
        ctx.cache.update(split=split, split_json=path, labels=y, label_mapping=mapping,
                         raw_expression=frame.copy(), eligible_ids=eligible.tolist())
