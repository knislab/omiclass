"""Observed protocol combinations; not a Cartesian product of arbitrary encoders."""
from itertools import product


def configurations():
    rows = []
    for model in ["bulkformer", "cellfm_1view", "cellfm_20view", "bulkrnabert"]:
        for pool, rep, head in product(["mean", "max_mean_median"] if model == "bulkformer" else ["mean"],
                ["full", "supervised", "pca", "supervised_pca"], ["omiclass", "lr", "svm", "mlp", "mlp_smote"]):
            rows.append(dict(model=model, pooling=pool, representation=rep, classifier=head,
                             evidence="saved manuscript protocol", diagnostic=False))
        rows.append(dict(model=model, pooling="mean", representation="tokens", classifier="mlp",
                         evidence="validated experimental protocol; no completed full-scale fits" if model == "bulkrnabert" else "saved manuscript protocol", diagnostic=False))
    return rows
