# Stage 6: CellFM sampled-position decision

The positional-alignment investigation originally stopped Stage 6 for the requested decision. The user subsequently approved retaining the existing full-token configurations as an explicitly documented **sampled-position protocol**. That approval resolves the implementation stop; it does not establish a fixed gene axis. See the [completed Stage 6 report](step3-stage6-report.md) for the subsequent corrections and validation.

## Confirmed limitation

There are two different alignment questions:

1. Replaying a particular sample/view with the same selection seed, input values and vocabulary. Stage 6 corrects new single-view extraction to use the existing sample/view-specific seed mechanism, also used by multiview extraction.
2. Assigning the same gene identity to a flattened position across different selected sets. Sorting establishes relative order inside each selected set, but cannot establish this stronger guarantee without changing membership or dimensions.

Two valid, sorted 2048-gene selections can be `[1, ..., 2048]` and `[2, ..., 2049]`. Their first positions have different identities, and their union exceeds 2048 genes. A common one-gene-per-position layout covering that union cannot fit in the unchanged 2048-position input. The existing article head applies LayerNorm, a shared projection to 8, GELU/masking, flattening, then the existing `[64]` dense head. The dense weights depend on position; the head consumes tokens and masks rather than explicit gene-ID alignment.

Consequently, token positions are **not guaranteed to represent the same gene across samples or views**. This limits positional interpretation and may affect generalization when selected sets or their distributions change. Reproducible selection and valid masks do not remove that limitation, and passing synthetic tests does not establish biological generalization.

## Approved resolution and scope

Both CellFM `tokens` + `mlp` configurations (one and twenty views) remain in the 104-entry catalogue. Their architecture, input dimensions, selection rule, per-view loss and probability averaging remain intact. No pooling replacement, common-vocabulary padding, restriction to a common selected set or configuration removal is introduced. New single-view runs necessarily realize different draws when switching from the old cohort-wide stream to the approved sample/view seeds; historical bundles explicitly replay the old stream.

New caches identify CellFM as `sampled_positions`. Validation checks gene-ID bounds/uniqueness, valid-prefix masks, padding, selected order and saved sampling/vocabulary/view contracts, while allowing different valid selected memberships. It does not claim a globally fixed gene order. Historical models without this metadata retain their original head and cache-loading path; unavailable historical metadata cannot be reconstructed.

The four cases in `tests/test_step3_stage6_alignment_review.py` retain the sorted-set counterexample for one/twenty views, demonstrate order sensitivity of the unchanged head on small synthetic tensors, and verify historical sampled-position prediction after serialization. They document the retained limitation rather than certify fixed positional alignment. No real checkpoint or manuscript result was used.
