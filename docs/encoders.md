# External encoders and checkpoints

All 104 configurations remain registered. This guide describes existing requirements, not a downloader or a promise that upstream HEAD is compatible. Supply authorized source/checkpoint assets and record their exact revision and content hashes. Never install into or write to a running research environment. No third-party code, vocabulary, annotation or checkpoint is redistributed with this candidate.

## BulkFormer

The primary upstream project is [KangBoming/BulkFormer](https://github.com/KangBoming/BulkFormer). Its current releases may differ from the historical 147M implementation used here. The previous source gitlink recorded `5bcf5b99e799cd55f41e4d6fe8200b2cdf3d9589`; public retrieval/compatibility of that revision has not been certified. Obtain the matching authorized source and its dependency manifest before claiming encoder execution.

Set `bulkformer_root_dir` and supply the existing expected files, or override their existing configuration paths:

- `data/G_tcga.pt`, `data/G_tcga_weight.pt`, `data/esm2_feature_concat.pt`
- `model/bulkformer_147M.pt`
- `data/bulkformer_gene_info.csv`, `data/gene_length_df.csv`, `data/interested_gene_list.pt`

The adapter imports `utils.BulkFormer` and `model.config`, and requires Torch Geometric's compatible SparseTensor dependency. Use the checkpoint's existing dimensions and interested-gene order. `max_mean_median` remains the historical elementwise max + mean + median sum. Do not substitute a new upstream model, pooling scheme or vocabulary.

## CellFM

The primary [CellFM repository](https://github.com/biomed-AI/CellFM) documents the original project. This framework requires the particular compatible PyTorch port exposing `model.py`, `layers/utils.py`, `Config_80M`, `Prepare`, `Cell_FM`, and `model.net.encode`. A cited CellFM-torch URL was not retrievable during this preparation; its exact authorized source and dependencies remain an owner decision. The remaining legacy tutorial fragments were not that runtime and are excluded.

Configure `cellfm_root_dir`, `cellfm_checkpoint_path`, `cellfm_gene_info_path`, `cellfm_vendor_dir`, and `cellfm_device`. The conventional resource locations are `CellFM/CellFM-torch`, `CellFM/weights/CellFM_80M_weight.ckpt`, and the port's `csv/expand_gene_info.csv`. These are user-provided paths, not files included in the package. The existing protocol requires 2048 sampled gene slots and width 1536. Do not convert or replace the architecture as part of setup.

The supplied vocabulary must be unique and match the checkpoint. New foundation input preserves literal symbols such as `AC118549.1`; CellFM's existing preparation sums repeated genes and mapping collisions. Ensembl mapping requires the permitted matching annotation, currently excluded pending review. One/twenty-view aliases use the existing sample/view seed mechanism. Full-token positions are sampled positions, with no guaranteed common gene identity across samples/views; positional interpretation and generalization are limited accordingly.

## BulkRNABert

The primary upstream source is [InstaDeep multiomics-open-research](https://github.com/instadeepai/multiomics-open-research). The previous source gitlink recorded `625e8165abd1dfab19dea7b5e9be7c67571aa5b1`; check that the supplied implementation and dependency manifest match. Upstream documents installation of its own package in a dedicated environment. Do not install the JAX stack into the core environment merely to make imports succeed.

Set `bulkrnabert_root_dir`, `bulkrnabert_python_executable`, `bulkrnabert_checkpoint_directory`, `bulkrnabert_vocabulary_path` and `bulkrnabert_device` explicitly. The existing selected model is `bulk_rna_bert_gtex_encode`, using its `config.json` and `params.joblib`. Supply its ordered 19,062-gene vocabulary (the default path is `data/bulkrnabert/common_gene_id.txt` relative to the source root); verify the actual supplied file rather than substituting a similarly named list. The layer-4 representation is 256 wide, with four transformer layers. Other checkpoints are not interchangeable.

Provide unlogged TPM with `foundation.input_scale='tpm'`, or raw counts with verified gene lengths/an explicit per-cohort TPM file. Existing version removal/vocabulary alignment is retained. Foundation preprocessing logs once; the worker bypasses a second log and lets the official tokenizer normalize/bin once. Corrected input retains its recorded negative-value zero floor; out-of-range bins fail rather than upper-clip. The encoder retains JAX key 0. Full tokens retain all vocabulary positions; this configuration has no completed full-scale manuscript fits.

## Validation and replay

Only locally supplied assets are hashed/loaded; no encoder training, checkpoint download or dependency repair occurs automatically. New bundle identities bind source, checkpoint/vocabulary and configuration. Preserve the saved absolute layout for historical replay; relocating identical assets is currently restricted. Configuration overrides cannot silently change a fitted encoder/protocol.

Run isolated small real-checkpoint and CPU/GPU smoke checks when the exact permitted assets are available. Mocked encoders verify API wiring, fitting boundaries and serialization, not the biological identity of arbitrary assets. Missing/unknown versions and incompatible layouts must remain explicit failures. Until source/resource decisions and real execution checks are complete, the candidate must not be advertised as certified end-to-end execution of every configuration.
