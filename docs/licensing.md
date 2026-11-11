# Licensing and resource review

The framework software license has **not** been selected. Packaging deliberately omits a license declaration, and the empty legacy LICENSE file is not included in candidate distributions. Legacy build tools may emit `License: UNKNOWN`; this is not a selected license. Version `0.0.0.dev0` is an unpublished packaging placeholder. The owner must approve the software license and public release; preparing or testing this candidate does not grant a license or redistribute it.

The following existing files remain in the private working copy and are excluded from the publication snapshot, wheel, source archive and Docker context pending source/permission verification:

- `omiclass/data/gencode_GRCh37_hg19.csv`
- `omiclass/data/gencode_v34_GRCh38_hg38.csv`
- `omiclass/data/filtered_surface_list.txt`
- Historical `examples/data/`, screenshots and generated reports.

No replacement annotation or synthetic runtime gene mapping is shipped. Core count workflows must receive verified gene lengths explicitly. Bundled annotation fallback, CellFM Ensembl-to-symbol mapping and surface-marker resource paths cannot be certified from the candidate without approved resources. Symbol-based CellFM input still follows its existing matching procedure; that is not a substitute claim for verified annotation coverage.

Options for the owner are to provide the source/version, applicable permission/attribution record and approved files; approve an authorized redistribution after review; or keep these resources external and decide how to supply them using a separately reviewed compatibility-preserving interface. The candidate does not choose among those scientific/resource arrangements automatically.

Third-party encoder repositories, vendored binaries, converted checkpoints, vocabularies and model weights are not bundled. Their source/checkpoint-specific terms and attribution must be verified independently. The exact CellFM PyTorch port and historical BulkFormer source revision also require acquisition/compatibility confirmation. Do not replace them with a newer architecture or assume a different checkpoint is equivalent.

Research scripts, notebooks, cluster launchers and obsolete HTML reports are quarantined outside the candidate. Historical source-adaptation hashes remain in `omiclass/foundation/SOURCE_PROVENANCE.md`. No inherited Git history or Git remotes are included in the source snapshot, and no shared history is rewritten. Public publication remains blocked on these decisions and remaining validation.
