# generic_classifier/pipeline/steps/define_genecode.py
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from generic_classifier.pipeline.orchestrator import RunContext
from pathlib import Path

class DefineGeneCodeStep:
    name = "define_genecode"

    def should_run(self, ctx: RunContext) -> bool:
        return ctx.opts.input_mode == "raw_counts"

    def run(self, ctx: RunContext) -> None:
        # cfg.normalization.gene_lengths_path
        gene_lengths = ctx.cfg.normalization.gene_lengths_path
        if gene_lengths is None:
            if ctx.cfg.project.version == 38:
                gene_lengths = self.get_gene_lengths(version=38)
            else:
                gene_lengths = self.get_gene_lengths()
            ctx.cfg.normalization.gene_lengths_path = gene_lengths
    
    def get_gene_lengths(self, version: int = 19) -> str:
        if version == 38:
            return Path(__file__).resolve().parents[2] / "data" / "gencode_v34_GRCh38_hg38.csv"
        else:
            return Path(__file__).resolve().parents[2] / "data" / "gencode_GRCh37_hg19.csv"


