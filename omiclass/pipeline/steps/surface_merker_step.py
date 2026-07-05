# omiclass/pipeline/steps/surface_merker_step.py
from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from omiclass.pipeline.orchestrator import RunContext
from omiclass.utils.surface_markers import fillter_surface_markers

class SurfaceMarkerStep:
    name = "surface_marker_filtering"

    def should_run(self, ctx: RunContext) -> bool:
        if ctx.opts.mode == "clean_and_predict":
            return False
        if bool(getattr(getattr(ctx.cfg, "bulkformer", None), "enabled", False)):
            return False
        return ctx.opts.surface_markers

    def run(self, ctx: RunContext) -> None:
        """
        Filter genes to only include surface markers.
        """
        try:
            gene_list = ctx.cache["gene_list"]
            filtered_genes = fillter_surface_markers(gene_list)
            ctx.cache["gene_list"] = filtered_genes
        except KeyError:
            print("[WARNING] surface_marker_filtering step enabled but no gene_list found in cache. Skipping surface marker filtering.")
        
