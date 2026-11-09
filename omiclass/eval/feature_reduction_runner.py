"""Record why the historical diagnostic cannot provide valid evaluation."""
import json
from omiclass.eval.feature_reduction import DISABLED_REASON


class FeatureReductionStep:
    name = 'feature_reduction'

    def run(self, ctx):
        status = dict(status='disabled', reason=DISABLED_REASON)
        directory = ctx.out_dir / 'feature_reduction'
        directory.mkdir(parents=True, exist_ok=True)
        (directory / 'status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
        ctx.cache['feature_reduction_status'] = status
