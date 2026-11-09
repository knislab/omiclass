# Promoted manuscript source

The hashes below identify the read-only source implementations used for Step 2.

```json
{
  "origin": "Manuscript source inspected read-only during Step 2",
  "sha256": {
    "experiments/article_foundation_heads/train.py": "4982c5ee33869c143d6258e7ada00590a62bbb57c7eab349cd32a57f5e034033",
    "final_results_runs/test_bulkrnabert/standalone/models.py": "ac229242c774944360f03d8c2c02eb88e0ecfff5c4089bae7d15226af3780f09",
    "experiments/article_input_correction/cellfm_fixed_views.py": "964577b12e0c6bdd580bebd7dc0435317439ce6a599604ada61cdd32600a1619",
    "experiments/article_input_correction/features.py": "192b5b2b52eddaa9a4b5bf8d747cbdbef38f2c21109ec46e45800bd71ac68e0c",
    "experiments/non_cancer_als/foundation_models/config/cancer_protocols.json": "e53d0b400fb7a8101e57f03988d214b7579067907048dee2feb068231e855667"
  },
  "adaptations": {
    "article_head": "Callable train(args), reusable exact architecture factory, NPZ/NPY token reader. Optimizer/loss/shuffling/epochs unchanged.",
    "bert_head": "Optional input width replaces the historical width-only FunctionType adapter. Optimizer/inner split/early stopping/SMOTE unchanged.",
    "cellfm_corrected": "Instance-local token/CLS observer and public JSON writer. Original normalization, view selection, corrected scalar replacement, finite checks unchanged.",
    "correction_compat": "Exact original zero-gene restoration and corrected-value floor functions."
  }
}
```
