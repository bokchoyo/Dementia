# Verification record

October 5, 2026. This is a documentation and reproducibility audit; no new model training was performed.

## Completed checks

- Compared current source against the supplied Word-contained balanced training script and earlier model files.
- Extracted the Word-contained code to Python and parsed all Python source files for syntax validity.
- Preserved the reported CSV byte-for-byte and recomputed binary metrics; topic confusion matrices reconcile with the overall matrix.
- Audited 2,222 local records: no duplicate record IDs, missing patient IDs, or empty sentence lists.
- Recreated the current default topic and fixed-fold splits using NumPy 1.26.4 and scikit-learn 1.3.2, without loading models; recorded participant overlap.
- Ran the review-tool regression tests and both launcher dry-run recipes.
- Checked repository Markdown file links and complete Python-file coverage in the code map.

The audit execution environment was Python 3.12.14. Split-audit dependencies were installed in an isolated temporary workspace; the original project environment was not modified.

## Verification limits

The existing project virtual environment references Python 3.10.11, but its executable could not be launched by the current execution environment. A fresh full machine-learning environment, model imports, GPU behavior, pretrained checkpoint key compatibility, parser execution, and end-to-end training/prediction were not verified. This limitation is explicit rather than describing source checks as a successful training run.

The provided CSV has no row-level predictions or run manifest. Its original metrics cannot be tied conclusively to a particular source revision, checkpoint, or split. Recreated split counts matching the table do not establish provenance. Existing model behavior was retained, including the limitations described in report section 6.

## Recheck commands

```console
python -m unittest discover -s tests -v
python scripts/audit_results.py
python scripts/audit_dataset.py data/dataset/dimentia/all_test_parsed_filtered_inference_ready.jsonl
python scripts/audit_default_splits.py
python scripts/run_experiment.py balanced --output runs/balanced-check --dry-run
python scripts/run_experiment.py tuned --output runs/tuned-check --dry-run
```

Training-specific checks after installing the full environment:

```console
python train_relcoh_binary_hyperparam_tuned.py --help
python train_relcoh_10fold_cv_dementia_binary_topic80_20_balanced.py --help
```

These help commands verify imports and argument parsing only, not checkpoint compatibility or training correctness. Save logs and a manifest from an actual run before claiming full reproduction.
