# Dementia classification with frozen coherence features

This repository studies patient/control classification using hidden features and class probabilities from a pretrained coherence model. Start with the [research report](docs/REPORT.md), then use the [code map](docs/CODE_MAP.md) to locate each implementation and the [reproduction guide](docs/REPRODUCING.md) to run it.

## Current findings

The supplied aggregate results contain 445 binary predictions: accuracy **54.4%**, macro-F1 **0.543**, control recall **47.1%**, and patient recall **66.1%**, recomputed from the confusion matrix. Always predicting control gives **61.6% accuracy** on those same labels. These are historical reported results, not a newly reproduced run. The source CSV has a substantial Tea macro-F1 discrepancy and smaller rounding discrepancies; see [the results audit](results/audits/results_audit.json).

The current training script supports balanced loss in both training modes. Its hyperparameter search uses one fixed 8/1/1 fold allocation. The separate coherence-model `--do_cv` path performs rotating cross-validation. These should not be conflated.

## Reading order

1. [Report](docs/REPORT.md): question, data, model, experiments, results, and limitations.
2. [Code map](docs/CODE_MAP.md): every Python file linked to a report section.
3. [Reproduction guide](docs/REPRODUCING.md): assets, setup, commands, and verification limits.
4. [Original aggregate results](results/reported/final_results.csv) and [audit outputs](results/audits/).

## Quick review without model downloads

From the repository root, using Python 3.10 or later:

```console
python scripts/audit_results.py
python -m unittest discover -s tests -v
python scripts/run_experiment.py tuned --output runs/tuned-review --dry-run
```

These commands audit the supplied results, test the review tools, and print a prospective training command. They do not train or download models. Full setup and experiment commands are in [REPRODUCING.md](docs/REPRODUCING.md).

## Repository structure

```text
docs/                 Report, code map, reproducibility and asset notes
results/reported/     Original user-supplied aggregate CSV, preserved unchanged
results/audits/       Recomputed metrics, dataset counts, and split checks
scripts/              Portable launch and audit tools
tests/                Review-tool regression checks
archive/              Older model source versions, not imported by training
model.py              Current coherence models and frozen binary wrapper
module.py             Attention, pooling, and embedding components
dataset.py            Active sentence-feature dataset
pdtb/                 Discourse parsing and connective utilities
train_relcoh_*.py      Balanced baseline and hyperparameter-search entry points
```

The current `model.py` works with both training entry points. The balanced script was extracted from the supplied Word document; no `.docx` is needed to run it. Training writes generated outputs outside the versioned aggregate results. Use a new run directory for each experiment.

## Review status and provenance

The report distinguishes measured source-table values, recomputed checks, and proposed future commands. The exact historical binary checkpoint, hyperparameters, original corpus acquisition details, and EntRelCoh implementation were not supplied. No claim is made that the supplied CSV has been reproduced end to end. Existing author attribution is preserved; no redistribution license has been invented. `hyejuj` has a pending GitHub collaborator invitation as checked on October 5, 2026.
