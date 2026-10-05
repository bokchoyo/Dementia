# Reproducing the project

This guide describes the code currently in the repository. The commands below are prospective recipes, not the verified command that produced the historical CSV. Run all commands from the repository root.

## 1. Clone and install

```console
git clone https://github.com/bokchoyo/Dementia.git
cd Dementia
git lfs install
git lfs pull
python -m venv .venv
```

Activate on Windows PowerShell with `.venv\Scripts\Activate.ps1`, or on macOS/Linux with `source .venv/bin/activate`, then:

```console
python -m pip install -r requirements.txt
python -m nltk.downloader punkt averaged_perceptron_tagger
```

Python 3.10.11 and the versions recorded in [requirements-observed.txt](../requirements-observed.txt) were present in the original project environment metadata. That file is an inventory, not a validated clean-environment lock. `requirements.txt` now includes the previously omitted direct dependencies `torch` and `matplotlib`, with the locally observed versions. CPU/GPU package availability must be checked on the target machine. No fresh installation of the full training stack was verified during this documentation pass.

## 2. Required assets

| Asset | Expected relative location | Status and use |
|---|---|---|
| Parsed dementia records | `data/dataset/dimentia/all_test_parsed_filtered_inference_ready.jsonl` | Present locally and tracked; 2,222 records. Original corpus/source and access terms still need attribution. |
| Coherence checkpoint | `data/result/gcdc/model/checkpoint_4/pytorch_model.bin` | Present locally and tracked with LFS. Training provenance and associated paper not verified. |
| Relation vocabularies | `data/parser/pdtb3/exp/l1/labels.txt`, `data/parser/pdtb3/imp/l1/labels.txt` | Required even by the sentence training route because `SentDataset` constructs relation arrays. |
| Explicit/implicit parser checkpoints | `data/parser/pdtb3/{exp,imp}/l1/pytorch_model.bin` | LFS assets; required only to reparse raw input. Each is approximately 1.42 GB. |
| Explicit connective vocabulary | `data/parser/pdtb3/exp/l1/conn_list.txt` | Used by the parser. |
| RoBERTa base model/tokenizer | Hugging Face identifier `roberta-base` | Downloaded by the trainer unless already cached. Frozen during these experiments. |
| RoBERTa large model/tokenizer | Hugging Face identifier `roberta-large` | Used by the raw-input parser. |
| Full GCDC coherence training data | `data/dataset/gcdc/combined/1/train_full.json` | Not present in the inspected checkout; required for coherence training/CV, not binary training from an existing checkpoint. |
| GloVe embeddings | Local `embeddings/` | Not needed by the documented sentence-transformer binary route. The 5.65 GB local file is not part of this handoff. |

Binary checkpoints must match the same coherence architecture, label order, and head dimensions used when saving. The current scripts use `strict=False` when loading weights, so success alone does not prove all intended weights matched. Inspect missing/unexpected keys before substituting other checkpoints.

## 3. Inspect before training

```console
python scripts/audit_results.py
python scripts/audit_dataset.py data/dataset/dimentia/all_test_parsed_filtered_inference_ready.jsonl
python -m unittest discover -s tests -v
python train_relcoh_binary_hyperparam_tuned.py --help
python train_relcoh_10fold_cv_dementia_binary_topic80_20_balanced.py --help
```

The audit scripts need only Python's standard library. Training `--help` imports the machine-learning dependencies, so it requires the full environment. Check [VERIFICATION.md](VERIFICATION.md) for exactly what was tested.

## 4. Balanced topic split experiment

```console
python scripts/run_experiment.py balanced --output runs/balanced-001 --epochs 30
```

This recipe uses weighted loss, learning rate 0.001, a 20% test split within topic, and validation carved from 12.5% of the remaining training portion. The resulting proportions are approximately 70% training / 10% validation / 20% testing. This explicitly enables validation; the underlying historical script defaults to no validation (`binary_dev_ratio=0`). Topic proportions involve integer rounding.

## 5. Learning-rate search experiment

```console
python scripts/run_experiment.py tuned --output runs/tuned-001 --epochs 30
```

This uses eight training folds, validation fold 9, and test fold 10. Rates are `1e-5,3e-5,1e-4,3e-4,1e-3`. It selects the lowest validation-loss checkpoint across rates and epochs and tests that checkpoint; it does not refit on training plus validation. Both training modes now default to balanced loss. Append `--loss-weighting none` for an explicit unweighted run and label it as a different experiment.

To print either command without execution, append `--dry-run`. Override asset paths with `--data PATH` and `--checkpoint PATH`. Each actual run writes `run_manifest.json` and `console.log` to its chosen run directory; the manifest records the command, Python/package versions, source/input hashes, start/end times, and exit code. Output directories must be empty to avoid selecting stale checkpoints. Training logs are written to disk; inspect `console.log` for progress.

The trainer additionally creates prepared split files, prediction JSONL, per-topic summaries, and checkpoints under `dementia/fast_flat_transformer_sent+roberta-base/binary_dementia/`. Tuning adds learning-rate curves and `hyperparameter_tuning/binary_hyperparameter_tuning_summary.json`.

## 6. Check participant separation

```console
python scripts/audit_dataset.py PATH_TO_TRAIN.jsonl PATH_TO_VALIDATION.jsonl PATH_TO_TEST.jsonl --output runs/participant_audit.json --fail-on-overlap
```

This audits actual generated splits using `(source_group, patient_id)` and exits nonzero if a participant overlaps or IDs are missing. Verify that participant IDs persist across visits. The current trainers split records, not participants. Their outputs cannot establish generalization to unseen people until a participant-disjoint protocol is implemented and evaluated. Do not relabel older results as participant-disjoint.

## 7. Optional raw-input parsing and coherence training

```console
python pdtb/pdtb_parser_single_file_updated.py --input_file PATH_TO_RAW.jsonl --output_file PATH_TO_PARSED.jsonl --parser_dir data/parser/pdtb3 --keep_text
```

Inspect the parser's `parse_file` for accepted JSON fields before supplying another corpus. Raw-source conversion and the filtering step preceding the provided parsed file are not fully represented by a verified pipeline in this checkout. The Stanza path is an optional legacy branch; the parser's current entry point uses `use_stanza=False`.

Coherence-model CV, once the GCDC asset is available:

```console
python train_relcoh_binary_hyperparam_tuned.py --do_cv --cv_file data/dataset/gcdc/combined/1/train_full.json --train_file data/dataset/gcdc/combined/1/train_full.json --dataset gcdc --output_dir runs/coherence-cv --model_type transformer_sent --model_name_or_path roberta-base --label_list low,medium,high
```

## 8. Sharing and completing the historical record

The GitHub collaborator invitation for `hyejuj` was visible as pending on October 5, 2026. Acceptance is performed by the recipient. Share the repository's README and report links for the walkthrough.

To make the historical CSV fully reproducible, add the exact run command, Git revision, binary checkpoint hash, split membership, per-record predictions, dataset acquisition/preprocessing provenance, and the missing EntRelCoh implementation. Keep each result tied to an experiment manifest. The report explicitly labels these unresolved links.
