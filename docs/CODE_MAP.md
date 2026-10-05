# Report to code map

Read [REPORT.md](REPORT.md) first. All paths below are relative to the repository root. “Historical/support” means the file is retained but is not called by the documented binary training route.

| Report section | File | Entry points and role |
|---|---|---|
| 2 Data | [dataset.py](../dataset.py) | Active `SentDataset`; merges sentences, prepares masks and cached encoder vectors. `LSTMDataset` is historical/support. |
| 2 Data | [pdtb/pdtb_parser_single_file_updated.py](../pdtb/pdtb_parser_single_file_updated.py) | `parse`, `parse_file`, `parse_doc`; optional raw JSONL discourse parsing. |
| 2 Data | [pdtb/pdtb_model.py](../pdtb/pdtb_model.py) | Parser `BaseClassifier` for explicit/implicit relations. |
| 2 Data | [pdtb/pdtb_utils.py](../pdtb/pdtb_utils.py) | Sentence splitting, connective candidates, argument spans, and parser batching. |
| 2 Data | [pdtb/conn_filter.py](../pdtb/conn_filter.py) | Connective-specific filters called by parser utilities. |
| 2 and 6 | [scripts/audit_dataset.py](../scripts/audit_dataset.py) | Dataset counts and participant overlap across supplied split files; exports aggregates only. |
| 3 Model | [model.py](../model.py) | Current `BaseClassifer`, `SentTransformer`, feature extraction, frozen binary wrapper; `FusionClassifier` retained but not selected by these trainers. |
| 3 Model | [module.py](../module.py) | Transformer attention, document pooling, embeddings, and fusion components. |
| 3 Model | [utils.py](../utils.py) | Label loading, embedding construction, and supporting relation masks. |
| 3 Historical/support | [task_dataset.py](../task_dataset.py) | Alternative relation/fusion dataset. Not imported by the two binary trainers. |
| 3 Historical/support | [archive/model_origin.py](../archive/model_origin.py) | Original model reference; missing `Transformer_Encoder` import. |
| 3 Historical/support | [archive/model_v2.py](../archive/model_v2.py) | Original model with the import fixed; no root model replacement required. |
| 4 Balanced experiment | [train_relcoh_10fold_cv_dementia_binary_topic80_20_balanced.py](../train_relcoh_10fold_cv_dementia_binary_topic80_20_balanced.py) | Word-extracted executable source. `EnhancedFrozenCoherenceBinaryClassifier`, `make_binary_train_dev_test_files`, `train_binary_classifier`, and prediction helpers. |
| 4 Tuning and coherence CV | [train_relcoh_binary_hyperparam_tuned.py](../train_relcoh_binary_hyperparam_tuned.py) | Current weighted-loss implementation. `make_binary_8_1_1_tuning_files`, `train_binary_classifier_with_hyperparameter_tuning`, `run_10fold_cv_model_selection`; also supports ordinary topic-split training. |
| 4 and 7 | [scripts/run_experiment.py](../scripts/run_experiment.py) | Portable balanced/tuned recipes; saves exact command, environment versions, hashes, and logs. |
| 5 Results | [scripts/audit_results.py](../scripts/audit_results.py) | Recomputes metrics from the supplied CSV and checks topic/overall reconciliation. |
| 5 Historical/support | [cal_coef.py](../cal_coef.py) | `stat_of_ngram`, `coef_analysis`, `search_example`; relation n-gram analysis. Not verified as the generator of the CSV connective-count table. |
| 6 Limitations | [scripts/audit_default_splits.py](../scripts/audit_default_splits.py) | Executes the current trainers' split functions without model loading and audits participant overlap. Requires NumPy/scikit-learn. |
| 7 Verification | [tests/test_review_tools.py](../tests/test_review_tools.py) | Regression tests for arithmetic auditing, source preservation, and participant-overlap detection. |

## Supporting files

| File | Purpose |
|---|---|
| [README.md](../README.md) | Starting point and reading order |
| [REPRODUCING.md](REPRODUCING.md) | Installation, assets, commands, and output locations |
| [VERIFICATION.md](VERIFICATION.md) | Tests performed and remaining verification limits |
| [requirements.txt](../requirements.txt) | Direct dependency pins/ranges, including training and plotting dependencies |
| [requirements-observed.txt](../requirements-observed.txt) | Local installed-package metadata inventory, not a historical lock |
| [results/reported/final_results.csv](../results/reported/final_results.csv) | Supplied aggregate results, byte-for-byte preserved |
| [results/audits/results_audit.json](../results/audits/results_audit.json) | Binary metrics and discrepancies |
| [results/audits/dataset_audit.json](../results/audits/dataset_audit.json) | Current dataset counts and input SHA-256 |
| [results/audits/split_audit.json](../results/audits/split_audit.json) | Recreated split sizes and participant overlap |
| [.gitattributes](../.gitattributes) | Existing LFS rule for model binaries under data |
| [.gitignore](../.gitignore) | Excludes environments, embeddings, generated caches, and run output |

## Missing historical links

The EntRelCoh entry point, original connective-count aggregation, raw transcript conversion/filtering, and exact result-producing binary command/checkpoint were not established by the supplied files. The report marks those links as unresolved rather than mapping them to unrelated code. Exact experiment results should be linked to a saved run manifest and prediction file, not just a training filename.
