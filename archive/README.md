# Historical model references

These files are copied from the supplied `model_origin.py` and `model_v2.py` for comparison. Training imports the root `model.py`, never these files.

`model_origin.py` omits the `Transformer_Encoder` import used by `SentTransformer`. `model_v2.py` adds that import. The current root model adds `extract_features` and the frozen binary wrapper. Both current training scripts also define and use an enhanced weighted-loss wrapper internally.

These references preserve their original author comments. They are not alternative runnable entry points or verified checkpoints.
