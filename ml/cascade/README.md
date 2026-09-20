# Cascade evaluation

Run the final held-out, end-to-end evaluation from the repository root with
the cascade's pinned `uv` environment:

```bash
uv run --locked --project ml/cascade python ml/cascade/evaluate.py
```

The script loads the frozen Stage 1 model and its validation-selected threshold, then invokes the frozen Stage 2 model only for windows routed as abnormal. It reuses the existing manifest/scaler normalization and ±8 clipping contract. It does not train models or select a threshold.

Outputs:

- `artifacts/results.json` — six-class metrics, confusion matrix values, and cascade routing/error counts.
- `artifacts/confusion_matrix.png` — held-out six-class confusion matrix.
