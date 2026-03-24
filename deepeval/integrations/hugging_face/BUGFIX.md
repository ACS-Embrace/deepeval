# HuggingFace Callback Table Bug Fix

## Problem

The display table would intermittently show "N/A" for previous epoch rows.

## Root Cause

In `_generate_table`, the column order was derived only from the **last history entry**:

```python
order = get_column_order(self.deepeval_metric_history[-1])
```

The HuggingFace Trainer logs two structurally different payloads depending on what fired last:

- **Training step logs**: `{'loss': ..., 'learning_rate': ..., 'epoch': ...}`
- **Evaluation logs**: `{'eval_loss': ..., 'eval_runtime': ..., 'eval_samples_per_second': ..., 'epoch': ...}`

When `on_log` fires after different log types across epochs, each epoch's history entry ends up with a different set of keys. Since the table columns were always based on the most recent entry, earlier rows were missing those columns and fell back to "N/A".

The bug was **intermittent** because whether a training-step log or eval log lands in `state.log_history[-1]` at epoch end depends on `logging_steps`, `eval_steps`, and exact step timing — varying between runs and configurations.

## Fix

Derive the column order from the **union of all keys** across every history entry:

```python
all_keys = {}
for row in self.deepeval_metric_history:
    all_keys.update(row)
order = get_column_order(all_keys)
```

This ensures the column set is always stable and complete. Rows that genuinely don't have a value for a column still show "N/A", but that's expected and consistent — not an artifact of which epoch happened to be last.
