## Robust Tagging Challenge

Not run through Codabench anymore — participants get the preprocessed `.pt` files directly and run the pipeline scripts (or `hackathon_playground.ipynb`) themselves. What they edit: `src/embedding/degradation.py` (currently a stub) and `src/embedding/models.py`, plus `configs/train_config.yaml` for hyperparameters.

**Data pipeline:**

```bash
# training data (SM backgrounds, 4-class) -> pt_files/robust_tagging_train_data.pt
python converterHLT.py --config configs/data_config_collide1m.yaml

# eval data (background + signal: HH->4b, four top) -> pt_files/robust_tagging_eval.pt
python converterHLT.py --config configs/data_config_eval.yaml

# train
python train.py \
  --data_cfg configs/data_config_collide1m.yaml \
  --train_cfg configs/train_config.yaml \
  --data pt_files/robust_tagging_train_data.pt \
  --outdir checkpoints

# evaluate: embeds the nominal eval set, trains a linear probe, then sweeps
# degradation severity on the fly (no separate degraded-files step) and plots
# AUC vs. severity
python eval.py \
  --train_cfg configs/train_config.yaml \
  --data_cfg configs/data_config_eval.yaml \
  --encoder checkpoints/<checkpoint>.pth \
  --data pt_files/robust_tagging_eval.pt \
  --outdir eval_plots \
  --sev_min 0.0 --sev_max 1.0 --num_severities 10 \
  --diagnostics
```

Full runs (`train.py`, `converterHLT.py`) are memory/time heavy — prefer the k8s Jobs in `k8s/` over running them directly for anything beyond a quick `--test_mode` check.
