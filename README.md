## Notes (2026-08-30)

**Done:**
- `degradation.py` (Arianna's layer) — done, committed
- `eval.py` — rewritten, tested
- `converterHLT.py` — parquet support added for collide1m (`gather_pfcands_collide1m`)
- `make_eval_files.py` — new script, generates fixed-geometry degraded eval files
- `codabench/` — built and smoke-tested; `model.py` matches notebook
- two eval configs: `configs/data_config_eval_public.yaml` (HH→4b) and `configs/data_config_eval_secret.yaml` (Four Top)
- bunch of bugs fixed (see session notes)

**Launch checklist (in order):**

1. move code to new namespace (fastml26-hackathon) after Arianna implements degrade layer into model
2. run `converterHLT.py` on Melissa's SM background samples → `robust_tagging_train_data.pt`
3. run `converterHLT.py` + `make_eval_files.py` for both eval sets (commands below)
4. build docker image and push to NRP registry:
   ```bash
   docker build -t registry.nrp-nautilus.io/fastml26-hackathon/challenge3:v0.0.1 .
   docker push registry.nrp-nautilus.io/fastml26-hackathon/challenge3:v0.0.1
   ```
5. fill in remaining `REPLACE_ME` in codabench files:
   - `competition.yaml`: queue name (find in Codabench admin panel), phase dates, terms/icon
   - `ingestion_program/ingestion.py`: PVC mount path for train + public eval data
   - `scoring_program/scoring.py`: PVC mount path for secret eval labels
   - `codabench/competition/solution/model.py`: actual filenames for train file, eval nominal, eval degraded dir
   - notebook `hackathon_playground.ipynb`: DATA_DIR, OUT_DIR, labels filename
6. copy secret eval labels to the scoring PVC path (never share with participants)
7. strip `degradation.py` and secret eval config from public repo, then publish
8. zoom with Arianna on notebook + submission instructions

**Data pipeline when ready:**

```bash
# training data (SM backgrounds, 4-class)
python converterHLT.py --config configs/data_config_collide1m.yaml

# public eval (HH->4b signal, participants see this)
python converterHLT.py --config configs/data_config_eval_public.yaml
python make_eval_files.py \
  --data preprocessed_data/robust_tagging_eval_public.pt \
  --out_dir preprocessed_data/eval_degraded_public/

# secret eval (Four Top, final scoring only — don't share)
python converterHLT.py --config configs/data_config_eval_secret.yaml
python make_eval_files.py \
  --data preprocessed_data/robust_tagging_eval_secret.pt \
  --out_dir preprocessed_data/eval_degraded_secret/

# extract labels.npy for codabench scoring
python -c "
import torch, numpy as np
for tag, fname in [('public', 'robust_tagging_eval_public.pt'), ('secret', 'robust_tagging_eval_secret.pt')]:
    d = torch.load(f'preprocessed_data/{fname}', map_location='cpu')
    np.save(f'preprocessed_data/labels_{tag}.npy', d[:, 0, -1].long().numpy())
"
```
