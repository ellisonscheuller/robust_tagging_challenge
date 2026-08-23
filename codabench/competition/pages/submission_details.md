# Submission Details

Submit a single `.py` file containing a `Model` class with this interface:

- `Model(data_dir, out_dir)` — constructor. Store both paths.
- `fit()` — train using whatever is under `data_dir`. No arguments, no return value.
- `predict()` — run inference and write predictions to `out_dir`. No arguments, no return value.

`fit()` and `predict()` are called once each, in that order, in the same process.

## What predict() must write

For each eval file found under `data_dir` (see Data Description), write predicted class
probabilities as a `.npy` array of shape `[num_events, num_classes]`:

- `out_dir/pred_nominal.npy` — for `data_dir/eval_nominal.pt`
- `out_dir/pred_severity_<N>.npy` — one per file in `data_dir/eval_degraded/`, `<N>` matching
  that file's severity number

Anything else you write to `out_dir` is ignored by scoring.

## Environment

Submissions run inside this competition's Docker image, which has this challenge's `embedding`
package pre-installed — `solution/model.py` shows the intended usage
(`from embedding.models import TransformerEncoder`, etc.). You're not required to use it; any
architecture is fine as long as your `Model` class implements the interface above.
