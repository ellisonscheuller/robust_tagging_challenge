# Data Description

Derived from `collide1m` (`fastmachinelearning/collide-1m`). Each event is a fixed-size PF
candidate tensor; degradation zeroes out candidates whose (eta, phi) falls in a dropped region,
the same way padding rows already are.

`data_dir` contains:

- `train.pt` — nominal training events, labeled.
- `train_degraded.pt` — the same events, same order, with a randomly placed dead region per
  event (a different patch per event, so the model can't just memorize one shape).
- `eval_nominal.pt` — held-out events, unlabeled, no degradation.
- `eval_degraded/severity_<N>.pt` — the same held-out events, unlabeled, with a dead region
  covering roughly `<N>`% of the eta-phi plane. Unlike training, every event in a given severity
  file shares the *same* dead region (a fixed geometry per severity, not resampled per event) —
  this mimics a detector region that's simply dead for a whole run, and is a harder
  generalization check than training on.

Development Phase and Testing Phase use the same file layout; only the contents under
`data_dir` change (a different, disjoint set of events), and only at the point the competition
moves phases.
