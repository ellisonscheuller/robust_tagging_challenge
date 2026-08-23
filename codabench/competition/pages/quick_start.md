# Quick Start

Train a model whose latent space stays useful when part of the detector goes dead. Concretely: an eta-phi region drops out (candidates in that region are zeroed), and your model needs to keep classifying events correctly anyway.

You submit a single Python file containing a `Model` class:

```python
class Model:
    def __init__(self, data_dir, out_dir):
        ...

    def fit(self):
        ...

    def predict(self):
        ...
```

`data_dir` and `out_dir` are passed in by the platform. Your model reads its training and eval data from `data_dir`, and writes predictions to `out_dir` (see Submission Details).

`solution/model.py` in this bundle is a working baseline you can start from.

Scored metric: AUC vs. degradation severity, summarized as the area under that curve (flat curve near AUC 1.0 = robust; steep drop-off = not robust).
