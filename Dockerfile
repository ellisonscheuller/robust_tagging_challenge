FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    jupyterlab \
    numpy \
    pyyaml \
    wandb \
    scikit-learn \
    matplotlib \
    uproot \
    awkward \
    pyarrow \
    tqdm

COPY . /robust_tagging
RUN pip install --no-cache-dir /robust_tagging

WORKDIR /workspace
