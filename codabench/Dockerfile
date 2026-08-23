# Build from the challenge9 repo root, not this codabench/ directory:
#   docker build -f codabench/Dockerfile -t <registry>/challenge9:v0.0.1 .
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

COPY . /challenge9
RUN pip install --no-cache-dir /challenge9
