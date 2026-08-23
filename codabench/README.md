# Challenge 9 Codabench Bundle

Competition bundle for "Robust Tagging Under Changing Detector Conditions" (see
`CHALLENGE9_DETECTOR_DEGRADATION_PLAN.txt` and `CHALLENGE9_V0_CANDIDATE_MASKING_PLAN.txt` at
the repo root). Based on
[codabench-competition-example](https://gitlab.cern.ch/mlops/platform/challenges/codabench-competition-example).

    codabench/
    ├── README.md
    ├── Dockerfile: environment used to run ingestion and scoring (build from the repo root, see Dockerfile)
    └── competition/
        ├── competition.yaml: main competition configuration (phases, tasks, pages, leaderboard)
        ├── icon.png: TODO, add a banner/logo image
        ├── ingestion_program/
        │   ├── ingestion.py: runs the user's submitted Model (fit, then predict)
        │   └── metadata.yaml
        ├── pages/
        │   ├── quick_start.md
        │   ├── submission_details.md
        │   ├── data_description.md
        │   └── terms.md
        ├── scoring_program/
        │   ├── scoring.py: reads predictions + held-out labels, writes scores.json
        │   └── metadata.yaml
        └── solution/
            └── model.py: baseline Model implementation

## Before this can be uploaded

Everything marked `REPLACE_ME` needs a real value:

- `competition/ingestion_program/ingestion.py`: `input_dir`, an EOS path
- `competition/scoring_program/scoring.py`: `reference_dir`, an EOS path (private, holds
  `labels.npy`)
- `competition/competition.yaml`: `docker_image`, `queue`, phase `start`/`end` dates
- `competition/pages/terms.md`: actual terms
- `competition/icon.png`: doesn't exist yet

Both EOS directories must be shared read-only with the `mlchallenges` service account via
CERNBox, and their contents swapped when the competition moves from Development Phase to
Testing Phase (same paths, different data — see `pages/data_description.md`).

## Building the image

    docker build -f codabench/Dockerfile -t <registry>/challenge9:v0.0.1 .

run from the repo root (not from `codabench/`), so the build can see `pyproject.toml` and
`src/embedding`.

## Uploading

Zip the contents of `competition/` (not the `codabench/` folder itself) and upload it via
Benchmarks / Competitions in the Challenges interface.
