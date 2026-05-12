# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Educational pipeline that trains a token-level LSTM to generate dinosaur names, tracks runs in MLflow, and exposes inference + LLM description + image generation through a FastAPI service. Code comments, prints, and docs are mixed Spanish/English — match the surrounding language in any given file.

## Layout

Directories are numbered to reflect pipeline order and **contain spaces**, so always quote them in shell:

- [000 data/](000%20data/) — `Dinosours.csv` is the source of ~1535 names (column `nombre`).
- [001 Proprocesamiento/](001%20Proprocesamiento/) — BPE training and encoding. Produces `tokens/vocabulary.txt` (one token per line, index = line number) and `tokens/merges.csv`. **Special tokens are fixed at indices 0–3: `<PAD>`, `<SOW>`, `<EOW>`, `<UNK>`.** `<PAD>` MUST stay at index 0 so the model's `Embedding(mask_zero=True)` works.
- [002 model/](002%20model/) — `rnn_model.py` (Embedding → LSTM → Dense softmax, teacher-forced) and `upload_model.py` (trains once, logs the same model under 5 sampling-config runs to MLflow experiment `RNN-nombres-dinosaurios`).
- [003 Ollama config/](003%20Ollama%20config/) — runbook (not code) for running `phi4` via Ollama and exposing it through ngrok. The API reads the resulting URL from `/home/ec2-user/ngrok_link.config`.
- [008 web/app/](008%20web/app/) — `main.py`, the FastAPI service. Loads the latest **finished** MLflow run's `rnn_model/saved_model.keras` artifact on startup and also pulls SDXL from HuggingFace.
- [008 web/dinos-web/](008%20web/dinos-web/) — static `dinosaurios.html` frontend.

## Common commands

Train + log to MLflow (cwd must be `002 model/`):

```bash
cd "002 model" && MLFLOW_TRACKING_URI=http://localhost:5000 python upload_model.py
```

Run the API locally without Docker (cwd must be `008 web/app/`; falls back to reading vocab/merges from `001 Proprocesamiento/tokens/`):

```bash
cd "008 web/app" && uvicorn main:app --host 0.0.0.0 --port 8000
```

Rebuild BPE artifacts:

```bash
cd "001 Proprocesamiento" && python vocabulary_creator.py "../000 data/Dinosours.csv"
```

Full deploy (starts MLflow on :5000, then builds and runs the Docker container on :8000):

```bash
./build.sh [image-name]    # default image name: dino-api
```

There are no tests or linters configured.

## Cross-module wiring you need to know

- **`token_encoder.py` lives in `001 Proprocesamiento/`** but is consumed by both `002 model/rnn_model.py` (which does `sys.path.insert(0, PREPRO_DIR)`) and `008 web/app/main.py`. `build.sh` works around the cross-directory import by **staging** files into a temp dir before `docker build`: `main.py`, `requirements.txt`, `token_encoder.py`, and `tokens/{vocabulary,merges}.*` are all copied flat into the image alongside an `assets/` directory. If you add a new asset the API needs, update the staging block in [build.sh](build.sh) — the Dockerfile alone is not enough.
- **`main.py` has a dual-mode path resolver**: in Docker, `assets/` and `token_encoder.py` sit next to `main.py`; in local dev it falls back to `../../001 Proprocesamiento/`. Preserve both branches when editing paths.
- **The API depends on a running MLflow with at least one FINISHED run** in experiment `RNN-nombres-dinosaurios`. Lifespan startup fails fast otherwise. `MLFLOW_TRACKING_URI` defaults to `http://localhost:5000`; inside Docker, `build.sh` sets it to `http://host.docker.internal:5000`.
- **`/predict` is a proxy** to an Ollama instance reached through ngrok. The URL is read from `/home/ec2-user/ngrok_link.config` on every request (mounted read-only into the container by `build.sh`). The hosted Ollama model is hardcoded to `phi4`.
- **`/generate-image` loads SDXL at startup** (`stabilityai/stable-diffusion-xl-base-1.0`, fp16). Cold start is heavy — expect long first-boot times and large memory use.

## Conventions

- The training script seeds NumPy and TF with `SEED = 42`; keep that deterministic when adding code paths.
- BPE merges are applied **in order of the `paso` column** in `merges.csv` (see `load_merges` sort). Don't rely on file row order.
- Sequence construction wraps each word with `<SOW>` … `<EOW>`; padding is right-side only. The model loss masks `<PAD>` via `mask_zero=True`, so do not reuse index 0 for anything else.
