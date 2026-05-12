import io
import os
import sys
import tempfile
from contextlib import asynccontextmanager

import requests

import mlflow.artifacts
import numpy as np
import tensorflow as tf
import torch
from diffusers import StableDiffusionXLPipeline
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# In Docker: assets/ lives next to main.py (copied by Dockerfile).
# In local dev: fall back to the preprocessing directory.
_ASSETS_DIR = os.path.join(BASE_DIR, "assets")
_PREPRO_DIR = os.path.join(BASE_DIR, "../../001 Proprocesamiento")

VOCAB_PATH = os.path.join(_ASSETS_DIR, "vocabulary.txt")
MERGES_PATH = os.path.join(_ASSETS_DIR, "merges.csv")

# token_encoder.py is copied to BASE_DIR in Docker; fall back to PREPRO_DIR locally
if not os.path.exists(os.path.join(BASE_DIR, "token_encoder.py")):
    sys.path.insert(0, _PREPRO_DIR)
    VOCAB_PATH = os.path.join(_PREPRO_DIR, "tokens/vocabulary.txt")
    MERGES_PATH = os.path.join(_PREPRO_DIR, "tokens/merges.csv")

from token_encoder import encode, load_merges, load_vocab  # type: ignore  # noqa: E402

# ── Config ────────────────────────────────────────────────────────────────────
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
EXPERIMENT_NAME = "RNN-nombres-dinosaurios"
MAX_GEN_TOKENS = 30

# ── Shared app state (populated at startup) ───────────────────────────────────
state: dict = {}


# ── Startup / shutdown ────────────────────────────────────────────────────────

def _load_model_from_mlflow() -> tuple[tf.keras.Model, str]:
    """Download the latest run's model artifact and load it."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found in MLflow")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="status = 'FINISHED'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No finished runs found in the experiment")

    run_id = runs[0].info.run_id
    tmp_dir = tempfile.mkdtemp(prefix="dino_model_")

    artifact_dir = mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="rnn_model",
        dst_path=tmp_dir,
        tracking_uri=MLFLOW_TRACKING_URI,
    )
    keras_path = os.path.join(artifact_dir, "saved_model.keras")
    model = tf.keras.models.load_model(keras_path)
    return model, run_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 1. Vocab + merges
    state["vocab"] = load_vocab(VOCAB_PATH)
    state["merges"] = load_merges(MERGES_PATH)
    state["ix_to_token"] = {v: k for k, v in state["vocab"].items()}

    # 2. RNN model (must finish before endpoints accept traffic)
    print(f"Connecting to MLflow at {MLFLOW_TRACKING_URI} …")
    model, run_id = _load_model_from_mlflow()
    state["model"] = model
    state["run_id"] = run_id
    print(f"RNN model ready  (run: {run_id})")

    # 3. SDXL image pipeline
    print("Loading SDXL pipeline ...")
    sdxl_pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    )
    state["flux_pipe"] = sdxl_pipe
    print("SDXL pipeline ready")

    yield

    state.clear()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dino Name Generator",
    description="Autoregressive LSTM that completes dinosaur names.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Generation logic ──────────────────────────────────────────────────────────

def _generate(prefix: str, temperature: float) -> str:
    vocab = state["vocab"]
    merges = state["merges"]
    model = state["model"]
    ix_to_token = state["ix_to_token"]

    sow_idx = vocab["<SOW>"]
    eow_idx = vocab["<EOW>"]
    unk_idx = vocab["<UNK>"]

    # Build starting sequence from prefix
    if prefix.strip():
        tokens = encode(prefix.strip(), merges, vocab=vocab)
        # Drop the <EOW> that encode() adds — generation continues from here
        token_strings = [t for t in tokens if t != "<EOW>"]
        ix = [vocab.get(t, unk_idx) for t in token_strings]
    else:
        ix = [sow_idx]

    # Autoregressive decoding
    for _ in range(MAX_GEN_TOKENS):
        x = np.array([ix])                          # (1, seq_len)
        preds = model.predict(x, verbose=0)         # (1, seq_len, vocab_size)
        logits = preds[0, -1, :].astype(np.float64) # last timestep

        # Temperature sampling with numerical stability
        logits = np.log(logits + 1e-10) / max(temperature, 1e-3)
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        next_ix = int(np.random.choice(len(probs), p=probs))

        if next_ix in (eow_idx, 0):  # <EOW> or <PAD> → stop
            break

        ix.append(next_ix)

    # Decode: skip special tokens, join BPE pieces
    name = "".join(
        ix_to_token[i]
        for i in ix
        if ix_to_token.get(i, "") not in ("<SOW>", "<EOW>", "<PAD>", "<UNK>", "")
    )
    return name.capitalize()


# ── Schemas ───────────────────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    prefix: str = ""
    temperature: float = 1.0


class GenerateResponse(BaseModel):
    name: str
    prefix: str
    run_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "run_id": state.get("run_id")}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if "model" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not 0.1 <= req.temperature <= 2.0:
        raise HTTPException(status_code=422, detail="temperature must be between 0.1 and 2.0")

    name = _generate(req.prefix, req.temperature)
    return GenerateResponse(name=name, prefix=req.prefix, run_id=state["run_id"])


NGROK_CONFIG_PATH = "/home/ec2-user/ngrok_link.config"
PREDICT_PROMPT = (
    "Create a brief description of the following made up dinosaur name, "
    "keeping the paleontologist point of view. Answer only with the description, "
    "no extra commentary.\n\nDinosaur: {name}"
)


def _read_ngrok_url() -> str:
    try:
        with open(NGROK_CONFIG_PATH) as f:
            return f.read().strip()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail=f"Config file not found: {NGROK_CONFIG_PATH}")


class PredictRequest(BaseModel):
    name: str


class PredictResponse(BaseModel):
    name: str
    description: str


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if not req.name.strip():
        raise HTTPException(status_code=422, detail="name must not be empty")

    ngrok_url = _read_ngrok_url()

    try:
        res = requests.post(
            f"{ngrok_url}/api/generate",
            json={"model": "phi4", "prompt": PREDICT_PROMPT.format(name=req.name), "stream": False},
            timeout=300,
        )
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=502, detail=f"Error reaching ngrok/Ollama: {e}")

    description = res.json().get("response", "").strip()
    if not description:
        raise HTTPException(status_code=502, detail="Empty response from model")

    return PredictResponse(name=req.name, description=description)


class GenerateImageRequest(BaseModel):
    # Resolución default 512×512 para cumplir con el spec del taller
    # ("limitar la resolución para disminuir consumo de recursos") y para
    # acelerar la generación si el host corre en CPU.
    prompt: str
    height: int = 512
    width: int = 512
    guidance_scale: float = 7.5
    num_inference_steps: int = 20
    seed: int = 0


@app.post("/generate-image", response_class=Response)
def generate_image(req: GenerateImageRequest):
    if "flux_pipe" not in state:
        raise HTTPException(status_code=503, detail="Image pipeline not loaded yet")

    pipe = state["flux_pipe"]
    image = pipe(
        req.prompt,
        height=req.height,
        width=req.width,
        guidance_scale=req.guidance_scale,
        num_inference_steps=req.num_inference_steps,
        generator=torch.Generator("cpu").manual_seed(req.seed),
    ).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.read(), media_type="image/png")
