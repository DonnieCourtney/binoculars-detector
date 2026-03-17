import os
import torch
import logging
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("binoculars")

# --- Configuration ---
OBSERVER_MODEL = os.getenv("OBSERVER_MODEL", "tiiuae/falcon-7b")
PERFORMER_MODEL = os.getenv("PERFORMER_MODEL", "tiiuae/falcon-7b-instruct")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = float(os.getenv("THRESHOLD", "0.9015"))  # Binoculars paper default
LOW_THRESHOLD = float(os.getenv("LOW_THRESHOLD", "0.8536"))  # High-confidence AI

# CPU fallback: use smaller models
if DEVICE == "cpu":
    OBSERVER_MODEL = os.getenv("OBSERVER_MODEL", "facebook/opt-1.3b")
    PERFORMER_MODEL = os.getenv("PERFORMER_MODEL", "facebook/opt-iml-1.3b")
    log.warning("Running on CPU with smaller models — accuracy will be reduced")

app = FastAPI(title="Binoculars AI Text Detector")

# --- Model loading ---
tokenizer = None
observer = None
performer = None


def load_models():
    global tokenizer, observer, performer
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    log.info(f"Loading observer: {OBSERVER_MODEL} on {DEVICE} ({dtype})")
    observer = AutoModelForCausalLM.from_pretrained(
        OBSERVER_MODEL, torch_dtype=dtype, device_map=DEVICE
    )
    observer.eval()
    log.info(f"Loading performer: {PERFORMER_MODEL} on {DEVICE} ({dtype})")
    performer = AutoModelForCausalLM.from_pretrained(
        PERFORMER_MODEL, torch_dtype=dtype, device_map=DEVICE
    )
    performer.eval()
    tokenizer = AutoTokenizer.from_pretrained(OBSERVER_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    log.info("Models loaded successfully")


@app.on_event("startup")
async def startup():
    load_models()


# --- Binoculars core ---
def compute_binoculars_score(text: str) -> float:
    """
    Binoculars score = perplexity(observer) / perplexity(performer)

    Low score (<threshold) = likely AI-generated
    High score (>threshold) = likely human-written

    The intuition: AI text is equally predictable to both models,
    so the ratio is low. Human text surprises the performer more
    than the observer, pushing the ratio higher.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        observer_logits = observer(**inputs).logits
        performer_logits = performer(**inputs).logits

    # Shift for next-token prediction alignment
    shift_logits_obs = observer_logits[:, :-1, :].contiguous()
    shift_logits_perf = performer_logits[:, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, 1:].contiguous()

    # Cross-entropy per token
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")

    ce_observer = loss_fn(
        shift_logits_obs.view(-1, shift_logits_obs.size(-1)),
        shift_labels.view(-1),
    )
    ce_performer = loss_fn(
        shift_logits_perf.view(-1, shift_logits_perf.size(-1)),
        shift_labels.view(-1),
    )

    # Mask padding tokens
    mask = (shift_labels != tokenizer.pad_token_id).view(-1).float()
    n_tokens = mask.sum().item()
    if n_tokens == 0:
        return 1.0

    ppl_observer = (ce_observer * mask).sum().item() / n_tokens
    ppl_performer = (ce_performer * mask).sum().item() / n_tokens

    if ppl_performer == 0:
        return 1.0

    return ppl_observer / ppl_performer


# --- API ---
class DetectRequest(BaseModel):
    text: str


class DetectResponse(BaseModel):
    score: float
    prediction: str
    confidence: str
    threshold: float
    details: str


@app.post("/api/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    text = req.text.strip()
    if len(text) < 50:
        return DetectResponse(
            score=0.0,
            prediction="insufficient_text",
            confidence="none",
            threshold=THRESHOLD,
            details="Need at least 50 characters for meaningful analysis",
        )

    score = compute_binoculars_score(text)

    if score < LOW_THRESHOLD:
        prediction = "ai_generated"
        confidence = "high"
        details = f"Score {score:.4f} is well below threshold {THRESHOLD}. Strong statistical signature of AI generation."
    elif score < THRESHOLD:
        prediction = "ai_generated"
        confidence = "moderate"
        details = f"Score {score:.4f} is below threshold {THRESHOLD}. Likely AI-generated but less certain."
    else:
        prediction = "human_written"
        confidence = "high" if score > THRESHOLD * 1.15 else "moderate"
        details = f"Score {score:.4f} is above threshold {THRESHOLD}. Text shows human-like unpredictability."

    return DetectResponse(
        score=round(score, 4),
        prediction=prediction,
        confidence=confidence,
        threshold=THRESHOLD,
        details=details,
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "observer": OBSERVER_MODEL,
        "performer": PERFORMER_MODEL,
    }


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text()
