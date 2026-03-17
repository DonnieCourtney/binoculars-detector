import os
import re
import json
import math
import torch
import logging
import statistics
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("binoculars")

# --- Configuration ---
OBSERVER_MODEL = os.getenv("OBSERVER_MODEL", "tiiuae/falcon-7b")
PERFORMER_MODEL = os.getenv("PERFORMER_MODEL", "tiiuae/falcon-7b-instruct")
DEVICE = os.getenv("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
THRESHOLD = float(os.getenv("THRESHOLD", "0.9015"))
LOW_THRESHOLD = float(os.getenv("LOW_THRESHOLD", "0.8536"))
# Scores between THRESHOLD and SUSPICIOUS_CEILING are "suspicious" — too close to call
SUSPICIOUS_MARGIN = float(os.getenv("SUSPICIOUS_MARGIN", "0.05"))  # 5% above threshold

if DEVICE == "cpu":
    OBSERVER_MODEL = os.getenv("OBSERVER_MODEL", "facebook/opt-1.3b")
    PERFORMER_MODEL = os.getenv("PERFORMER_MODEL", "facebook/opt-iml-1.3b")
    log.warning("Running on CPU with smaller models — accuracy will be reduced")

app = FastAPI(title="Binoculars AI Text Detector")

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


def auto_calibrate():
    """Run default calibration samples on first startup if no calibration exists."""
    cal = load_calibration()
    if cal["samples"]:
        log.info(f"Existing calibration found ({len(cal['samples'])} samples), skipping auto-calibrate")
        return

    default_file = Path("default_samples.json")
    if not default_file.exists():
        log.warning("No default_samples.json found, using paper thresholds")
        return

    log.info("Running default calibration (first startup)...")
    defaults = json.loads(default_file.read_text())
    samples = defaults.get("samples", [])

    for i, sample in enumerate(samples):
        text = sample["text"].strip()
        score = compute_score(text)
        cal["samples"].append({
            "label": sample["label"],
            "name": sample.get("name", f"{sample['label']}_{i+1}"),
            "score": round(score, 6),
            "text_preview": text[:100],
            "text_length": len(text),
        })
        log.info(f"  [{sample['label'].upper():5s}] {sample.get('name', '?'):30s} → {score:.4f}")

    # Compute and apply thresholds
    human_scores = [s["score"] for s in cal["samples"] if s["label"] == "human"]
    ai_scores = [s["score"] for s in cal["samples"] if s["label"] == "ai"]

    if human_scores and ai_scores:
        global THRESHOLD, LOW_THRESHOLD
        h_mean = statistics.mean(human_scores)
        a_mean = statistics.mean(ai_scores)
        a_std = statistics.stdev(ai_scores) if len(ai_scores) > 1 else 0.01
        separation = h_mean - a_mean

        if separation > 0:
            THRESHOLD = round(a_mean + (separation * 0.4), 4)
            LOW_THRESHOLD = round(max(a_mean - a_std, 0.5), 4)
        else:
            THRESHOLD = round((h_mean + a_mean) / 2, 4)
            LOW_THRESHOLD = round(min(h_mean, a_mean) - 0.02, 4)

        cal["computed_threshold"] = THRESHOLD
        cal["computed_low_threshold"] = LOW_THRESHOLD
        save_calibration(cal)

        log.info(f"Auto-calibration complete:")
        log.info(f"  Human mean: {h_mean:.4f} (n={len(human_scores)})")
        log.info(f"  AI mean:    {a_mean:.4f} (n={len(ai_scores)})")
        log.info(f"  Separation: {separation:.4f}")
        log.info(f"  Threshold:  {THRESHOLD}")
        log.info(f"  Low thresh: {LOW_THRESHOLD}")
    else:
        save_calibration(cal)
        log.warning("Auto-calibration scored samples but couldn't compute thresholds")


@app.on_event("startup")
async def startup():
    load_models()
    auto_calibrate()


# --- Binoculars core ---
def compute_score(text: str) -> float:
    """Compute binoculars score for a text chunk."""
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=512, padding=True,
    ).to(DEVICE)

    with torch.no_grad():
        obs_logits = observer(**inputs).logits
        perf_logits = performer(**inputs).logits

    shift_obs = obs_logits[:, :-1, :].contiguous()
    shift_perf = perf_logits[:, :-1, :].contiguous()
    shift_labels = inputs["input_ids"][:, 1:].contiguous()

    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    ce_obs = loss_fn(shift_obs.view(-1, shift_obs.size(-1)), shift_labels.view(-1))
    ce_perf = loss_fn(shift_perf.view(-1, shift_perf.size(-1)), shift_labels.view(-1))

    mask = (shift_labels != tokenizer.pad_token_id).view(-1).float()
    n = mask.sum().item()
    if n == 0:
        return 1.0

    ppl_obs = (ce_obs * mask).sum().item() / n
    ppl_perf = (ce_perf * mask).sum().item() / n
    return ppl_obs / ppl_perf if ppl_perf > 0 else 1.0


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, preserving meaningful chunks."""
    # Split on sentence-ending punctuation followed by space or newline
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    # Merge very short fragments (< 30 chars) with the previous sentence
    merged = []
    for s in raw:
        s = s.strip()
        if not s:
            continue
        if merged and len(s) < 30:
            merged[-1] = merged[-1] + " " + s
        else:
            merged.append(s)
    return merged


# --- Pattern Analysis ---

# Words that AI models overuse — statistically more frequent in LLM output
AI_OVERUSED_WORDS = {
    "furthermore", "moreover", "additionally", "consequently", "nevertheless",
    "comprehensive", "crucial", "fundamental", "significant", "substantial",
    "demonstrate", "facilitate", "utilize", "implement", "leverage",
    "multifaceted", "nuanced", "paradigm", "synergy", "holistic",
    "delve", "embark", "foster", "underscore", "pivotal",
    "realm", "landscape", "tapestry", "beacon", "testament",
    "robust", "streamline", "optimize", "enhance", "elevate",
    "in conclusion", "it is worth noting", "it is important to note",
    "plays a crucial role", "serves as a",
}

# Transition phrases AI overuses
AI_TRANSITIONS = {
    "in addition to this", "on the other hand", "as a result",
    "in light of this", "with that being said", "having said that",
    "it goes without saying", "needless to say", "at the end of the day",
    "when it comes to", "in terms of", "with regard to",
    "that said", "that being said",
}


def analyze_patterns(text: str, sentences: list[str]) -> dict:
    """Analyze writing patterns that distinguish human from AI text."""
    words = text.lower().split()
    word_count = len(words)

    # --- Sentence length variance (burstiness) ---
    sent_lengths = [len(s.split()) for s in sentences]
    avg_len = statistics.mean(sent_lengths) if sent_lengths else 0
    len_stdev = statistics.stdev(sent_lengths) if len(sent_lengths) > 1 else 0
    # Coefficient of variation — humans typically > 0.4, AI tends < 0.3
    burstiness = len_stdev / avg_len if avg_len > 0 else 0

    # --- Sentence length pattern (AI tends toward uniform lengths) ---
    length_buckets = {"short": 0, "medium": 0, "long": 0}
    for l in sent_lengths:
        if l <= 10:
            length_buckets["short"] += 1
        elif l <= 25:
            length_buckets["medium"] += 1
        else:
            length_buckets["long"] += 1

    # --- AI vocabulary detection ---
    text_lower = text.lower()
    found_ai_words = []
    for w in AI_OVERUSED_WORDS:
        if w in text_lower:
            found_ai_words.append(w)

    found_ai_transitions = []
    for t in AI_TRANSITIONS:
        if t in text_lower:
            found_ai_transitions.append(t)

    # --- Structural parallelism (AI loves parallel constructions) ---
    # Check for sentences starting with the same word pattern
    starts = [s.split()[0].lower() if s.split() else "" for s in sentences]
    start_counts = {}
    for st in starts:
        start_counts[st] = start_counts.get(st, 0) + 1
    repeated_starts = {k: v for k, v in start_counts.items() if v >= 3}

    # Check for sentences with very similar structure (same length ± 2 words)
    similar_length_groups = 0
    for i in range(len(sent_lengths)):
        group = 0
        for j in range(i + 1, min(i + 4, len(sent_lengths))):
            if abs(sent_lengths[i] - sent_lengths[j]) <= 2:
                group += 1
        if group >= 2:
            similar_length_groups += 1

    # --- Contraction usage (humans use more contractions) ---
    contractions = re.findall(
        r"\b(?:I'm|I'll|I've|I'd|don't|doesn't|didn't|won't|wouldn't|can't|couldn't|"
        r"shouldn't|isn't|aren't|wasn't|weren't|hasn't|haven't|hadn't|it's|that's|"
        r"there's|here's|what's|who's|let's|they're|we're|you're|he's|she's)\b",
        text, re.IGNORECASE,
    )
    contraction_rate = len(contractions) / max(word_count, 1) * 100

    # --- First person usage (personal voice) ---
    first_person = len(re.findall(r"\b(?:I|my|me|mine|myself)\b", text))
    first_person_rate = first_person / max(word_count, 1) * 100

    # --- Paragraph length variance ---
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    para_lengths = [len(p.split()) for p in paragraphs]
    para_variance = statistics.stdev(para_lengths) if len(para_lengths) > 1 else 0

    # --- Build signals list ---
    signals = []

    if burstiness < 0.25:
        signals.append({
            "type": "flag",
            "category": "burstiness",
            "message": f"Very uniform sentence lengths (CV={burstiness:.2f}). Human writing is messier — mix short punchy sentences with longer ones.",
        })
    elif burstiness < 0.35:
        signals.append({
            "type": "warn",
            "category": "burstiness",
            "message": f"Somewhat uniform sentence lengths (CV={burstiness:.2f}). Could use more variation.",
        })
    else:
        signals.append({
            "type": "pass",
            "category": "burstiness",
            "message": f"Good sentence length variation (CV={burstiness:.2f}). Natural rhythm.",
        })

    if found_ai_words:
        signals.append({
            "type": "flag",
            "category": "vocabulary",
            "message": f"AI-overused words detected: {', '.join(found_ai_words)}. Replace with simpler or more specific alternatives.",
        })

    if found_ai_transitions:
        signals.append({
            "type": "flag",
            "category": "transitions",
            "message": f"AI-typical transitions: {', '.join(found_ai_transitions)}. Use more natural connectors or just start the next thought directly.",
        })

    if repeated_starts:
        items = [f"'{k}' ({v}x)" for k, v in repeated_starts.items()]
        signals.append({
            "type": "warn",
            "category": "parallelism",
            "message": f"Repeated sentence starters: {', '.join(items)}. Vary your openings.",
        })

    if similar_length_groups >= 3:
        signals.append({
            "type": "flag",
            "category": "structure",
            "message": f"Multiple clusters of nearly identical sentence lengths. This uniformity is an AI signature.",
        })

    if contraction_rate < 0.5 and word_count > 100:
        signals.append({
            "type": "warn",
            "category": "voice",
            "message": f"Very few contractions ({contraction_rate:.1f}%). Natural informal writing uses more. Consider 'do not' → 'don't', etc.",
        })
    elif contraction_rate > 1.5:
        signals.append({
            "type": "pass",
            "category": "voice",
            "message": f"Natural contraction usage ({contraction_rate:.1f}%). Reads like a real person.",
        })

    if first_person_rate > 2.0:
        signals.append({
            "type": "pass",
            "category": "voice",
            "message": f"Strong personal voice ({first_person_rate:.1f}% first-person). Reads authentically.",
        })
    elif first_person_rate < 0.5 and word_count > 100:
        signals.append({
            "type": "warn",
            "category": "voice",
            "message": "Very little first-person voice. Adding personal perspective makes text read more human.",
        })

    if length_buckets["medium"] > 0.7 * len(sentences) and len(sentences) > 5:
        signals.append({
            "type": "flag",
            "category": "structure",
            "message": f"{length_buckets['medium']}/{len(sentences)} sentences are medium length (11-25 words). AI defaults to this range. Add some short punchy sentences and longer complex ones.",
        })

    return {
        "burstiness": round(burstiness, 3),
        "avg_sentence_length": round(avg_len, 1),
        "sentence_length_stdev": round(len_stdev, 1),
        "length_distribution": length_buckets,
        "contraction_rate": round(contraction_rate, 2),
        "first_person_rate": round(first_person_rate, 2),
        "ai_vocabulary_count": len(found_ai_words),
        "ai_transition_count": len(found_ai_transitions),
        "signals": signals,
    }


# --- Rewrite Coach ---
def generate_suggestions(
    sentences: list[str],
    sentence_scores: list[dict],
    patterns: dict,
) -> list[dict]:
    """Generate specific rewrite suggestions for flagged sentences and patterns."""
    suggestions = []

    # Per-sentence suggestions for the worst offenders
    flagged = sorted(
        [s for s in sentence_scores if s["score"] < THRESHOLD],
        key=lambda x: x["score"],
    )

    for item in flagged[:5]:  # Top 5 worst
        sent = item["text"]
        score = item["score"]
        tips = []

        words = sent.split()
        sent_lower = sent.lower()

        # Check for AI vocabulary in this sentence
        ai_words_here = [w for w in AI_OVERUSED_WORDS if w in sent_lower]
        if ai_words_here:
            replacements = {
                "furthermore": "also / and / plus",
                "moreover": "and / on top of that",
                "additionally": "also / and",
                "consequently": "so / because of that",
                "nevertheless": "still / but / even so",
                "comprehensive": "[be specific: what does it cover?]",
                "crucial": "matters because [reason]",
                "fundamental": "[say what it actually is]",
                "significant": "[quantify it or name the impact]",
                "substantial": "[give the number or scale]",
                "demonstrate": "show / prove",
                "facilitate": "help / make easier / enable",
                "utilize": "use",
                "implement": "build / set up / add",
                "leverage": "use / take advantage of",
                "robust": "[describe what makes it strong]",
                "streamline": "simplify / speed up",
                "optimize": "improve / tune / make faster",
                "enhance": "improve / add to",
                "elevate": "raise / improve",
                "delve": "dig into / look at / explore",
                "pivotal": "key / important because [reason]",
                "holistic": "[describe what aspects you mean]",
            }
            for w in ai_words_here:
                alt = replacements.get(w, "[use a more specific word]")
                tips.append(f"Replace '{w}' with: {alt}")

        # Check sentence length
        if 12 <= len(words) <= 22:
            tips.append(
                "Medium-length sentence — either shorten it to punch harder, "
                "or extend it with a specific detail"
            )

        # Check for passive voice indicators
        passive = re.search(
            r"\b(?:is|are|was|were|be|been|being)\s+(?:\w+ed|written|known|seen|made|done|given|taken)\b",
            sent, re.IGNORECASE,
        )
        if passive:
            tips.append(
                f"Passive voice detected ('{passive.group()}'). "
                "Rewrite with the subject doing the action"
            )

        # Check for hedging language
        hedges = re.findall(
            r"\b(?:somewhat|relatively|fairly|rather|quite|perhaps|possibly|"
            r"tend to|seems to|appears to|might be|could be)\b",
            sent, re.IGNORECASE,
        )
        if hedges:
            tips.append(
                f"Hedging language: {', '.join(hedges)}. "
                "Commit to the statement or cut it"
            )

        if tips:
            suggestions.append({
                "sentence_index": item["index"],
                "sentence": sent[:120] + ("..." if len(sent) > 120 else ""),
                "score": score,
                "severity": "high" if score < LOW_THRESHOLD else "medium",
                "tips": tips,
            })

    # Global suggestions from pattern analysis
    for signal in patterns.get("signals", []):
        if signal["type"] in ("flag", "warn"):
            suggestions.append({
                "sentence_index": -1,
                "sentence": "[Overall pattern]",
                "score": 0,
                "severity": "high" if signal["type"] == "flag" else "medium",
                "tips": [signal["message"]],
            })

    return suggestions


# --- API Models ---
class DetectRequest(BaseModel):
    text: str


class SentenceScore(BaseModel):
    index: int
    text: str
    score: float
    rating: str  # "human", "borderline", "ai"


class DetectResponse(BaseModel):
    score: float
    prediction: str
    confidence: str
    threshold: float
    details: str
    sentences: list[SentenceScore]
    patterns: dict
    suggestions: list[dict]


@app.post("/api/detect", response_model=DetectResponse)
async def detect(req: DetectRequest):
    text = req.text.strip()
    if len(text) < 50:
        return DetectResponse(
            score=0.0, prediction="insufficient_text", confidence="none",
            threshold=THRESHOLD,
            details="Need at least 50 characters for meaningful analysis",
            sentences=[], patterns={}, suggestions=[],
        )

    # Overall score
    overall = compute_score(text)

    # Per-sentence scoring
    sentences = split_sentences(text)
    sentence_scores = []
    for i, sent in enumerate(sentences):
        if len(sent.split()) < 4:
            # Too short to score meaningfully
            s_score = overall
        else:
            s_score = compute_score(sent)

        if s_score >= THRESHOLD:
            rating = "human"
        elif s_score >= LOW_THRESHOLD:
            rating = "borderline"
        else:
            rating = "ai"

        sentence_scores.append({
            "index": i,
            "text": sent,
            "score": round(s_score, 4),
            "rating": rating,
        })

    # Pattern analysis
    patterns = analyze_patterns(text, sentences)

    # Rewrite suggestions
    suggestions = generate_suggestions(sentences, sentence_scores, patterns)

    # Count pattern red flags for composite scoring
    red_flags = sum(1 for s in patterns.get("signals", []) if s["type"] == "flag")
    warn_flags = sum(1 for s in patterns.get("signals", []) if s["type"] == "warn")
    flagged_sentences = sum(1 for s in sentence_scores if s["rating"] == "ai")
    borderline_sentences = sum(1 for s in sentence_scores if s["rating"] == "borderline")
    total_sentences = len(sentence_scores)

    # Proportion of sentences that are AI or borderline
    suspect_ratio = (flagged_sentences + borderline_sentences * 0.5) / max(total_sentences, 1)

    suspicious_ceiling = THRESHOLD + (THRESHOLD * SUSPICIOUS_MARGIN)

    # Overall verdict — composite of score + patterns + sentence distribution
    if overall < LOW_THRESHOLD:
        prediction, confidence = "ai_generated", "high"
        details = f"Score {overall:.4f} is well below threshold {THRESHOLD:.4f}. Strong AI signature."
    elif overall < THRESHOLD:
        prediction, confidence = "ai_generated", "moderate"
        details = f"Score {overall:.4f} is below threshold {THRESHOLD:.4f}. Likely AI-generated."
    elif overall < suspicious_ceiling:
        # In the suspicious zone — too close to call on score alone
        # Use pattern signals and sentence distribution to tip the verdict
        if red_flags >= 2 or suspect_ratio > 0.3:
            prediction, confidence = "ai_generated", "moderate"
            details = (
                f"Score {overall:.4f} barely clears threshold {THRESHOLD:.4f} "
                f"but {red_flags} pattern flags and {flagged_sentences}/{total_sentences} "
                f"flagged sentences indicate AI generation."
            )
        elif red_flags >= 1 or suspect_ratio > 0.15:
            prediction, confidence = "suspicious", "moderate"
            details = (
                f"Score {overall:.4f} is marginally above threshold {THRESHOLD:.4f}. "
                f"Pattern analysis found {red_flags} flags, {warn_flags} warnings. "
                f"{flagged_sentences + borderline_sentences}/{total_sentences} sentences "
                f"scored below threshold. Likely AI-generated or heavily AI-assisted."
            )
        else:
            prediction, confidence = "suspicious", "low"
            details = (
                f"Score {overall:.4f} is in the margin zone above threshold {THRESHOLD:.4f}. "
                f"Pattern analysis is mostly clean. Could be human with very predictable "
                f"style, or lightly edited AI text."
            )
    else:
        # Clearly above suspicious ceiling
        if red_flags >= 3 or suspect_ratio > 0.4:
            # Score says human but patterns scream AI — flag it
            prediction, confidence = "suspicious", "moderate"
            details = (
                f"Score {overall:.4f} is above threshold but {red_flags} pattern "
                f"flags detected. {flagged_sentences}/{total_sentences} sentences "
                f"individually flagged. Mixed signals — investigate further."
            )
        elif overall > suspicious_ceiling * 1.1:
            prediction, confidence = "human_written", "high"
            details = f"Score {overall:.4f} is well above threshold {THRESHOLD:.4f}. Human-like unpredictability."
        else:
            prediction, confidence = "human_written", "moderate"
            details = f"Score {overall:.4f} is above threshold {THRESHOLD:.4f}. Appears human-written."

    return DetectResponse(
        score=round(overall, 4),
        prediction=prediction,
        confidence=confidence,
        threshold=THRESHOLD,
        details=details,
        sentences=[SentenceScore(**s) for s in sentence_scores],
        patterns=patterns,
        suggestions=suggestions,
    )


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "observer": OBSERVER_MODEL,
        "performer": PERFORMER_MODEL,
        "threshold": THRESHOLD,
        "low_threshold": LOW_THRESHOLD,
    }


# --- Calibration System ---
CALIBRATION_FILE = Path(os.getenv("CALIBRATION_FILE", "/models/calibration.json"))
# Fallback for non-docker runs
if not CALIBRATION_FILE.parent.exists():
    CALIBRATION_FILE = Path("calibration.json")


def load_calibration() -> dict:
    if CALIBRATION_FILE.exists():
        return json.loads(CALIBRATION_FILE.read_text())
    return {"samples": [], "computed_threshold": None, "computed_low_threshold": None}


def save_calibration(data: dict):
    CALIBRATION_FILE.write_text(json.dumps(data, indent=2))


class CalibrationSample(BaseModel):
    text: str
    label: str  # "human" or "ai"
    name: str = ""  # optional label for the sample


class CalibrationResult(BaseModel):
    sample_count: int
    human_scores: list[float]
    ai_scores: list[float]
    human_mean: float | None
    human_stdev: float | None
    ai_mean: float | None
    ai_stdev: float | None
    computed_threshold: float | None
    computed_low_threshold: float | None
    separation: float | None  # gap between distributions
    active_threshold: float
    active_low_threshold: float


@app.post("/api/calibrate/add")
async def calibrate_add(sample: CalibrationSample):
    """Add a labeled sample to the calibration set and score it."""
    if sample.label not in ("human", "ai"):
        return {"error": "label must be 'human' or 'ai'"}
    if len(sample.text.strip()) < 50:
        return {"error": "sample must be at least 50 characters"}

    score = compute_score(sample.text.strip())
    cal = load_calibration()
    cal["samples"].append({
        "label": sample.label,
        "name": sample.name or f"{sample.label}_{len(cal['samples']) + 1}",
        "score": round(score, 6),
        "text_preview": sample.text.strip()[:100],
        "text_length": len(sample.text.strip()),
    })
    save_calibration(cal)

    return {
        "score": round(score, 4),
        "label": sample.label,
        "name": cal["samples"][-1]["name"],
        "total_samples": len(cal["samples"]),
    }


@app.get("/api/calibrate/status", response_model=CalibrationResult)
async def calibrate_status():
    """Get current calibration state and computed thresholds."""
    cal = load_calibration()
    human_scores = [s["score"] for s in cal["samples"] if s["label"] == "human"]
    ai_scores = [s["score"] for s in cal["samples"] if s["label"] == "ai"]

    h_mean = statistics.mean(human_scores) if human_scores else None
    h_std = statistics.stdev(human_scores) if len(human_scores) > 1 else None
    a_mean = statistics.mean(ai_scores) if ai_scores else None
    a_std = statistics.stdev(ai_scores) if len(ai_scores) > 1 else None

    computed_threshold = None
    computed_low = None
    separation = None

    if h_mean is not None and a_mean is not None:
        # Optimal threshold sits between the two distributions
        # Weight it toward the AI side to minimize false negatives
        # (better to flag human text than miss AI text)
        separation = h_mean - a_mean

        if separation > 0:
            # Good separation — threshold between distributions
            # Place it at 40% from AI mean toward human mean (biased toward catching AI)
            computed_threshold = a_mean + (separation * 0.4)
            # Low threshold = 1 stdev below AI mean (high confidence AI)
            computed_low = a_mean - (a_std if a_std else separation * 0.1)
        else:
            # Distributions overlap or are inverted — use midpoint
            computed_threshold = (h_mean + a_mean) / 2
            computed_low = min(h_mean, a_mean) - 0.02

        computed_threshold = round(computed_threshold, 4)
        computed_low = round(computed_low, 4)

    return CalibrationResult(
        sample_count=len(cal["samples"]),
        human_scores=sorted(human_scores),
        ai_scores=sorted(ai_scores),
        human_mean=round(h_mean, 4) if h_mean else None,
        human_stdev=round(h_std, 4) if h_std else None,
        ai_mean=round(a_mean, 4) if a_mean else None,
        ai_stdev=round(a_std, 4) if a_std else None,
        computed_threshold=computed_threshold,
        computed_low_threshold=computed_low,
        separation=round(separation, 4) if separation is not None else None,
        active_threshold=THRESHOLD,
        active_low_threshold=LOW_THRESHOLD,
    )


@app.post("/api/calibrate/apply")
async def calibrate_apply():
    """Apply computed thresholds from calibration data."""
    global THRESHOLD, LOW_THRESHOLD, SUSPICIOUS_MARGIN
    cal = load_calibration()
    human_scores = [s["score"] for s in cal["samples"] if s["label"] == "human"]
    ai_scores = [s["score"] for s in cal["samples"] if s["label"] == "ai"]

    if not human_scores or not ai_scores:
        return {"error": "Need at least 1 human and 1 AI sample to calibrate"}

    h_mean = statistics.mean(human_scores)
    a_mean = statistics.mean(ai_scores)
    a_std = statistics.stdev(ai_scores) if len(ai_scores) > 1 else 0.01
    separation = h_mean - a_mean

    if separation > 0:
        new_threshold = a_mean + (separation * 0.4)
        new_low = a_mean - (a_std if a_std else separation * 0.1)
    else:
        new_threshold = (h_mean + a_mean) / 2
        new_low = min(h_mean, a_mean) - 0.02

    THRESHOLD = round(new_threshold, 4)
    LOW_THRESHOLD = round(max(new_low, 0.5), 4)  # floor at 0.5

    cal["computed_threshold"] = THRESHOLD
    cal["computed_low_threshold"] = LOW_THRESHOLD
    save_calibration(cal)

    return {
        "applied": True,
        "threshold": THRESHOLD,
        "low_threshold": LOW_THRESHOLD,
        "separation": round(separation, 4),
        "human_mean": round(h_mean, 4),
        "ai_mean": round(a_mean, 4),
    }


@app.post("/api/calibrate/reset")
async def calibrate_reset():
    """Clear calibration data and re-run default calibration."""
    global THRESHOLD, LOW_THRESHOLD
    save_calibration({"samples": [], "computed_threshold": None, "computed_low_threshold": None})
    auto_calibrate()
    return {"reset": True, "threshold": THRESHOLD, "low_threshold": LOW_THRESHOLD}


@app.delete("/api/calibrate/sample/{index}")
async def calibrate_delete_sample(index: int):
    """Remove a specific calibration sample by index."""
    cal = load_calibration()
    if 0 <= index < len(cal["samples"]):
        removed = cal["samples"].pop(index)
        save_calibration(cal)
        return {"removed": removed["name"], "remaining": len(cal["samples"])}
    return {"error": "invalid index"}


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text()
