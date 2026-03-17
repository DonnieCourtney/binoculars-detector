# Binoculars AI Text Detector

Self-hosted AI text detection using the [Binoculars](https://github.com/ahans30/Binoculars) method (ICML 2024). Compares perplexity ratios between two LLMs to detect AI-generated text — no training data needed, works on any model's output.

## Quick Start

**GPU (recommended — requires NVIDIA GPU + Container Toolkit):**
```bash
./run.sh
```

**CPU (smaller models, reduced accuracy):**
```bash
./run.sh cpu
```

Open `http://localhost:8111` and paste text to analyze.

## How It Works

Binoculars measures how predictable text is to two related language models (observer vs performer). AI-generated text is equally predictable to both models, producing a low ratio. Human text surprises the performer more, producing a higher ratio.

- **Score < 0.8536**: High confidence AI-generated
- **Score < 0.9015**: Likely AI-generated
- **Score > 0.9015**: Likely human-written

## Configuration

Environment variables (set in docker-compose.yml):

| Variable | Default (GPU) | Default (CPU) |
|---|---|---|
| `OBSERVER_MODEL` | `tiiuae/falcon-7b` | `facebook/opt-1.3b` |
| `PERFORMER_MODEL` | `tiiuae/falcon-7b-instruct` | `facebook/opt-iml-1.3b` |
| `THRESHOLD` | `0.9015` | `0.9015` |
| `DEVICE` | `cuda` | `cpu` |

## Hardware Requirements

- **GPU mode**: NVIDIA GPU with 16GB+ VRAM (RTX 4090, A100, etc.), NVIDIA Container Toolkit
- **CPU mode**: 8GB+ RAM, any x86_64 CPU (slower inference, ~30-60s per analysis)

## API

```bash
curl -X POST http://localhost:8111/api/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Your text here..."}'
```

```bash
curl http://localhost:8111/api/health
```
