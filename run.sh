#!/bin/bash
set -e

MODE="${1:-gpu}"

echo "==================================="
echo "  Binoculars AI Text Detector"
echo "==================================="

if [ "$MODE" = "cpu" ]; then
    echo "Mode: CPU (smaller models, reduced accuracy)"
    echo "Models: facebook/opt-1.3b + opt-iml-1.3b"
    docker compose -f docker-compose.cpu.yml up --build -d
else
    # Check for NVIDIA runtime
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        if ! command -v nvidia-smi &>/dev/null; then
            echo ""
            echo "WARNING: No NVIDIA GPU detected."
            echo "  Run with CPU mode: ./run.sh cpu"
            echo "  Or install NVIDIA Container Toolkit:"
            echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            echo ""
            read -p "Try GPU mode anyway? [y/N] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
        fi
    fi
    echo "Mode: GPU (falcon-7b models, full accuracy)"
    echo "Models: tiiuae/falcon-7b + falcon-7b-instruct"
    docker compose up --build -d
fi

echo ""
echo "Starting up... first run downloads models (~14GB for GPU, ~5GB for CPU)"
echo "Watch logs: docker compose logs -f"
echo ""
echo "UI available at: http://localhost:8111"
