#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export no_proxy="localhost,127.0.0.1,${no_proxy:-}"
export NO_PROXY="localhost,127.0.0.1,${NO_PROXY:-}"
DEMO_DIR="$ROOT_DIR/customer_agent_demo"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-si-l-z-sync}"

export PYTHONNOUSERSITE=1

if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME"
  PYTHON_BIN="python"
elif [[ -x "/home/q2li/miniconda3/envs/$CONDA_ENV_NAME/bin/python" ]]; then
  PYTHON_BIN="/home/q2li/miniconda3/envs/$CONDA_ENV_NAME/bin/python"
else
  echo "Cannot find conda env: $CONDA_ENV_NAME" >&2
  echo "Expected either 'conda activate $CONDA_ENV_NAME' or /home/q2li/miniconda3/envs/$CONDA_ENV_NAME/bin/python" >&2
  exit 1
fi

cd "$ROOT_DIR"

echo "[1/2] Checking Qdrant..."
if curl -fsS --connect-timeout 5 http://localhost:6333/collections >/dev/null 2>&1; then
  echo "Qdrant is already available; reusing it."
else
  echo "Qdrant is not running. Starting Qdrant..."
  if ! command -v docker >/dev/null 2>&1; then
    echo "Docker is required to start Qdrant." >&2
    exit 1
  fi
  docker compose -f "$DEMO_DIR/docker-compose.yml" up -d qdrant

  echo "Waiting for Qdrant..."
  READY=false
  for _ in $(seq 1 30); do
    if curl -fsS --connect-timeout 2 http://localhost:6333/collections >/dev/null 2>&1; then
      READY=true
      break
    fi
    sleep 1
  done

  if [ "$READY" = false ]; then
    echo "Error: Qdrant did not become ready within 30 seconds." >&2
    exit 1
  fi
fi

echo "[2/2] Running Document Ingestion..."
exec "$PYTHON_BIN" -m customer_agent_demo.ingest.run
