#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
source .venv/bin/activate
set -a; source .env; set +a
exec python -m uvicorn serving.app:app --host 127.0.0.1 --port 8000 "$@"