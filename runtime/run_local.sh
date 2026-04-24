#!/bin/bash
# AlphaGrid v8 — Start server
# Recommended direct CLI path: bash runtime/run_local.sh

cd "$(dirname "$0")/.."

echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║        AlphaGrid v8 — Premium Client Platform        ║"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║  Landing page: http://localhost:8080/login           ║"
echo "  ║  Dashboard:   http://localhost:8080/dashboard        ║"
echo "  ║  API docs:    http://localhost:8080/docs             ║"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║  Private owner account configured via .env          ║"
echo "  ║  Standard users create accounts from the login page ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo ""

# Load .env if it exists
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

exec uvicorn dashboard.app:app \
  --host 0.0.0.0 \
  --port 8080 \
  --reload \
  --log-level info
