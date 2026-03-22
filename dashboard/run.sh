#!/bin/bash
# AlphaGrid v7 — Start server
# Run from project root: bash dashboard/run.sh

cd "$(dirname "$0")/.."

echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║        AlphaGrid v7 — Trading Intelligence           ║"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║  Login page:  http://localhost:8080                  ║"
echo "  ║  Dashboard:   http://localhost:8080/dashboard        ║"
echo "  ║  API docs:    http://localhost:8080/docs             ║"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║  Default credentials are in SETUP.md                ║"
echo "  ║  Override via ALPHAGRID_OWNER_PASSWORD in .env      ║"
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
