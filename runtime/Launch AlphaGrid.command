#!/bin/bash

# ── AlphaGrid Launcher ─────────────────────────────────────────────
RUNTIME_DIR="$(cd "$(dirname "$0")" && pwd)"
DIR="$(cd "$RUNTIME_DIR/.." && pwd)"
PORT=8080
VENV="$DIR/venv"
PYTHON="$VENV/bin/python"
APP_URL="http://localhost:$PORT/login"
LOG_FILE="$DIR/logs/launcher.log"
PID_FILE="$DIR/logs/launcher.pid"

# ── Terminal title ─────────────────────────────────────────────────
echo -e "\033]0;AlphaGrid v8\007"
clear

echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║        AlphaGrid v8 — Premium Client Launcher       ║"
echo "  ╠══════════════════════════════════════════════════════╣"
echo "  ║  Landing page:  http://localhost:$PORT/login         ║"
echo "  ║  Dashboard:     http://localhost:$PORT/dashboard     ║"
echo "  ║  API docs:      http://localhost:$PORT/docs          ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo ""

# ── Check if already running ──────────────────────────────────────
if lsof -i :$PORT -sTCP:LISTEN -t &>/dev/null; then
  echo "  ✓ AlphaGrid already running on port $PORT"
  echo "  ▸ Opening browser..."
  echo ""
  open "$APP_URL"
  exit 0
fi

# ── Check virtualenv ───────────────────────────────────────────────
if [ ! -f "$PYTHON" ]; then
  echo "  [!] Virtual environment not found."
  echo "      Run once in terminal:"
  echo "      cd \"$DIR\""
  echo "      python3.11 -m venv venv"
  echo "      source venv/bin/activate"
  echo "      pip install -r requirements.txt"
  echo ""
  read -p "  Press Enter to exit..."
  exit 1
fi

mkdir -p "$DIR/logs"
cd "$DIR"

if [ -f "$DIR/.env" ]; then
  echo "  ▸ Loading environment from .env"
  set -a
  source "$DIR/.env"
  set +a
fi

echo "  ▸ Starting AlphaGrid on $APP_URL ..."

nohup "$PYTHON" -m uvicorn dashboard.app:app \
  --host 0.0.0.0 \
  --port $PORT \
  --log-level warning \
  < /dev/null \
  > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"
disown "$SERVER_PID" 2>/dev/null || true
echo "  ▸ Server PID: $SERVER_PID"

# ── Wait until ready ───────────────────────────────────────────────
echo "  ▸ Waiting for server..."
ATTEMPTS=0
until curl -s "$APP_URL" &>/dev/null; do
  sleep 0.5
  ATTEMPTS=$((ATTEMPTS + 1))
  if [ $ATTEMPTS -ge 40 ]; then
    echo ""
    echo "  [!] Server failed to start. Check $LOG_FILE"
    echo ""
    read -p "  Press Enter to exit..."
    exit 1
  fi
done

# ── Open browser ───────────────────────────────────────────────────
echo ""
echo "  ✓ AlphaGrid is live →  $APP_URL"
echo ""
open "$APP_URL"

echo "  ─────────────────────────────────────────"
echo "  Server running in background (PID $SERVER_PID)"
echo "  Close this window anytime."
echo "  To stop server:  kill $SERVER_PID"
echo "  ─────────────────────────────────────────"
echo ""
