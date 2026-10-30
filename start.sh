#!/usr/bin/env bash
#
# MusicRAG - one-click startup.
# Activates the environment, launches Streamlit on port 8501, and opens the browser
# once the server responds.
#
# The IP "90.166.58.170" is the router's public IP: it is NOT accessible from
# within your network (the router does not support NAT loopback). That is why
# this script uses localhost by default and only displays the local IP if you
# want to access it from another device on the same network.
#
# Usage:
#   ./start.sh                  -> abre http://localhost:8501/
#   MUSICRAG_URL=http://localhost:8501 ./start.sh
#   MUSICRAG_PORT=8502 ./start.sh
#
set -euo pipefail
cd "$(dirname "$0")"

source .venv/Scripts/activate

# Ollama: we use the classic binary (not the 0.33.x app/TUI). If no
# server is responding on 11434, we launch it in the background and wait.
OLLAMA_EXE="${OLLAMA_EXE:-$LOCALAPPDATA/Programs/Ollama/ollama.exe}"
if [[ ! -x "$OLLAMA_EXE" ]]; then
  OLLAMA_EXE="$(command -v ollama 2>/dev/null || true)"
fi
if [[ -z "$OLLAMA_EXE" ]]; then
  echo "[start.sh] Warning: could not find ollama.exe; please install Ollama or define OLLAMA_EXE." >&2
elif curl -s -o /dev/null --max-time 2 "http://127.0.0.1:11434/"; then
  echo "[start.sh] Ollama is already running."
else
  echo "[start.sh] Launching Ollama: $OLLAMA_EXE serve"
  "$OLLAMA_EXE" serve >"${TMPDIR:-/tmp}/musicrag-ollama.log" 2>&1 &
  OLLAMA_PID=$!
  for _ in $(seq 1 30); do
    if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:11434/"; then
      echo "[start.sh] Ollama is running."
      break
    fi
    sleep 1
  done
  if ! curl -s -o /dev/null --max-time 2 "http://127.0.0.1:11434/"; then
    echo "[start.sh] Warning: Ollama did not respond yet; check ${TMPDIR:-/tmp}/musicrag-ollama.log" >&2
  fi
fi

HOST="0.0.0.0"
PORT="${MUSICRAG_PORT:-8501}"
DEFAULT_URL="http://localhost:${PORT}/"
OPEN_URL="${MUSICRAG_URL:-$DEFAULT_URL}"

# Local IP on the LAN (to access from another device).
LAN_IP="$(
  python -c 'import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.connect(("8.8.8.8", 80)); print(s.getsockname()[0])' 2>/dev/null || echo "no detectable"
)"

(
  for _ in $(seq 1 30); do
    if curl -s -o /dev/null --max-time 2 "http://127.0.0.1:${PORT}/"; then
      echo ""
      echo "[start.sh] Server is up."
      echo "  Same machine :  $DEFAULT_URL"
      echo "  On your LAN:  http://${LAN_IP}:${PORT}/   (from another device; requires allowing 8501 through the firewall)"
      echo "  Opening     :  $OPEN_URL"
      python -m webbrowser -t "$OPEN_URL"
      exit 0
    fi
    sleep 1
  done
  echo "[start.sh] No server detected on port ${PORT}." >&2
) &
opener_pid=$!
trap 'kill "$opener_pid" 2>/dev/null || true' EXIT

if python -m streamlit run app.py --server.address "${HOST}" --server.port "${PORT}"; then
  streamlit_rc=0
else
  streamlit_rc=$?
fi

# When the last tab of https://localhost:8501/ is closed (or the window),
# app.py exits by itself (watchdog) and `streamlit run` returns. Here we also
# stop the Ollama this script may have launched.
if [[ -n "${OLLAMA_PID:-}" ]]; then
  echo "[start.sh] Stopping Ollama (PID ${OLLAMA_PID}) launched by this script."
  taskkill //PID "${OLLAMA_PID}" //T //F >/dev/null 2>&1 || kill "${OLLAMA_PID}" 2>/dev/null || true
fi
echo "[start.sh] Server closed (exit ${streamlit_rc})."
exit "${streamlit_rc}"