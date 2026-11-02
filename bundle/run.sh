#!/bin/bash
set -euo pipefail

ARTEL=http://127.0.0.1:8100
mkdir -p /data/artel /data/worlds
chown -R artel /data

eval "$(/app/.venv/bin/python /worlds/bundle/keys.py)"

run() {
  env -i PATH=/worlds/.venv/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/artel ARTEL_URL=$ARTEL "$@" &
}


artel() {
  env -i PATH=/app/.venv/bin:/usr/local/bin:/usr/bin:/bin HOME=/home/artel \
    HOST=127.0.0.1 PORT=8100 DB_PATH=/data/artel/artel.db \
    FASTEMBED_CACHE_PATH=/data/fastembed_cache RUN_INLINE_ARCHIVIST=0 \
    AGENT_KEYS="$ALL" HF_HUB_DISABLE_XET_DOWNLOAD=1 \
    DEMO_MODE=true PUBLIC_URL=https://live.artel.run \
    UI_PASSWORD="$(head -c 32 /dev/urandom | base64)" \
    REGISTRATION_KEY="$(head -c 32 /dev/urandom | base64)" \
    setpriv --reuid=artel --regid=artel --init-groups \
    sh -c 'cd /app && exec python -m artel.server' &
}

world() {
  local module=$1 port=$2
  shift 2
  run "$@" setpriv --reuid=artel --regid=artel --init-groups \
    sh -c "cd /worlds && exec uvicorn $module:app --host 127.0.0.1 --port $port"
}

artel

for _ in $(seq 1 90); do
  python3 -c "import urllib.request; urllib.request.urlopen('$ARTEL/health', timeout=1)" 2>/dev/null && break
  sleep 1
done
python3 -c "import urllib.request; urllib.request.urlopen('$ARTEL/health', timeout=2)" \
  || { echo "bundle: Artel never came up; refusing to run worlds uncoordinated" >&2; exit 1; }

ALL="$ALL" python3 /worlds/bundle/check.py $ARTEL

KEY=${OPENROUTER_API_KEY:-}
FREE=nvidia/nemotron-3-ultra-550b-a55b:free
OPENROUTER=https://openrouter.ai/api/v1/chat/completions

world automata.server 8101 AUTOMATA_STATE_DIR=/data/worlds \
  WORLDS_UI_PASSWORD="${WORLDS_UI_PASSWORD:?}" WORLDS_UI_SECRET="${WORLDS_UI_SECRET:?}" \
  WORLDS_ADMIN_TOKEN="${WORLDS_ADMIN_TOKEN:-}" \
  ARTEL_AGENT_ID="$AUTOMATA_IDS" ARTEL_KEY="$AUTOMATA_KEYS" \
  PHALANX_DEBUG_URL=http://127.0.0.1:8102 PITCH_DEBUG_URL=http://127.0.0.1:8103 \
  VERGLAS_DEBUG_URL=http://127.0.0.1:8104 VIBEQUEST_DEBUG_URL=http://127.0.0.1:8105 \
  WATCHTOWER_DEBUG_URL=http://127.0.0.1:8106
world phalanx.server 8102 PHALANX_STATE=/data/worlds/phalanx_state.json \
  PHALANX_AGENT_IDS="$PHALANX_IDS" PHALANX_AGENT_KEYS="$PHALANX_KEYS" \
  WORLDS_ADMIN_TOKEN="${WORLDS_ADMIN_TOKEN:-}" \
  OPENROUTER_API_KEY="$KEY"
world pitch.server 8103 PITCH_AGENT_IDS="$PITCH_IDS" PITCH_AGENT_KEYS="$PITCH_KEYS" \
  PITCH_LLM_KEY="$KEY" PITCH_MODEL=$FREE PITCH_LLM_URL=$OPENROUTER
world verglas.server 8104 VERGLAS_STATE=/data/worlds/verglas_state.json \
  VERGLAS_AGENT_IDS="$VERGLAS_IDS" VERGLAS_AGENT_KEYS="$VERGLAS_KEYS" \
  WORLDS_ADMIN_TOKEN="${WORLDS_ADMIN_TOKEN:-}" \
  OPENROUTER_API_KEY="$KEY"
world vibequest.server 8105 VIBEQUEST_ARTEL_AGENT_ID="$VIBEQUEST_IDS" VIBEQUEST_ARTEL_KEY="$VIBEQUEST_KEYS" \
  OPENROUTER_API_KEY="$KEY"
world watchtower.server 8106 WATCHTOWER_DB=/data/worlds/watchtower.db \
  WATCHTOWER_AGENT_IDS="$WATCHTOWER_IDS" WATCHTOWER_AGENT_KEYS="$WATCHTOWER_KEYS" \
  WORLDS_ADMIN_TOKEN="${WORLDS_ADMIN_TOKEN:-}" \
  WATCHTOWER_LLM_KEY="$KEY" WATCHTOWER_MODEL=$FREE WATCHTOWER_LLM_URL=$OPENROUTER

run setpriv --reuid=artel --regid=artel --init-groups caddy run --config /worlds/bundle/Caddyfile --adapter caddyfile

wait -n
echo "bundle: a process exited; stopping so Fly restarts the machine" >&2
exit 1
