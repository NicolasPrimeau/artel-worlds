#!/usr/bin/env bash
# Artel Worlds on Fly: one machine runs every world and a private Artel, and it sleeps
# when idle. CI only deploys on an explicit workflow_dispatch; this is the same
# operation from a laptop. Log in first with `flyctl auth login`, or set FLY_API_TOKEN.
#
#   ./scripts/worlds.sh status     what is running
#   ./scripts/worlds.sh up         deploy the repo
#   ./scripts/worlds.sh down       stop the machine, keep the volume
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP=artel-worlds

die() { echo "error: $*" >&2; exit 1; }

command -v flyctl >/dev/null 2>&1 || export PATH="$HOME/.fly/bin:$PATH"
command -v flyctl >/dev/null 2>&1 || die "flyctl not found (looked in PATH and ~/.fly/bin)"

machine_ids() { flyctl machines list -a "$APP" --json | python3 -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)))'; }

cmd_status() {
    flyctl machines list -a "$APP"
    flyctl volumes list -a "$APP"
}

cmd_down() {
    for id in $(machine_ids); do
        flyctl machine stop "$id" -a "$APP"
    done
}

cmd_up() {
    (cd "$REPO_ROOT" && flyctl deploy --app "$APP" --remote-only --yes --strategy immediate)
    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' -m 60 "https://worlds.artel.run/" 2>/dev/null)" || true
    [[ "$code" == 200 ]] || die "deployed but https://worlds.artel.run/ returned ${code:-000}. Check 'flyctl logs -a $APP'."
    echo "$APP is up"
}

case "${1:-status}" in
    status) cmd_status ;;
    down)   cmd_down ;;
    up)     cmd_up ;;
    *)      die "usage: $0 {status|down|up}" ;;
esac
