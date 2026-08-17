#!/usr/bin/env bash
# The worlds on Fly, brought up and down on purpose.
#
# Every push to master used to redeploy all six, so they ran — and billed —
# continuously, and shutting one down lasted until the next merge. CI now only
# deploys on an explicit workflow_dispatch; this is the same operation from a laptop.
#
#   ./scripts/worlds.sh status          what is running everywhere
#   ./scripts/worlds.sh down [app…]     destroy machines, keep volumes (default: all)
#   ./scripts/worlds.sh up <app>        deploy one world back
#
# App names are the fly app, not the package: verglas deploys as artel-alibi.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# fly app | fly config | package it runs
APPS=(
    "artel-worlds:fly.toml:automata"
    "artel-alibi:fly-verglas.toml:verglas"
    "artel-phalanx:fly-phalanx.toml:phalanx"
    "artel-pitch:fly-pitch.toml:pitch"
    "artel-vibequest:fly-vibequest.toml:vibequest"
    "artel-watchtower:fly-watchtower.toml:watchtower"
)

die() { echo "error: $*" >&2; exit 1; }

command -v flyctl >/dev/null 2>&1 || export PATH="$HOME/.fly/bin:$PATH"
command -v flyctl >/dev/null 2>&1 || die "flyctl not found (looked in PATH and ~/.fly/bin)"

if [[ -z "${FLY_API_TOKEN:-}" ]]; then
    # This repo has no fly token of its own; it is stored once, with Artel's secrets.
    ENV_FILE="$HOME/projects/Artel/.env"
    [[ -f "$ENV_FILE" ]] || die "set FLY_API_TOKEN, or put it in $ENV_FILE"
    FLY_API_TOKEN="$(grep -m1 '^FLY_API_TOKEN=' "$ENV_FILE" | cut -d= -f2-)"
    [[ -n "$FLY_API_TOKEN" ]] || die "FLY_API_TOKEN is not set in $ENV_FILE"
    export FLY_API_TOKEN
fi

app_config() { for e in "${APPS[@]}"; do [[ "${e%%:*}" == "$1" ]] && { echo "$e" | cut -d: -f2; return; }; done; die "unknown app '$1'"; }
all_apps() { for e in "${APPS[@]}"; do echo "${e%%:*}"; done; }
machine_ids() { flyctl machines list -a "$1" --json 2>/dev/null | python3 -c 'import json,sys; print(" ".join(m["id"] for m in json.load(sys.stdin)))'; }
volume_names() { flyctl volumes list -a "$1" --json 2>/dev/null | python3 -c 'import json,sys; print(",".join(v["name"] for v in json.load(sys.stdin)) or "-")'; }

cmd_status() {
    printf "%-18s %-10s %s\n" APP MACHINES VOLUMES
    local total=0
    for app in $(all_apps); do
        local ids count vols
        ids="$(machine_ids "$app")"
        count=$(wc -w <<<"$ids")
        vols="$(volume_names "$app")"
        total=$((total + count))
        printf "  %-16s %-10s %s\n" "$app" "$( ((count)) && echo "$count running" || echo down )" "$vols"
    done
    echo
    echo "$total machine(s) running across all worlds"
}

cmd_down() {
    local targets=("$@")
    [[ ${#targets[@]} -gt 0 ]] || mapfile -t targets < <(all_apps)
    for app in "${targets[@]}"; do
        local before ids
        before="$(volume_names "$app")"
        ids="$(machine_ids "$app")"
        if [[ -z "$ids" ]]; then
            echo "$app: already down"
        else
            for id in $ids; do
                echo "$app: destroying machine $id…"
                flyctl machine destroy "$id" -a "$app" --force >/dev/null
            done
            echo "$app: down"
        fi
        # Destroying a machine detaches a volume but never deletes it, and this script
        # never deletes one. Assert it rather than assume — a lost volume is only
        # noticed on the next 'up', when nothing can be done about it.
        [[ "$(volume_names "$app")" == "$before" ]] || die "$app: volumes changed during shutdown (was '$before', now '$(volume_names "$app")'). Investigate before deploying again."
    done
}

cmd_up() {
    local app="${1:-}"
    [[ -n "$app" ]] || die "usage: $0 up <app>  (one of: $(all_apps | tr '\n' ' '))"
    local config
    config="$(app_config "$app")"

    # A world with a volume in its config cannot start without one present and
    # unattached; fly reports that only deep in the deploy, as "requires an unattached
    # volume", which reads like an outage. artel-watchtower fails exactly this way.
    if grep -q "^\[mounts\]" "$REPO_ROOT/$config" 2>/dev/null && [[ "$(volume_names "$app")" == "-" ]]; then
        die "$app declares a volume in $config but none exists. Create it with 'flyctl volumes create <name> -a $app' and expect empty state."
    fi

    echo "deploying $app from $config…"
    (cd "$REPO_ROOT" && flyctl deploy --app "$app" --config "$config" --remote-only --yes --strategy immediate)

    local code
    code="$(curl -s -o /dev/null -w '%{http_code}' -m 30 "https://$app.fly.dev/health" 2>/dev/null)" || true
    [[ "$code" == 200 ]] || die "$app deployed but https://$app.fly.dev/health returned ${code:-000}. Check 'flyctl logs -a $app'."
    echo "$app is up (health 200)"
}

case "${1:-status}" in
    status) cmd_status ;;
    down)   shift; cmd_down "$@" ;;
    up)     shift; cmd_up "$@" ;;
    *)      die "usage: $0 {status|down [app…]|up <app>}" ;;
esac
