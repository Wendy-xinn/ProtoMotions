#!/usr/bin/env bash
set -euo pipefail

MIN_AVAILABLE_GB="${MEMORY_GUARD_MIN_AVAILABLE_GB:-4}"
CHECK_INTERVAL_SECONDS="${MEMORY_GUARD_INTERVAL_SECONDS:-1}"

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 COMMAND [ARG ...]" >&2
    exit 2
fi
if [[ ! "$MIN_AVAILABLE_GB" =~ ^[0-9]+$ ]] || (( MIN_AVAILABLE_GB < 1 )); then
    echo "MEMORY_GUARD_MIN_AVAILABLE_GB must be a positive integer" >&2
    exit 2
fi

minimum_kb=$((MIN_AVAILABLE_GB * 1024 * 1024))
child_pid=""

stop_child() {
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -TERM -- "-$child_pid" 2>/dev/null || kill -TERM "$child_pid" 2>/dev/null || true
    fi
}
trap stop_child INT TERM HUP
# Ctrl+Z previously suspended only this wrapper because the Isaac process is
# intentionally in its own session, leaving training alive in the background.
# Treat terminal suspend as a request to terminate the guarded job instead.
trap 'stop_child; exit 148' TSTP

# A separate session makes the complete IsaacLab process tree terminable as a
# group, including simulator subprocesses spawned after Python starts.
setsid "$@" &
child_pid=$!

while kill -0 "$child_pid" 2>/dev/null; do
    available_kb="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)"
    if [[ -n "$available_kb" ]] && (( available_kb < minimum_kb )); then
        echo >&2
        echo "Memory guard: MemAvailable fell below ${MIN_AVAILABLE_GB} GiB; stopping training to keep WSL responsive." >&2
        stop_child
        for _ in {1..10}; do
            kill -0 "$child_pid" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$child_pid" 2>/dev/null; then
            kill -KILL -- "-$child_pid" 2>/dev/null || kill -KILL "$child_pid" 2>/dev/null || true
        fi
        wait "$child_pid" 2>/dev/null || true
        exit 137
    fi
    sleep "$CHECK_INTERVAL_SECONDS"
done

wait "$child_pid"
