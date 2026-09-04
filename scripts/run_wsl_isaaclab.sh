#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INSTALL_ROOT="${WSL_VULKAN_ROOT:-$REPO_ROOT/.wsl-vulkan/mesa-25.0.7}"
EXTRACT_ROOT="$INSTALL_ROOT/root"
ICD_PATH="$INSTALL_ROOT/dzn_icd.json"
DRIVER_DIR="$EXTRACT_ROOT/usr/lib/x86_64-linux-gnu"
NATIVE_RUNTIME_LIB_DIR="${ISAAC_RUNTIME_LIB_DIR:-$REPO_ROOT/.runtime/isaac-system-libs/lib}"

# Native Linux training nodes do not need the WSLg Mesa compatibility layer.
if ! grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
    if [[ -d "$NATIVE_RUNTIME_LIB_DIR" ]]; then
        export LD_LIBRARY_PATH="$NATIVE_RUNTIME_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    export PROTOMOTIONS_MJCF_USD_CACHE_DIR="${PROTOMOTIONS_MJCF_USD_CACHE_DIR:-${TMPDIR:-/tmp}/protomotions-${USER:-user}/isaaclab_mjcf_usd}"
    mkdir -p "$PROTOMOTIONS_MJCF_USD_CACHE_DIR"
    exec "$@"
fi

if [[ ! -f "$ICD_PATH" ]]; then
    echo "WSL Vulkan compatibility layer is missing." >&2
    echo "Run: bash scripts/setup_wsl_isaacsim_vulkan.sh" >&2
    exit 2
fi

export PATH="/usr/lib/wsl/lib:$PATH"
export LD_LIBRARY_PATH="$DRIVER_DIR:/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VK_ICD_FILENAMES="$ICD_PATH"
export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"

# WSLg may provide the sockets while the shell inherits an invalid runtime
# directory (especially from systemd or VSCode). Set these explicitly so
# IsaacLab's GLFW viewer can create a window and the recorder can capture PNGs.
if [[ -S /mnt/wslg/runtime-dir/wayland-0 ]]; then
    export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
    export XDG_RUNTIME_DIR="/mnt/wslg/runtime-dir"
    export DISPLAY="${DISPLAY:-:0}"
fi

exec "$@"
