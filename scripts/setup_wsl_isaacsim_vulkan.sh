#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"
INSTALL_ROOT="${WSL_VULKAN_ROOT:-$REPO_ROOT/.wsl-vulkan/mesa-25.0.7}"
DEB_PATH="$INSTALL_ROOT/mesa-vulkan-drivers.deb"
EXTRACT_ROOT="$INSTALL_ROOT/root"
SOURCE_URL="https://ppa.launchpadcontent.net/kisak/turtle/ubuntu/pool/main/m/mesa/mesa-vulkan-drivers_25.0.7~kisak3~j_amd64.deb"

if [[ "$(uname -r)" != *microsoft-standard-WSL2* ]]; then
    echo "This helper is only intended for Ubuntu 22.04 under WSL2." >&2
    exit 2
fi

mkdir -p "$INSTALL_ROOT"

if [[ ! -f "$DEB_PATH" ]]; then
    echo "Downloading Mesa Dozen Vulkan driver..."
    curl --fail --location --retry 3 "$SOURCE_URL" --output "$DEB_PATH"
fi

if [[ ! -f "$EXTRACT_ROOT/usr/lib/x86_64-linux-gnu/libvulkan_dzn.so" ]]; then
    echo "Extracting Vulkan driver without changing system packages..."
    mkdir -p "$EXTRACT_ROOT"
    dpkg-deb --extract "$DEB_PATH" "$EXTRACT_ROOT"
fi

ICD_PATH="$INSTALL_ROOT/dzn_icd.json"
DRIVER_PATH="$EXTRACT_ROOT/usr/lib/x86_64-linux-gnu/libvulkan_dzn.so"
SOURCE_ICD="$EXTRACT_ROOT/usr/share/vulkan/icd.d/dzn_icd.x86_64.json"

cp "$SOURCE_ICD" "$ICD_PATH"
sed -i "s|/usr/lib/x86_64-linux-gnu/libvulkan_dzn.so|$DRIVER_PATH|" "$ICD_PATH"

echo "WSL Vulkan compatibility layer installed at: $INSTALL_ROOT"
echo "Use scripts/run_wsl_isaaclab.sh to launch IsaacLab commands."
