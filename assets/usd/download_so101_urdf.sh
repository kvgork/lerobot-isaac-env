#!/usr/bin/env bash
# download_so101_urdf.sh
#
# Fetches the SO-ARM100 (SO-101) URDF and mesh assets from the official
# TheRobotStudio/SO-ARM100 GitHub repository, then prints the Isaac Lab
# URDF→USD conversion command to run next.
#
# Usage:
#   bash assets/usd/download_so101_urdf.sh [--out-dir DIR]
#
# Options:
#   --out-dir DIR   Directory to clone/fetch into. Default: /tmp/so-arm100
#
# After this script completes, run the printed conversion command inside
# your Isaac Lab pixi environment:
#   pixi shell
#   python <...conversion command...>
#
# Then copy the resulting so101.usd into this assets/usd/ directory.

set -euo pipefail

REPO_URL="https://github.com/TheRobotStudio/SO-ARM100.git"
URDF_SUBPATH="Simulation/SO101/so101.urdf"
DEFAULT_OUT_DIR="/tmp/so-arm100"

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
OUT_DIR="${DEFAULT_OUT_DIR}"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        -h|--help)
            sed -n '/^#/p' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Step 1: Clone or update the SO-ARM100 repository
# ---------------------------------------------------------------------------
echo "==> Fetching SO-ARM100 URDF from ${REPO_URL}"
if [ -d "${OUT_DIR}/.git" ]; then
    echo "    Repository already cloned at ${OUT_DIR} — pulling latest."
    git -C "${OUT_DIR}" pull --ff-only
else
    echo "    Cloning into ${OUT_DIR} (sparse checkout — URDF + meshes only)."
    git clone \
        --depth 1 \
        --filter=blob:none \
        --sparse \
        "${REPO_URL}" \
        "${OUT_DIR}"
    git -C "${OUT_DIR}" sparse-checkout set "Simulation/SO101"
fi

URDF_PATH="${OUT_DIR}/${URDF_SUBPATH}"

if [ ! -f "${URDF_PATH}" ]; then
    echo "ERROR: URDF not found at ${URDF_PATH}."
    echo "       Check the repo structure — the path may have changed."
    exit 1
fi

echo "    URDF found at: ${URDF_PATH}"

# ---------------------------------------------------------------------------
# Step 2: Print the Isaac Lab conversion command
# ---------------------------------------------------------------------------

# Resolve the output directory (where this script lives → assets/usd/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "==> URDF download complete."
echo ""
echo "Next: run the following command inside your Isaac Lab pixi environment"
echo "      to convert the URDF to USD:"
echo ""
echo "  pixi shell"
echo "  python -c \""
echo "  from isaaclab.utils.assets import convert_urdf"
echo "  convert_urdf("
echo "      urdf_path='${URDF_PATH}',"
echo "      usd_dir='${SCRIPT_DIR}',"
echo "      usd_file_name='so101.usd',"
echo "      merge_fixed_joints=False,"
echo "  )"
echo "  \""
echo ""
echo "Then verify the USD loaded correctly:"
echo "  python -c \"from isaaclab.utils.assets import check_usd_file; check_usd_file('${SCRIPT_DIR}/so101.usd')\""
echo ""
echo "See assets/usd/README.md for troubleshooting notes."
