#!/bin/bash
# 应用所有 patches 到 upstream/inspect_evals 子模块
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SUBMODULE="$REPO_ROOT/upstream/inspect_evals"
PATCHES_DIR="$REPO_ROOT/patches/inspect_evals"

if [ ! -d "$SUBMODULE" ]; then
    echo "Error: Submodule not found at $SUBMODULE"
    echo "Run 'git submodule update --init --recursive' first."
    exit 1
fi

if [ ! -d "$PATCHES_DIR" ]; then
    echo "Error: Patches directory not found at $PATCHES_DIR"
    exit 1
fi

echo "Applying patches to upstream/inspect_evals..."

for patch in "$PATCHES_DIR"/*.patch; do
    [ -f "$patch" ] || continue
    patch_name=$(basename "$patch")
    echo "  Applying $patch_name..."
    if git -C "$SUBMODULE" apply --check "$patch" 2>/dev/null; then
        git -C "$SUBMODULE" apply "$patch"
        echo "    ✓ Applied successfully"
    else
        echo "    ⚠ Patch already applied or conflicts detected, skipping"
    fi
done

echo "Done."
