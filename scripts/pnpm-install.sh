#!/usr/bin/env bash
# pnpm install helper for projects whose node_modules is a symlink to ext4
# (/mnt/e project + WSL 9P workaround). pnpm refuses to install into a
# symlinked node_modules (ENOTDIR), so we install in a temp dir on ext4
# and move the result into place.
#
# Usage:
#   scripts/pnpm-install.sh              # install per current package.json + lockfile
#   scripts/pnpm-install.sh --frozen     # frozen-lockfile install
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NM_TARGET="$(readlink -f "$PROJECT_DIR/node_modules" 2>/dev/null || echo "")"

if [ -z "$NM_TARGET" ]; then
	echo "ERROR: node_modules is not a symlink. Just run 'pnpm install' normally."
	exit 1
fi

WORK_DIR="${TMPDIR:-/tmp}/fuwari-pnpm-$$"
trap 'rm -rf "$WORK_DIR"' EXIT

mkdir -p "$WORK_DIR"
cp "$PROJECT_DIR/package.json" "$PROJECT_DIR/pnpm-lock.yaml" "$WORK_DIR/"

echo ">>> Installing in $WORK_DIR (ext4)..."
(
	cd "$WORK_DIR"
	corepack pnpm install "$@"
)

echo ">>> Moving node_modules into $NM_TARGET..."
rm -rf "$NM_TARGET"
mv "$WORK_DIR/node_modules" "$NM_TARGET"

echo ">>> Syncing updated lockfile back to project..."
cp "$WORK_DIR/pnpm-lock.yaml" "$PROJECT_DIR/pnpm-lock.yaml"

echo ">>> Done. node_modules is ready at $NM_TARGET"
