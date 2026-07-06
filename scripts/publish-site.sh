#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${SITE_REPO:-$HOME/Documents/Abhishek21g.github.io}"

if [[ ! -d "$DEST/.git" ]]; then
  echo "error: main site repo not found at $DEST" >&2
  exit 1
fi

cd "$DEST"
git checkout gh-pages
git pull origin gh-pages --no-rebase

rsync -av --delete "$ROOT/site/" "$DEST/nuro-ftl-receipts/site/"
cp "$ROOT/index.html" "$DEST/nuro-ftl-receipts/index.html"

git add nuro-ftl-receipts/
if git diff --cached --quiet; then
  echo "No site changes to publish."
  exit 0
fi

git commit -m "Sync FTL Segment Receipts site from nuro-ftl-segment-receipts"
git pull origin gh-pages --no-rebase
git push origin gh-pages

echo "Published: https://enaguthi.com/nuro-ftl-receipts/site/"
