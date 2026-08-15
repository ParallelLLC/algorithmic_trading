#!/usr/bin/env bash
#
# Assemble and push the Hugging Face Space.
#
# The Space gets only what it needs to run: app.py, the algotrader package, the
# Space card, and the minimal requirements file. The v1 agentic system and its
# heavy ML stack stay in this repo, so the Space builds in under a minute.
#
# Usage:
#   HF_TOKEN=hf_xxx ./scripts/deploy_hf_space.sh <hf-username>/<space-name>
#
# Create the Space first at https://huggingface.co/new-space (SDK: Gradio).

set -euo pipefail

SPACE_ID="${1:-}"
if [[ -z "$SPACE_ID" ]]; then
  echo "usage: HF_TOKEN=hf_xxx $0 <hf-username>/<space-name>" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "error: HF_TOKEN is not set. Create a write token at https://huggingface.co/settings/tokens" >&2
  exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

echo "==> Staging Space contents in $STAGING"
git clone --quiet "https://user:${HF_TOKEN}@huggingface.co/spaces/${SPACE_ID}" "$STAGING/space"
cd "$STAGING/space"

# Replace tracked content wholesale so deletions propagate, but keep .git.
find . -mindepth 1 -maxdepth 1 ! -name .git -exec rm -rf {} +

cp -r "$REPO_ROOT/algotrader" ./algotrader
cp "$REPO_ROOT/app.py" ./app.py
cp "$REPO_ROOT/LICENSE" ./LICENSE
cp "$REPO_ROOT/SPACE_README.md" ./README.md
cp "$REPO_ROOT/requirements-space.txt" ./requirements.txt
find ./algotrader -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

# A small test suite ships too: reviewers who check whether the statistics are
# real are exactly the audience worth convincing.
mkdir -p tests
cp "$REPO_ROOT/tests/test_v2_engine.py" \
   "$REPO_ROOT/tests/test_v2_validation.py" \
   "$REPO_ROOT/tests/test_v2_strategies.py" \
   "$REPO_ROOT/tests/test_v2_portfolio.py" tests/

echo "==> Files staged:"
find . -path ./.git -prune -o -type f -print | sed 's|^\./|  |'

git add -A
if git diff --cached --quiet; then
  echo "==> Space is already up to date; nothing to push."
  exit 0
fi

git -c user.email="deploy@localhost" -c user.name="space-deploy" \
    commit --quiet -m "Deploy algotrader $(python3 -c 'import sys; sys.path.insert(0, "'"$REPO_ROOT"'"); import algotrader; print(algotrader.__version__)')"
git push --quiet origin HEAD:main

echo "==> Pushed. Live at https://huggingface.co/spaces/${SPACE_ID}"
