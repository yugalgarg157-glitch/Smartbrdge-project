#!/usr/bin/env bash
set -euo pipefail

if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is not installed. Run: xcode-select --install"
  exit 1
fi

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/publish_github.sh <github_repo_url> [commit_message]"
  echo "Example: bash scripts/publish_github.sh https://github.com/user/smartbridge-project.git"
  exit 1
fi

REPO_URL="$1"
COMMIT_MSG="${2:-Initial commit: malaria detection project}"

if [[ ! -d .git ]]; then
  git init
fi

git add .
if git diff --cached --quiet; then
  echo "No staged changes to commit."
else
  git commit -m "$COMMIT_MSG"
fi

git branch -M main

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REPO_URL"
else
  git remote add origin "$REPO_URL"
fi

git push -u origin main

echo "Done. Project pushed to: $REPO_URL"
