#!/bin/bash

set -e

BRANCH="feature/comprehensive-test-suite-fixes"
BASE="main"

# Push to both remotes
echo "Pushing to origin..."
git push origin $BRANCH

echo "Pushing to eaname..."
git push eaname $BRANCH

# Open PR creation pages in browser
PR_URL1="https://github.com/ParallelLLC/algorithmic_trading/compare/$BASE...$BRANCH?expand=1"
PR_URL2="https://github.com/EAName/algorithmic_trading/compare/$BASE...$BRANCH?expand=1"

echo "Opening PR creation pages in browser..."
open "$PR_URL1"
open "$PR_URL2"

echo "Done." 