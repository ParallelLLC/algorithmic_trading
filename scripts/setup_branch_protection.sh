#!/bin/bash

# Branch Protection Setup Script
# Run this script to automatically configure branch protection

echo "🛡️ Setting up branch protection for algorithmic trading repository..."

# Configuration
REPO="EAName/algorithmic_trading"
BRANCH="main"
REQUIRED_REVIEWS=2
REQUIRED_CHECKS='["ci-cd/quality-check","ci-cd/test","ci-cd/security","ci-cd/backtesting"]'

echo "📋 Configuration:"
echo "  Repository: $REPO"
echo "  Branch: $BRANCH"
echo "  Required reviews: $REQUIRED_REVIEWS"
echo "  Required checks: $REQUIRED_CHECKS"

echo ""
echo "⚠️  IMPORTANT: You need a GitHub Personal Access Token with 'repo' permissions"
echo "   Get one from: https://github.com/settings/tokens"
echo ""

read -p "Enter your GitHub Personal Access Token: " GITHUB_TOKEN

if [ -z "$GITHUB_TOKEN" ]; then
    echo "❌ No token provided. Exiting."
    exit 1
fi

echo ""
echo "🔧 Applying branch protection rules..."

# Apply branch protection
curl -X PUT \
  -H "Authorization: token $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  "https://api.github.com/repos/$REPO/branches/$BRANCH/protection" \
  -d "{
    \"required_status_checks\": {
      \"strict\": true,
      \"contexts\": $REQUIRED_CHECKS
    },
    \"enforce_admins\": true,
    \"required_pull_request_reviews\": {
      \"required_approving_review_count\": $REQUIRED_REVIEWS,
      \"dismiss_stale_reviews\": true,
      \"require_code_owner_reviews\": true
    },
    \"restrictions\": null,
    \"allow_force_pushes\": false,
    \"allow_deletions\": false
  }"

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Branch protection successfully applied!"
    echo ""
    echo "📋 Applied rules:"
    echo "  - Require pull request before merging"
    echo "  - Require $REQUIRED_REVIEWS approvals"
    echo "  - Require code owner reviews"
    echo "  - Require status checks: $REQUIRED_CHECKS"
    echo "  - No force pushes allowed"
    echo "  - No deletions allowed"
    echo ""
    echo "🔗 View settings: https://github.com/$REPO/settings/branches"
else
    echo ""
    echo "❌ Failed to apply branch protection. Check your token and permissions."
fi 