#!/bin/bash

# Dependabot PR Review Workflow Script
# This script helps review all Dependabot PRs efficiently

echo "🤖 Dependabot PR Review Workflow"
echo "=================================="

# Configuration
EANAME_REPO="EAName/algorithmic_trading"
PARALLEL_REPO="ParallelLLC/algorithmic_trading"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check PR status
check_pr_status() {
    local repo=$1
    local pr_number=$2
    local pr_title=$3
    
    echo -e "\n${BLUE}📋 Reviewing PR #$pr_number: $pr_title${NC}"
    echo "Repository: $repo"
    
    # Open PR in browser
    echo -e "${YELLOW}🔗 Opening PR in browser...${NC}"
    open "https://github.com/$repo/pull/$pr_number"
    
    # Wait for user to review
    echo -e "${YELLOW}⏳ Review the PR in your browser, then press Enter to continue...${NC}"
    read -r
    
    # Ask for decision
    echo -e "${GREEN}✅ Decision for PR #$pr_number:${NC}"
    echo "1. Approve"
    echo "2. Request changes"
    echo "3. Comment only"
    echo "4. Skip for now"
    
    read -p "Enter your choice (1-4): " choice
    
    case $choice in
        1)
            echo -e "${GREEN}✅ Approved PR #$pr_number${NC}"
            echo "$repo PR #$pr_number: APPROVED - $pr_title" >> review_log.txt
            ;;
        2)
            echo -e "${RED}❌ Requested changes for PR #$pr_number${NC}"
            echo "$repo PR #$pr_number: CHANGES_REQUESTED - $pr_title" >> review_log.txt
            ;;
        3)
            echo -e "${YELLOW}💬 Commented on PR #$pr_number${NC}"
            echo "$repo PR #$pr_number: COMMENTED - $pr_title" >> review_log.txt
            ;;
        4)
            echo -e "${YELLOW}⏭️ Skipped PR #$pr_number${NC}"
            echo "$repo PR #$pr_number: SKIPPED - $pr_title" >> review_log.txt
            ;;
        *)
            echo -e "${RED}❌ Invalid choice, skipping...${NC}"
            echo "$repo PR #$pr_number: SKIPPED - $pr_title" >> review_log.txt
            ;;
    esac
}

# Function to run local tests
run_local_tests() {
    echo -e "\n${BLUE}🧪 Running local tests...${NC}"
    
    # Check if we're in the right directory
    if [ ! -f "requirements.txt" ]; then
        echo -e "${RED}❌ Not in algorithmic_trading directory${NC}"
        return 1
    fi
    
    # Run tests
    echo "Running pytest..."
    python -m pytest tests/ -v --tb=short
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Tests passed${NC}"
    else
        echo -e "${RED}❌ Tests failed${NC}"
        return 1
    fi
    
    # Check code formatting
    echo "Checking code formatting..."
    python -m black --check .
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Code formatting OK${NC}"
    else
        echo -e "${YELLOW}⚠️ Code formatting issues found${NC}"
    fi
    
    # Check for security issues
    echo "Checking for security issues..."
    if command -v safety &> /dev/null; then
        safety check
    else
        echo -e "${YELLOW}⚠️ Safety not installed, skipping security check${NC}"
    fi
}

# Function to check Docker build
check_docker_build() {
    echo -e "\n${BLUE}🐳 Testing Docker build...${NC}"
    
    # Build Docker image
    docker build -t test-algorithmic-trading .
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✅ Docker build successful${NC}"
        
        # Test Docker image
        docker run --rm test-algorithmic-trading python -c "print('Docker test passed')"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✅ Docker image test passed${NC}"
        else
            echo -e "${RED}❌ Docker image test failed${NC}"
        fi
        
        # Clean up
        docker rmi test-algorithmic-trading
    else
        echo -e "${RED}❌ Docker build failed${NC}"
        return 1
    fi
}

# Main workflow
main() {
    echo -e "${GREEN}🚀 Starting Dependabot PR Review Workflow${NC}"
    
    # Initialize review log
    echo "# Dependabot PR Review Log - $(date)" > review_log.txt
    echo "" >> review_log.txt
    
    # Run local tests first
    run_local_tests
    
    # Check Docker build
    check_docker_build
    
    echo -e "\n${BLUE}📋 Reviewing EAName Repository PRs${NC}"
    echo "=================================="
    
    # EAName Repository PRs
    check_pr_status "$EANAME_REPO" "6" "docker(deps): bump python from 3.11-slim to 3.13-slim"
    check_pr_status "$EANAME_REPO" "5" "github-actions(deps): bump peter-evans/create-pull-request from 4 to 7"
    check_pr_status "$EANAME_REPO" "4" "github-actions(deps): bump peaceiris/actions-gh-pages from 3 to 4"
    check_pr_status "$EANAME_REPO" "3" "github-actions(deps): bump actions/upload-artifact from 3 to 4"
    check_pr_status "$EANAME_REPO" "2" "github-actions(deps): bump docker/login-action from 2 to 3"
    check_pr_status "$EANAME_REPO" "1" "github-actions(deps): bump github/codeql-action from 2 to 3"
    
    echo -e "\n${BLUE}📋 Reviewing ParallelLLC Repository PRs${NC}"
    echo "=================================="
    
    # ParallelLLC Repository PRs
    check_pr_status "$PARALLEL_REPO" "6" "docker(deps): bump python from 3.11-slim to 3.13-slim"
    check_pr_status "$PARALLEL_REPO" "5" "github-actions(deps): bump actions/setup-python from 4 to 5"
    check_pr_status "$PARALLEL_REPO" "4" "github-actions(deps): bump docker/login-action from 2 to 3"
    check_pr_status "$PARALLEL_REPO" "3" "github-actions(deps): bump docker/metadata-action from 4 to 5"
    check_pr_status "$PARALLEL_REPO" "2" "github-actions(deps): bump peter-evans/create-pull-request from 4 to 7"
    check_pr_status "$PARALLEL_REPO" "1" "github-actions(deps): bump docker/build-push-action from 4 to 6"
    
    # Summary
    echo -e "\n${GREEN}✅ Review workflow completed!${NC}"
    echo -e "${BLUE}📝 Review log saved to: review_log.txt${NC}"
    echo -e "\n${YELLOW}📊 Summary:${NC}"
    cat review_log.txt
}

# Run the workflow
main 