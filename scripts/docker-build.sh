#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to build the Docker image
build_image() {
    print_status "Building Docker image..."
    docker build -t algorithmic-trading:latest .
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to run tests in Docker
run_tests() {
    print_status "Running tests in Docker..."
    docker run --rm -v $(pwd):/app algorithmic-trading:latest pytest -v
    if [ $? -eq 0 ]; then
        print_success "Tests passed"
    else
        print_error "Tests failed"
        exit 1
    fi
}

# Function to start development environment
start_dev() {
    print_status "Starting development environment..."
    docker-compose -f docker-compose.dev.yml up -d
    print_success "Development environment started"
    print_status "Jupyter Lab available at: http://localhost:8888"
    print_status "Trading system available at: http://localhost:8000"
    print_status "TensorBoard available at: http://localhost:6006"
}

# Function to start production environment
start_prod() {
    print_status "Starting production environment..."
    docker-compose -f docker-compose.prod.yml up -d
    print_success "Production environment started"
    print_status "Trading system available at: http://localhost:8000"
    print_status "Grafana available at: http://localhost:3000 (admin/admin)"
    print_status "Prometheus available at: http://localhost:9090"
}

# Function to stop all containers
stop_all() {
    print_status "Stopping all containers..."
    docker-compose -f docker-compose.yml down
    docker-compose -f docker-compose.dev.yml down
    docker-compose -f docker-compose.prod.yml down
    print_success "All containers stopped"
}

# Function to clean up Docker resources
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker system prune -f
    docker volume prune -f
    print_success "Cleanup completed"
}

# Function to show logs
show_logs() {
    local service=${1:-trading-system}
    print_status "Showing logs for $service..."
    docker-compose logs -f $service
}

# Function to run a specific command in the container
run_command() {
    local command="$1"
    print_status "Running command: $command"
    docker run --rm -v $(pwd):/app algorithmic-trading:latest $command
}

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  build       Build the Docker image"
    echo "  test        Run tests in Docker"
    echo "  dev         Start development environment"
    echo "  prod        Start production environment"
    echo "  hub         Start using Docker Hub images"
    echo "  stop        Stop all containers"
    echo "  cleanup     Clean up Docker resources"
    echo "  logs [SVC]  Show logs for a service (default: trading-system)"
    echo "  run CMD     Run a specific command in the container"
    echo "  deploy      Deploy to Docker Hub (requires docker-hub-deploy.sh)"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 build"
    echo "  $0 dev"
    echo "  $0 hub"
    echo "  $0 logs"
    echo "  $0 run 'python demo.py'"
    echo "  $0 deploy"
}

# Main script logic
case "${1:-help}" in
    build)
        build_image
        ;;
    test)
        build_image
        run_tests
        ;;
    dev)
        build_image
        start_dev
        ;;
    prod)
        build_image
        start_prod
        ;;
    stop)
        stop_all
        ;;
    cleanup)
        cleanup
        ;;
    logs)
        show_logs $2
        ;;
    run)
        if [ -z "$2" ]; then
            print_error "No command specified"
            show_help
            exit 1
        fi
        build_image
        run_command "$2"
        ;;
    hub)
        print_status "Starting services using Docker Hub images..."
        if [ -z "$DOCKER_USERNAME" ]; then
            print_error "DOCKER_USERNAME environment variable not set"
            print_status "Please set it: export DOCKER_USERNAME=yourusername"
            exit 1
        fi
        docker compose -f docker-compose.hub.yml up -d
        print_success "Docker Hub services started"
        print_status "Trading system available at: http://localhost:8000"
        print_status "Jupyter Lab available at: http://localhost:8888"
        ;;
    deploy)
        print_status "Deploying to Docker Hub..."
        if [ -f "scripts/docker-hub-deploy.sh" ]; then
            ./scripts/docker-hub-deploy.sh "$@"
        else
            print_error "docker-hub-deploy.sh not found"
            exit 1
        fi
        ;;
    help|*)
        show_help
        ;;
esac 