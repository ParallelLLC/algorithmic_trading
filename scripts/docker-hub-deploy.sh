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

# Default values
DOCKER_USERNAME=""
IMAGE_NAME="algorithmic-trading"
TAG="latest"
REGISTRY="docker.io"

# Function to show help
show_help() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -u, --username USERNAME    Docker Hub username (required)"
    echo "  -i, --image IMAGE_NAME     Image name (default: algorithmic-trading)"
    echo "  -t, --tag TAG              Tag (default: latest)"
    echo "  -r, --registry REGISTRY    Registry URL (default: docker.io)"
    echo "  -h, --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -u myusername"
    echo "  $0 -u myusername -i my-trading-system -t v1.0.0"
    echo ""
    echo "Environment Variables:"
    echo "  DOCKER_USERNAME            Set your Docker Hub username"
    echo "  DOCKER_PASSWORD            Set your Docker Hub password/token"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--username)
            DOCKER_USERNAME="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            TAG="$2"
            shift 2
            ;;
        -r|--registry)
            REGISTRY="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if username is provided
if [ -z "$DOCKER_USERNAME" ]; then
    if [ -n "$DOCKER_USERNAME" ]; then
        DOCKER_USERNAME="$DOCKER_USERNAME"
    else
        print_error "Docker Hub username is required!"
        echo "Use -u option or set DOCKER_USERNAME environment variable"
        show_help
        exit 1
    fi
fi

# Function to build the image
build_image() {
    print_status "Building Docker image..."
    docker build -t ${IMAGE_NAME}:${TAG} .
    if [ $? -eq 0 ]; then
        print_success "Docker image built successfully"
    else
        print_error "Failed to build Docker image"
        exit 1
    fi
}

# Function to login to Docker Hub
login_to_dockerhub() {
    print_status "Logging in to Docker Hub..."
    if [ -n "$DOCKER_PASSWORD" ]; then
        echo "$DOCKER_PASSWORD" | docker login -u "$DOCKER_USERNAME" --password-stdin
    else
        docker login -u "$DOCKER_USERNAME"
    fi
    
    if [ $? -eq 0 ]; then
        print_success "Successfully logged in to Docker Hub"
    else
        print_error "Failed to login to Docker Hub"
        exit 1
    fi
}

# Function to tag the image
tag_image() {
    local full_image_name="${REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
    print_status "Tagging image as: $full_image_name"
    docker tag ${IMAGE_NAME}:${TAG} "$full_image_name"
    if [ $? -eq 0 ]; then
        print_success "Image tagged successfully"
        echo "$full_image_name"
    else
        print_error "Failed to tag image"
        exit 1
    fi
}

# Function to push the image
push_image() {
    local full_image_name="${REGISTRY}/${DOCKER_USERNAME}/${IMAGE_NAME}:${TAG}"
    print_status "Pushing image to Docker Hub: $full_image_name"
    docker push "$full_image_name"
    if [ $? -eq 0 ]; then
        print_success "Image pushed successfully to Docker Hub!"
        print_status "You can now pull it with:"
        echo "  docker pull $full_image_name"
        print_status "Or use it in docker-compose with:"
        echo "  image: $full_image_name"
    else
        print_error "Failed to push image to Docker Hub"
        exit 1
    fi
}

# Function to run tests before pushing
run_tests() {
    print_status "Running tests before deployment..."
    docker run --rm -v $(pwd):/app ${IMAGE_NAME}:${TAG} pytest -v --tb=short
    if [ $? -eq 0 ]; then
        print_success "Tests passed"
    else
        print_warning "Some tests failed, but continuing with deployment..."
    fi
}

# Function to clean up local images
cleanup() {
    print_status "Cleaning up local images..."
    docker rmi ${IMAGE_NAME}:${TAG} 2>/dev/null || true
    print_success "Cleanup completed"
}

# Main deployment process
main() {
    print_status "Starting Docker Hub deployment..."
    print_status "Username: $DOCKER_USERNAME"
    print_status "Image: $IMAGE_NAME"
    print_status "Tag: $TAG"
    print_status "Registry: $REGISTRY"
    echo ""
    
    # Build the image
    build_image
    
    # Run tests (optional)
    read -p "Run tests before deployment? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        run_tests
    fi
    
    # Login to Docker Hub
    login_to_dockerhub
    
    # Tag the image
    local full_image_name=$(tag_image)
    
    # Push the image
    push_image
    
    # Cleanup
    read -p "Clean up local images? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cleanup
    fi
    
    print_success "Deployment completed successfully!"
    print_status "Your image is now available at:"
    echo "  https://hub.docker.com/r/${DOCKER_USERNAME}/${IMAGE_NAME}"
}

# Run main function
main "$@" 