# Docker Hub Setup Guide

This guide will help you deploy your algorithmic trading system to Docker Hub and use it from the cloud.

## Prerequisites

1. **Docker Hub Account**: Create an account at [hub.docker.com](https://hub.docker.com)
2. **Docker CLI**: Ensure Docker is installed and running
3. **Repository**: Create a repository on Docker Hub (optional, will be created automatically)

## Quick Start

### 1. Set Environment Variables

```bash
# Set your Docker Hub username
export DOCKER_USERNAME=yourusername

# Optionally set your password/token (for automated login)
export DOCKER_PASSWORD=yourpassword
```

### 2. Deploy to Docker Hub

```bash
# Deploy with default settings
./scripts/docker-build.sh deploy

# Or use the dedicated deployment script
./scripts/docker-hub-deploy.sh -u yourusername

# Deploy with custom image name and tag
./scripts/docker-hub-deploy.sh -u yourusername -i my-trading-system -t v1.0.0
```

### 3. Use from Docker Hub

```bash
# Start services using Docker Hub images
./scripts/docker-build.sh hub

# Or manually pull and run
docker pull yourusername/algorithmic-trading:latest
docker run -p 8000:8000 yourusername/algorithmic-trading:latest
```

## Detailed Instructions

### Step 1: Create Docker Hub Account

1. Go to [hub.docker.com](https://hub.docker.com)
2. Click "Sign Up" and create an account
3. Verify your email address
4. Create a repository (optional):
   - Click "Create Repository"
   - Name: `algorithmic-trading`
   - Description: "Algorithmic Trading System with FinRL"
   - Visibility: Public or Private

### Step 2: Generate Access Token (Recommended)

1. Go to Docker Hub → Account Settings → Security
2. Click "New Access Token"
3. Name: `algorithmic-trading-deploy`
4. Permissions: Read & Write
5. Copy the token (you won't see it again)

### Step 3: Configure Environment

```bash
# Add to your ~/.bashrc or ~/.zshrc
export DOCKER_USERNAME=yourusername
export DOCKER_PASSWORD=your_access_token
```

### Step 4: Deploy

```bash
# Build and deploy
./scripts/docker-hub-deploy.sh -u yourusername

# The script will:
# 1. Build the Docker image
# 2. Run tests (optional)
# 3. Login to Docker Hub
# 4. Tag the image
# 5. Push to Docker Hub
```

### Step 5: Verify Deployment

1. Check your Docker Hub repository: `https://hub.docker.com/r/yourusername/algorithmic-trading`
2. You should see your image with the latest tag

## Usage Examples

### Local Development with Docker Hub

```bash
# Start development environment
DOCKER_USERNAME=yourusername ./scripts/docker-build.sh hub

# Access services:
# - Jupyter Lab: http://localhost:8888
# - Trading System: http://localhost:8000
```

### Production Deployment

```bash
# Deploy to production
docker run -d \
  --name trading-prod \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/config.yaml:/app/config.yaml:ro \
  yourusername/algorithmic-trading:latest
```

### Using Docker Compose

```bash
# Create .env file
echo "DOCKER_USERNAME=yourusername" > .env
echo "TAG=latest" >> .env

# Start services
docker compose -f docker-compose.hub.yml up -d
```

## Advanced Usage

### Multiple Tags

```bash
# Deploy with version tags
./scripts/docker-hub-deploy.sh -u yourusername -t v1.0.0
./scripts/docker-hub-deploy.sh -u yourusername -t v1.1.0
./scripts/docker-hub-deploy.sh -u yourusername -t latest
```

### Custom Image Names

```bash
# Deploy with custom name
./scripts/docker-hub-deploy.sh -u yourusername -i my-trading-bot -t production
```

### Automated Deployment

```bash
# Create a deployment script
cat > deploy.sh << 'EOF'
#!/bin/bash
export DOCKER_USERNAME=yourusername
export DOCKER_PASSWORD=your_token
./scripts/docker-hub-deploy.sh -u $DOCKER_USERNAME -t $(date +%Y%m%d)
EOF

chmod +x deploy.sh
./deploy.sh
```

## Troubleshooting

### Login Issues

```bash
# Manual login
docker login -u yourusername

# Check login status
docker info | grep Username
```

### Push Failures

```bash
# Check if repository exists
curl https://hub.docker.com/v2/repositories/yourusername/algorithmic-trading/

# Create repository manually if needed
# Go to Docker Hub → Create Repository
```

### Permission Issues

```bash
# Check Docker permissions
docker ps

# If permission denied, add user to docker group
sudo usermod -aG docker $USER
# Log out and back in
```

### Network Issues

```bash
# Check Docker Hub connectivity
curl -I https://hub.docker.com

# Use different DNS if needed
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf
```

## Best Practices

### 1. Use Access Tokens
- Never use your password in scripts
- Generate access tokens with limited permissions
- Rotate tokens regularly

### 2. Tag Strategically
- Use semantic versioning: `v1.0.0`, `v1.1.0`
- Keep `latest` tag updated
- Use date tags for testing: `2024-01-15`

### 3. Security
- Don't commit credentials to git
- Use environment variables
- Consider private repositories for sensitive code

### 4. Automation
- Set up CI/CD pipelines
- Automate testing before deployment
- Use GitHub Actions or similar

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/docker-hub.yml
name: Deploy to Docker Hub

on:
  push:
    tags: ['v*']

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Login to Docker Hub
        uses: docker/login-action@v1
        with:
          username: ${{ secrets.DOCKER_USERNAME }}
          password: ${{ secrets.DOCKER_PASSWORD }}
      - name: Build and push
        run: |
          docker build -t ${{ secrets.DOCKER_USERNAME }}/algorithmic-trading:${{ github.ref_name }} .
          docker push ${{ secrets.DOCKER_USERNAME }}/algorithmic-trading:${{ github.ref_name }}
```

## Support

- **Docker Hub Documentation**: [docs.docker.com/docker-hub](https://docs.docker.com/docker-hub/)
- **Repository Issues**: Create an issue in this repository
- **Community**: Join Docker Hub community forums

## Next Steps

1. **Set up automated deployment** with CI/CD
2. **Create multiple environments** (dev, staging, prod)
3. **Monitor usage** with Docker Hub analytics
4. **Share with team** by adding collaborators
5. **Scale deployment** to multiple servers 