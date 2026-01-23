#!/bin/bash
# Quick Start Script for TEN-GNN Production System

set -e

echo "=========================================="
echo "TEN-GNN Quick Start Script"
echo "=========================================="
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

echo "✓ Docker found"

# Check docker-compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    exit 1
fi

echo "✓ docker-compose found"

# Create .env if it doesn't exist
if [ ! -f .env ]; then
    echo ""
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "⚠️  Please edit .env file and set secure passwords!"
    echo "   Run: nano .env"
    read -p "Press Enter to continue..."
fi

# Create necessary directories
echo ""
echo "Creating directories..."
mkdir -p pretrained_models logs data alerts benchmarks/results

# Create placeholder for pretrained model
if [ ! -f pretrained_models/ten_model_synthetic.pth ]; then
    echo ""
    echo "⚠️  Pre-trained model not found!"
    echo "   You need to either:"
    echo "   1. Train a model: python scripts/train_pretrained_model.py"
    echo "   2. Download pre-trained weights"
    echo ""
    read -p "Do you want to continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Select deployment profile
echo ""
echo "Select deployment profile:"
echo "  1) CPU only (recommended for testing)"
echo "  2) GPU (requires NVIDIA GPU + docker runtime)"
echo "  3) Full stack with monitoring"
echo ""
read -p "Enter choice [1-3]: " profile_choice

case $profile_choice in
    1)
        PROFILE="cpu"
        ;;
    2)
        # Check for NVIDIA GPU
        if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
            echo "❌ NVIDIA GPU or docker runtime not available"
            exit 1
        fi
        PROFILE="gpu"
        ;;
    3)
        PROFILE="cpu --profile streaming --profile monitoring"
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac

# Build and start services
echo ""
echo "Building Docker images..."
docker-compose build

echo ""
echo "Starting services..."
docker-compose --profile $PROFILE up -d

# Wait for services to be ready
echo ""
echo "Waiting for services to start..."
sleep 10

# Health check
echo ""
echo "Checking API health..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health &> /dev/null; then
        echo "✓ API is healthy!"
        break
    fi
    echo "Waiting... ($i/30)"
    sleep 2
done

# Show status
echo ""
echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo ""
docker-compose ps

echo ""
echo "Access points:"
echo "  - API: http://localhost:8000"
echo "  - API Docs: http://localhost:8000/docs"

if [[ $PROFILE == *"monitoring"* ]]; then
    echo "  - Grafana: http://localhost:3000"
    echo "  - Prometheus: http://localhost:9090"
fi

echo ""
echo "Test the API:"
echo '  curl http://localhost:8000/health'
echo ""
echo "View logs:"
echo "  docker-compose logs -f"
echo ""
echo "Stop services:"
echo "  docker-compose down"
echo ""
