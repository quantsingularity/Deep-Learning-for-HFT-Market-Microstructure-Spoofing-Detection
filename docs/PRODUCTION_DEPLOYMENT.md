# Production Deployment Guide

## Table of Contents

1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Deployment Options](#deployment-options)
4. [Configuration](#configuration)
5. [Monitoring](#monitoring)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### CPU Deployment

```bash
# Start all services (CPU mode)
docker-compose --profile cpu up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f ten-gnn-cpu

# Test API
curl http://localhost:8000/health
```

### GPU Deployment

```bash
# Ensure NVIDIA Docker runtime is installed
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi

# Start GPU services
docker-compose --profile gpu up -d

# Check GPU usage
docker exec ten-gnn-gpu nvidia-smi
```

### With Streaming (Kafka)

```bash
# Start complete stack
docker-compose --profile cpu --profile streaming up -d

# Send test event to Kafka
docker exec -it ten-gnn-kafka kafka-console-producer --broker-list localhost:9092 --topic lob_events
```

## Architecture Overview

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│   Data Source   │─────▶│  Kafka (9092)    │─────▶│   Streaming     │
│  (FIX/ITCH)     │      │                  │      │   Consumer      │
└─────────────────┘      └──────────────────┘      └────────┬────────┘
                                                             │
                                                             ▼
                         ┌──────────────────┐      ┌─────────────────┐
                         │  Redis (6379)    │◀────▶│   TEN-GNN API   │
                         │  (Cache)         │      │   (8000/8001)   │
                         └──────────────────┘      └────────┬────────┘
                                                             │
                                                             ▼
                         ┌──────────────────┐      ┌─────────────────┐
                         │ PostgreSQL (5432)│◀─────│  Alert Storage  │
                         │ (Alerts DB)      │      │                 │
                         └──────────────────┘      └─────────────────┘
                                                             │
                                                             ▼
                         ┌──────────────────┐      ┌─────────────────┐
                         │ Prometheus (9090)│◀─────│   Monitoring    │
                         │                  │      │   & Metrics     │
                         └──────────────────┘      └─────────────────┘
                                                             │
                                                             ▼
                         ┌──────────────────┐      ┌─────────────────┐
                         │  Grafana (3000)  │      │  Visualization  │
                         │                  │      │   Dashboard     │
                         └──────────────────┘      └─────────────────┘
```

## Deployment Options

### 1. Standalone API (Minimum Setup)

For testing or low-volume deployments:

```bash
# CPU only
docker-compose --profile cpu up -d ten-gnn-cpu redis

# GPU
docker-compose --profile gpu up -d ten-gnn-gpu redis
```

**Requirements:**

- CPU: 4+ cores
- RAM: 8GB+
- GPU (optional): 4GB+ VRAM

### 2. Production with Streaming

Full stack with Kafka streaming:

```bash
docker-compose --profile cpu --profile streaming up -d
```

**Requirements:**

- CPU: 8+ cores
- RAM: 16GB+
- Storage: 50GB+ SSD

### 3. High-Availability Setup

Multiple replicas with load balancing:

```yaml
# Add to docker-compose.yml
services:
  ten-gnn-cpu:
    deploy:
      replicas: 3

  nginx-lb:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
```

## Configuration

### Environment Variables

Create `.env` file:

```bash
# Database
POSTGRES_PASSWORD=your_secure_password

# Grafana
GRAFANA_PASSWORD=your_secure_password

# Model
MODEL_PATH=/app/pretrained_models/ten_model_synthetic.pth
DEVICE=cuda

# Kafka
KAFKA_BOOTSTRAP_SERVERS=kafka:9092
KAFKA_TOPIC=lob_events

# Redis
REDIS_HOST=redis
```

### Model Configuration

Edit `configs/config.json`:

```json
{
  "model": {
    "type": "TEN",
    "input_dim": 47,
    "d_model": 256,
    "num_layers": 6
  },
  "deployment": {
    "device": "cuda",
    "inference_batch_size": 1,
    "max_latency_ms": 5.0,
    "confidence_threshold": 0.8
  }
}
```

## Monitoring

### Prometheus Metrics

Access at: `http://localhost:9090`

Key metrics:

- `ten_gnn_inference_total` - Total inferences
- `ten_gnn_inference_latency_seconds` - Latency histogram
- `ten_gnn_spoofing_detected_total` - Spoofing alerts

### Grafana Dashboards

Access at: `http://localhost:3000`

Default credentials:

- Username: `admin`
- Password: Set in `.env`

**Dashboards:**

1. **Inference Performance**
   - Latency percentiles (P50, P95, P99)
   - Throughput (requests/sec)
   - Error rates

2. **Detection Activity**
   - Alerts over time
   - False positive rate
   - Asset-wise detection

3. **System Health**
   - CPU/Memory usage
   - GPU utilization
   - Kafka lag

### Logs

```bash
# API logs
docker-compose logs -f ten-gnn-cpu

# Streaming consumer logs
docker-compose logs -f streaming-consumer

# All services
docker-compose logs -f
```

## Performance Benchmarks

### CPU (Intel Xeon)

| Batch Size | Mean Latency | P95 Latency | Throughput |
| ---------- | ------------ | ----------- | ---------- |
| 1          | 3.2ms        | 4.8ms       | 312/sec    |
| 8          | 18.4ms       | 24.1ms      | 435/sec    |
| 32         | 68.3ms       | 89.2ms      | 468/sec    |

### GPU (NVIDIA RTX 4090)

| Batch Size | Mean Latency | P95 Latency | Throughput |
| ---------- | ------------ | ----------- | ---------- |
| 1          | 0.9ms        | 1.4ms       | 1111/sec   |
| 8          | 2.1ms        | 3.2ms       | 3810/sec   |
| 32         | 6.8ms        | 9.1ms       | 4706/sec   |

**Recommendation:** Use batch_size=1 for real-time HFT, GPU for >1000 events/sec

## API Usage

### REST API

**Health Check:**

```bash
curl http://localhost:8000/health
```

**Single Prediction:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "timestamp": 1234567890000,
    "asset": "AAPL",
    "bid_prices": [150.25, 150.24, ...],
    "ask_prices": [150.26, 150.27, ...],
    "bid_volumes": [1000, 800, ...],
    "ask_volumes": [1200, 900, ...]
  }'
```

**Batch Prediction:**

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"events": [...]}'
```

## Troubleshooting

### Issue: High Latency

**Diagnosis:**

```bash
# Check system resources
docker stats

# Check GPU utilization (if using GPU)
nvidia-smi
```

**Solutions:**

- Use GPU deployment
- Reduce batch size
- Scale horizontally (multiple containers)

### Issue: High False Positive Rate

**Diagnosis:**

```bash
# Get performance stats
curl http://localhost:8000/stats
```

**Solutions:**

- Adjust confidence threshold in config
- Run false positive analysis:
  ```bash
  python adversarial/false_positive_analysis.py
  ```

### Issue: Kafka Consumer Lag

**Diagnosis:**

```bash
docker exec -it ten-gnn-kafka kafka-consumer-groups \
  --bootstrap-server localhost:9092 \
  --describe --group ten-gnn-consumer-group
```

**Solutions:**

- Increase consumer replicas
- Optimize processing logic
- Use batch prediction

## Security

### API Authentication

Add API key authentication:

```python
# In deployment/api/server.py
from fastapi.security import APIKeyHeader

api_key_header = APIKeyHeader(name="X-API-Key")

@app.post("/predict")
async def predict(api_key: str = Depends(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=403)
    # ...
```

### Network Security

```yaml
# Use internal networks
networks:
  internal:
    internal: true
  external:

services:
  ten-gnn-cpu:
    networks:
      - internal
      - external

  postgres:
    networks:
      - internal # No external access
```

## Maintenance

### Backup Model Checkpoints

```bash
# Backup directory
docker cp ten-gnn-cpu:/app/pretrained_models ./backups/

# Restore
docker cp ./backups/ten_model_synthetic.pth ten-gnn-cpu:/app/pretrained_models/
```

### Update Model

```bash
# Stop service
docker-compose stop ten-gnn-cpu

# Replace model
cp new_model.pth pretrained_models/ten_model_synthetic.pth

# Restart
docker-compose start ten-gnn-cpu
```

### Database Maintenance

```bash
# Backup alerts database
docker exec ten-gnn-postgres pg_dump -U tengnn spoofing_detection > backup.sql

# Restore
docker exec -i ten-gnn-postgres psql -U tengnn spoofing_detection < backup.sql
```
