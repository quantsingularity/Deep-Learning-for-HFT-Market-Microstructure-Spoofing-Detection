"""
Production FastAPI Server for TEN-GNN Spoofing Detection
Provides REST API for real-time inference with monitoring
"""

import os
import sys
import time
import json
import logging
from typing import List, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import redis
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi.responses import PlainTextResponse

from code.models.transformer_encoder import TransformerEncoderNetwork
from code.deployment.inference import RealTimeDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ============================================
# Pydantic Models
# ============================================


class LOBEvent(BaseModel):
    """Single LOB event for inference"""

    timestamp: float = Field(..., description="Event timestamp in milliseconds")
    asset: str = Field(..., description="Asset symbol")
    bid_prices: List[float] = Field(..., description="Top 10 bid prices")
    ask_prices: List[float] = Field(..., description="Top 10 ask prices")
    bid_volumes: List[float] = Field(..., description="Volumes at bid levels")
    ask_volumes: List[float] = Field(..., description="Volumes at ask levels")
    mid_price: Optional[float] = None
    spread: Optional[float] = None
    order_imbalance: Optional[float] = None


class BatchInferenceRequest(BaseModel):
    """Batch inference request"""

    events: List[LOBEvent]


class InferenceResponse(BaseModel):
    """Inference response"""

    prediction: int
    confidence: float
    inference_time_ms: float
    timestamp: float
    alert: bool = False


class HealthResponse(BaseModel):
    """Health check response"""

    status: str
    device: str
    model_loaded: bool
    redis_connected: bool
    uptime_seconds: float


# ============================================
# Prometheus Metrics
# ============================================

INFERENCE_COUNTER = Counter("ten_gnn_inference_total", "Total number of inferences")

INFERENCE_LATENCY = Histogram(
    "ten_gnn_inference_latency_seconds",
    "Inference latency in seconds",
    buckets=[0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1],
)

SPOOFING_DETECTED = Counter(
    "ten_gnn_spoofing_detected_total", "Total number of spoofing detections"
)

ACTIVE_CONNECTIONS = Gauge("ten_gnn_active_connections", "Number of active connections")

# ============================================
# Application Setup
# ============================================

app = FastAPI(
    title="TEN-GNN Spoofing Detection API",
    description="Real-time HFT spoofing detection using Transformer-Encoder Networks and Graph Neural Networks",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global state
class AppState:
    def __init__(self):
        self.detector: Optional[RealTimeDetector] = None
        self.redis_client: Optional[redis.Redis] = None
        self.start_time = time.time()
        self.device = os.getenv("DEVICE", "cpu")


state = AppState()

# ============================================
# Startup/Shutdown Events
# ============================================


@app.on_event("startup")
async def startup_event():
    """Initialize model and connections"""
    logger.info("Starting TEN-GNN API Server...")

    # Get configuration
    model_path = os.getenv(
        "MODEL_PATH", "/app/pretrained_models/ten_model_synthetic.pth"
    )
    config_path = os.getenv("CONFIG_PATH", "/app/configs/config.json")
    device = state.device

    logger.info(f"Device: {device}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Config path: {config_path}")

    try:
        # Load config
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create model
        model = TransformerEncoderNetwork(
            input_dim=config["model"]["input_dim"],
            d_model=config["model"]["d_model"],
            num_layers=config["model"]["num_layers"],
            num_heads=config["model"]["num_heads"],
            d_ff=config["model"]["d_ff"],
            dropout=config["model"]["dropout"],
            max_seq_len=config["data"]["window_size"],
            num_classes=2,
        )

        # Initialize detector
        state.detector = RealTimeDetector(
            model=model,
            model_path=model_path,
            device=device,
            window_size=config["data"]["window_size"],
            confidence_threshold=0.8,
        )

        logger.info("✓ Model loaded successfully")

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

    # Connect to Redis
    try:
        redis_host = os.getenv("REDIS_HOST", "localhost")
        state.redis_client = redis.Redis(
            host=redis_host, port=6379, db=0, decode_responses=True
        )
        state.redis_client.ping()
        logger.info(f"✓ Connected to Redis at {redis_host}")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}")
        state.redis_client = None

    logger.info("TEN-GNN API Server started successfully!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down TEN-GNN API Server...")

    if state.redis_client:
        state.redis_client.close()

    logger.info("Shutdown complete")


# ============================================
# API Endpoints
# ============================================


@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "service": "TEN-GNN Spoofing Detection",
        "version": "1.0.0",
        "status": "running",
        "device": state.device,
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if state.detector else "unhealthy",
        device=state.device,
        model_loaded=state.detector is not None,
        redis_connected=state.redis_client is not None,
        uptime_seconds=time.time() - state.start_time,
    )


@app.post("/predict", response_model=InferenceResponse, tags=["Inference"])
async def predict_single(event: LOBEvent):
    """
    Predict spoofing for a single LOB event

    This endpoint processes a single LOB event and returns a prediction.
    The detector maintains an internal sliding window for temporal context.
    """
    if not state.detector:
        raise HTTPException(status_code=503, detail="Model not loaded")

    INFERENCE_COUNTER.inc()

    start_time = time.time()

    try:
        # Convert to dict
        event_dict = event.dict()

        # Process event
        alert = state.detector.process_event(event_dict)

        # Get prediction (even if no alert)
        prediction, confidence, inference_time = state.detector.predict()

        inference_latency = time.time() - start_time
        INFERENCE_LATENCY.observe(inference_latency)

        if alert:
            SPOOFING_DETECTED.inc()

            # Cache alert in Redis
            if state.redis_client:
                try:
                    alert_key = f"alert:{event.asset}:{int(event.timestamp)}"
                    state.redis_client.setex(
                        alert_key, 3600, json.dumps(alert)  # 1 hour TTL
                    )
                except Exception as e:
                    logger.warning(f"Failed to cache alert: {e}")

        return InferenceResponse(
            prediction=prediction,
            confidence=confidence,
            inference_time_ms=inference_time,
            timestamp=event.timestamp,
            alert=alert is not None,
        )

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch", response_model=List[InferenceResponse], tags=["Inference"])
async def predict_batch(request: BatchInferenceRequest):
    """
    Batch inference for multiple LOB events

    Processes multiple events efficiently. Useful for backtesting or
    processing historical data.
    """
    if not state.detector:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if len(request.events) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 events per batch")

    results = []

    for event in request.events:
        event_dict = event.dict()
        alert = state.detector.process_event(event_dict)
        prediction, confidence, inference_time = state.detector.predict()

        INFERENCE_COUNTER.inc()

        if alert:
            SPOOFING_DETECTED.inc()

        results.append(
            InferenceResponse(
                prediction=prediction,
                confidence=confidence,
                inference_time_ms=inference_time,
                timestamp=event.timestamp,
                alert=alert is not None,
            )
        )

    return results


@app.post("/reset", tags=["Control"])
async def reset_detector():
    """
    Reset the detector's internal state

    Clears the sliding window buffer. Use when switching to a new asset
    or time period.
    """
    if not state.detector:
        raise HTTPException(status_code=503, detail="Model not loaded")

    state.detector.reset_buffers()

    return {"status": "success", "message": "Detector reset"}


@app.get("/stats", tags=["Monitoring"])
async def get_stats():
    """
    Get performance statistics

    Returns inference latency metrics and alert counts.
    """
    if not state.detector:
        raise HTTPException(status_code=503, detail="Model not loaded")

    stats = state.detector.get_performance_stats()

    return {
        "status": "success",
        "stats": stats,
        "uptime_seconds": time.time() - state.start_time,
    }


@app.get("/metrics", response_class=PlainTextResponse, tags=["Monitoring"])
async def metrics():
    """
    Prometheus metrics endpoint
    """
    return generate_latest()


@app.get("/alerts/recent", tags=["Monitoring"])
async def get_recent_alerts(limit: int = 100):
    """
    Get recent alerts from Redis cache
    """
    if not state.redis_client:
        raise HTTPException(status_code=503, detail="Redis not connected")

    try:
        # Get all alert keys
        keys = state.redis_client.keys("alert:*")

        # Get alert data
        alerts = []
        for key in keys[:limit]:
            alert_data = state.redis_client.get(key)
            if alert_data:
                alerts.append(json.loads(alert_data))

        # Sort by timestamp
        alerts.sort(key=lambda x: x.get("timestamp", 0), reverse=True)

        return {"status": "success", "count": len(alerts), "alerts": alerts[:limit]}

    except Exception as e:
        logger.error(f"Failed to retrieve alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Main Entry Point
# ============================================

if __name__ == "__main__":
    import uvicorn
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument(
        "--reload", action="store_true", help="Auto-reload on code changes"
    )

    args = parser.parse_args()

    os.environ["DEVICE"] = args.device

    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
