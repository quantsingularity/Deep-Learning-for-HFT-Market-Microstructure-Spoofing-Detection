"""
Kafka Consumer for Real-Time LOB Event Processing
Consumes market data from Kafka and sends to TEN-GNN API for inference
"""

import os
import json
import time
import logging
from typing import Dict
import signal

from kafka import KafkaConsumer
import redis
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LOBStreamingConsumer:
    """
    Kafka consumer for LOB events with real-time inference
    """

    def __init__(
        self,
        kafka_bootstrap_servers: str,
        kafka_topic: str,
        api_url: str,
        redis_host: str = "localhost",
    ):
        """
        Args:
            kafka_bootstrap_servers: Kafka bootstrap servers
            kafka_topic: Topic to consume from
            api_url: TEN-GNN API URL
            redis_host: Redis host for caching
        """
        self.kafka_bootstrap_servers = kafka_bootstrap_servers
        self.kafka_topic = kafka_topic
        self.api_url = api_url

        # Create Kafka consumer
        self.consumer = KafkaConsumer(
            kafka_topic,
            bootstrap_servers=kafka_bootstrap_servers,
            auto_offset_reset="latest",
            enable_auto_commit=True,
            group_id="ten-gnn-consumer-group",
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        )

        # Connect to Redis
        try:
            self.redis_client = redis.Redis(
                host=redis_host, port=6379, db=0, decode_responses=True
            )
            self.redis_client.ping()
            logger.info(f"✓ Connected to Redis at {redis_host}")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None

        # Statistics
        self.events_processed = 0
        self.alerts_generated = 0
        self.errors = 0
        self.start_time = time.time()

        # Shutdown flag
        self.should_shutdown = False

        logger.info(f"✓ Consumer initialized")
        logger.info(f"  - Kafka: {kafka_bootstrap_servers}")
        logger.info(f"  - Topic: {kafka_topic}")
        logger.info(f"  - API: {api_url}")

    def process_event(self, event: Dict) -> None:
        """
        Process a single LOB event

        Args:
            event: LOB event dictionary
        """
        try:
            # Send to API for inference
            response = requests.post(f"{self.api_url}/predict", json=event, timeout=5.0)

            if response.status_code == 200:
                result = response.json()

                self.events_processed += 1

                # Check for alert
                if result.get("alert", False):
                    self.alerts_generated += 1

                    logger.warning(
                        f"🚨 SPOOFING DETECTED - "
                        f"Asset: {event.get('asset', 'UNKNOWN')}, "
                        f"Confidence: {result['confidence']:.4f}, "
                        f"Latency: {result['inference_time_ms']:.2f}ms"
                    )

                    # Store alert
                    self.store_alert(event, result)

                # Log progress every 1000 events
                if self.events_processed % 1000 == 0:
                    self.log_statistics()

            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                self.errors += 1

        except requests.exceptions.Timeout:
            logger.error(f"API request timeout")
            self.errors += 1

        except Exception as e:
            logger.error(f"Failed to process event: {e}")
            self.errors += 1

    def store_alert(self, event: Dict, result: Dict) -> None:
        """
        Store alert in Redis and persistent storage

        Args:
            event: Original LOB event
            result: Inference result
        """
        alert_data = {
            "timestamp": event.get("timestamp", time.time() * 1000),
            "asset": event.get("asset", "UNKNOWN"),
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "inference_time_ms": result["inference_time_ms"],
            "mid_price": event.get("mid_price"),
            "spread": event.get("spread"),
        }

        # Store in Redis
        if self.redis_client:
            try:
                alert_key = (
                    f"alert:{alert_data['asset']}:{int(alert_data['timestamp'])}"
                )
                self.redis_client.setex(
                    alert_key, 3600, json.dumps(alert_data)  # 1 hour TTL
                )
            except Exception as e:
                logger.warning(f"Failed to cache alert: {e}")

        # Append to alert log file
        try:
            alert_dir = "/app/alerts"
            os.makedirs(alert_dir, exist_ok=True)

            alert_file = os.path.join(alert_dir, "alerts.jsonl")
            with open(alert_file, "a") as f:
                f.write(json.dumps(alert_data) + "\n")

        except Exception as e:
            logger.warning(f"Failed to write alert to file: {e}")

    def log_statistics(self) -> None:
        """Log processing statistics"""
        uptime = time.time() - self.start_time
        rate = self.events_processed / uptime if uptime > 0 else 0

        logger.info(
            f"Statistics - "
            f"Events: {self.events_processed}, "
            f"Alerts: {self.alerts_generated}, "
            f"Errors: {self.errors}, "
            f"Rate: {rate:.2f} events/sec, "
            f"Uptime: {uptime:.1f}s"
        )

    def run(self) -> None:
        """
        Start consuming events
        """
        logger.info("Starting event consumption...")
        logger.info(f"Listening on topic: {self.kafka_topic}")

        try:
            for message in self.consumer:
                if self.should_shutdown:
                    logger.info("Shutdown signal received, stopping...")
                    break

                event = message.value
                self.process_event(event)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")

        except Exception as e:
            logger.error(f"Consumer error: {e}")

        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Cleanup resources"""
        logger.info("Cleaning up...")

        # Log final statistics
        self.log_statistics()

        # Close consumer
        self.consumer.close()

        # Close Redis
        if self.redis_client:
            self.redis_client.close()

        logger.info("Consumer shutdown complete")

    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}")
        self.should_shutdown = True


def main():
    """Main entry point"""
    # Get configuration from environment
    kafka_bootstrap_servers = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
    kafka_topic = os.getenv("KAFKA_TOPIC", "lob_events")
    api_url = os.getenv("API_URL", "http://localhost:8000")
    redis_host = os.getenv("REDIS_HOST", "localhost")

    logger.info("=" * 70)
    logger.info("TEN-GNN Streaming Consumer")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  Kafka: {kafka_bootstrap_servers}")
    logger.info(f"  Topic: {kafka_topic}")
    logger.info(f"  API: {api_url}")
    logger.info(f"  Redis: {redis_host}")
    logger.info("=" * 70)

    # Wait for API to be ready
    logger.info("Waiting for API to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{api_url}/health", timeout=2.0)
            if response.status_code == 200:
                logger.info("✓ API is ready")
                break
        except:
            pass
        time.sleep(2)
    else:
        logger.error("API failed to become ready")
        return

    # Create and run consumer
    consumer = LOBStreamingConsumer(
        kafka_bootstrap_servers=kafka_bootstrap_servers,
        kafka_topic=kafka_topic,
        api_url=api_url,
        redis_host=redis_host,
    )

    # Register signal handlers
    signal.signal(signal.SIGINT, consumer.signal_handler)
    signal.signal(signal.SIGTERM, consumer.signal_handler)

    # Start consuming
    consumer.run()


if __name__ == "__main__":
    main()
