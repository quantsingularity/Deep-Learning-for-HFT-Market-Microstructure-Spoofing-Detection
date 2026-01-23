# Data Integration Guide

## Overview

This guide explains how to integrate TEN-GNN with your existing market data infrastructure. We provide adapters for the three most common HFT data formats:

1. **FIX Protocol** - Industry standard for order routing and market data
2. **NASDAQ ITCH** - High-performance binary protocol for Nasdaq order book data
3. **OUCH Protocol** - Order entry and execution

## Quick Start

```python
from data_adapters.market_data_adapters import FIXProtocolAdapter, ITCHProtocolAdapter

# FIX Protocol
fix_adapter = FIXProtocolAdapter()
lob_event = fix_adapter.to_lob_event(fix_message)

# Send to API
response = requests.post('http://localhost:8000/predict', json=lob_event)
```

## 47-Feature Format

All adapters transform market data into a standardized 47-feature vector:

| Features | Description                | Example Values        |
| -------- | -------------------------- | --------------------- |
| 0-9      | Bid Prices (Top 10 levels) | [150.25, 150.24, ...] |
| 10-19    | Ask Prices (Top 10 levels) | [150.26, 150.27, ...] |
| 20-29    | Bid Volumes                | [1000, 800, ...]      |
| 30-39    | Ask Volumes                | [1200, 900, ...]      |
| 40       | Spread                     | 0.01                  |
| 41       | Mid Price                  | 150.255               |
| 42       | Order Imbalance            | 0.15                  |
| 43       | Weighted Mid Price         | 150.254               |
| 44       | Price Range                | 0.05                  |
| 45       | Volume Ratio               | 0.83                  |
| 46       | Volatility Estimate        | 0.002                 |

## FIX Protocol Integration

### Supported Message Types

- **35=W** - Market Data Snapshot/Full Refresh
- **35=X** - Market Data Incremental Refresh

### Example: Process FIX Market Data

```python
from data_adapters.market_data_adapters import FIXProtocolAdapter
import requests

adapter = FIXProtocolAdapter()

# FIX message (pipe-delimited)
fix_message = "8=FIX.4.2|9=178|35=W|49=SENDER|56=TARGET|55=AAPL|268=2|269=0|270=150.25|271=1000|269=1|270=150.30|271=800"

# Convert to LOB event
lob_event = adapter.to_lob_event(fix_message)

# Send to TEN-GNN API
response = requests.post(
    'http://localhost:8000/predict',
    json=lob_event
)

if response.json()['alert']:
    print(f"🚨 Spoofing detected! Confidence: {response.json()['confidence']:.2f}")
```

### Integration with QuickFIX

```python
import quickfix as fix
from data_adapters.market_data_adapters import FIXProtocolAdapter

class MarketDataApplication(fix.Application):
    def __init__(self):
        super().__init__()
        self.adapter = FIXProtocolAdapter()

    def fromApp(self, message, sessionID):
        msgType = fix.MsgType()
        message.getHeader().getField(msgType)

        if msgType.getValue() == fix.MsgType_MarketDataSnapshotFullRefresh:
            # Convert to string
            fix_string = message.toString()

            # Process with adapter
            lob_event = self.adapter.to_lob_event(fix_string)

            # Send to API (async recommended)
            self.send_to_api(lob_event)
```

## NASDAQ ITCH Integration

### Supported Message Types

- **A** - Add Order
- **D** - Delete Order
- **U** - Replace Order
- **E** - Execute Order
- **P** - Trade (non-cross)

### Example: Process ITCH Messages

```python
from data_adapters.market_data_adapters import ITCHProtocolAdapter
import struct

adapter = ITCHProtocolAdapter()

# Parse binary ITCH message
def parse_itch_add_order(data):
    return {
        'type': 'A',
        'timestamp': struct.unpack('>Q', data[5:13])[0],
        'order_id': struct.unpack('>Q', data[11:19])[0],
        'side': chr(data[19]),
        'shares': struct.unpack('>I', data[20:24])[0],
        'stock': data[24:32].decode('utf-8').strip(),
        'price': struct.unpack('>I', data[32:36])[0]
    }

# Process message
itch_data = b'...'  # Binary ITCH data
message = parse_itch_add_order(itch_data)

# Convert to LOB event
lob_event = adapter.to_lob_event(message, asset='AAPL')

# Send to API
response = requests.post('http://localhost:8000/predict', json=lob_event)
```

### ITCH Stream Processing

```python
import asyncio
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

adapter = ITCHProtocolAdapter()

async def process_itch_stream(itch_source):
    """Process ITCH stream and send to Kafka"""
    async for message in itch_source:
        lob_event = adapter.to_lob_event(message, asset=message.get('stock'))

        # Send to Kafka for streaming inference
        producer.send('lob_events', value=lob_event)
```

## OUCH Protocol Integration

OUCH tracks your firm's own orders. Useful for monitoring your order activity in context of spoofing detection.

### Example: Monitor Own Orders

```python
from data_adapters.market_data_adapters import OUCHProtocolAdapter

adapter = OUCHProtocolAdapter()

# Track order entry
enter_order = {
    'order_token': 'ABC123',
    'side': 'B',
    'shares': 1000,
    'stock': 'AAPL',
    'price': 1502500,
    'time_in_force': 0
}
adapter.process_enter_order(enter_order)

# Get activity summary
activity = adapter.get_own_order_activity()
print(f"Active orders: {activity['active_orders']}")
print(f"Order imbalance: {activity['order_imbalance']:.2f}")
```

## Kafka Streaming Integration

### Architecture

```
ITCH/FIX Source → Adapter → Kafka → Consumer → TEN-GNN API → Alerts
```

### Producer Example

```python
from kafka import KafkaProducer
from data_adapters.market_data_adapters import FIXProtocolAdapter
import json

producer = KafkaProducer(
    bootstrap_servers='localhost:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

adapter = FIXProtocolAdapter()

# Process incoming FIX messages
for fix_message in fix_message_stream:
    lob_event = adapter.to_lob_event(fix_message)
    producer.send('lob_events', value=lob_event)
```

### Consumer (Automatic)

The streaming consumer (`deployment/streaming/kafka_consumer.py`) automatically:

1. Consumes events from Kafka
2. Sends to TEN-GNN API
3. Generates alerts
4. Stores in PostgreSQL

Start with:

```bash
docker-compose --profile streaming up -d
```

## Performance Optimization

### Batching

Process multiple events in batches for higher throughput:

```python
events = []
for message in message_stream:
    lob_event = adapter.to_lob_event(message)
    events.append(lob_event)

    if len(events) >= 32:  # Batch size
        response = requests.post(
            'http://localhost:8000/predict/batch',
            json={'events': events}
        )
        events = []
```

### Async Processing

```python
import aiohttp
import asyncio

async def process_event_async(session, lob_event):
    async with session.post('http://localhost:8000/predict', json=lob_event) as resp:
        return await resp.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = []
        for event in events:
            task = asyncio.create_task(process_event_async(session, event))
            tasks.append(task)

        results = await asyncio.gather(*tasks)
```

## Testing

### Test Data Generator

```python
# Generate test FIX messages
def generate_test_fix_messages(num_messages=1000):
    for i in range(num_messages):
        yield f"8=FIX.4.2|35=W|55=AAPL|268=2|269=0|270={150 + i*0.01:.2f}|271=1000"

# Send to API
for message in generate_test_fix_messages():
    lob_event = adapter.to_lob_event(message)
    response = requests.post('http://localhost:8000/predict', json=lob_event)
```

### Performance Testing

```bash
# Run benchmark
python benchmarks/performance_benchmark.py

# Test with real data
python -m data_adapters.market_data_adapters
```

## Troubleshooting

### Issue: Missing Features

**Problem:** Some features are 0 or NaN

**Solution:** Ensure all 10 price levels are provided. If fewer levels available:

```python
# Pad with last valid price
bid_prices = [150.25, 150.24, 150.23]  # Only 3 levels
bid_prices.extend([bid_prices[-1]] * 7)  # Pad to 10
```

### Issue: Timestamp Mismatch

**Problem:** Time deltas are incorrect

**Solution:** Use consistent timestamp units (milliseconds):

```python
timestamp_ms = time.time() * 1000  # Current time in ms
```

## Custom Adapter Development

### Template

```python
class CustomProtocolAdapter:
    def __init__(self):
        self.transformer = LOBFeatureTransformer()

    def parse_message(self, message):
        # Parse your custom format
        pass

    def to_lob_event(self, message):
        # Extract data
        # Update transformer
        # Return standardized format
        return {
            'timestamp': ...,
            'asset': ...,
            'bid_prices': [...],
            'ask_prices': [...],
            'bid_volumes': [...],
            'ask_volumes': [...],
            'mid_price': ...,
            'spread': ...,
            'order_imbalance': ...
        }
```

## Support

- Example adapters: `data_adapters/market_data_adapters.py`
- API documentation: `docs/PRODUCTION_DEPLOYMENT.md`
- Performance benchmarks: `benchmarks/performance_benchmark.py`
