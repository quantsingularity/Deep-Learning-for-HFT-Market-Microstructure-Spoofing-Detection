# LOB Data Format Specification

This directory should contain Limit Order Book (LOB) data for training and evaluation.

## Expected Data Format

### CSV Format

```csv
timestamp,best_bid,best_ask,bid_volume_1,ask_volume_1,bid_volume_2,ask_volume_2,...
1609459200000,100.50,100.51,1000,800,1500,1200,...
1609459200005,100.50,100.51,1100,850,1450,1150,...
```

### Columns

- `timestamp`: Unix timestamp in milliseconds
- `best_bid`: Best bid price
- `best_ask`: Best ask price
- `bid_volume_N`: Volume at Nth bid level (N=1 to 10)
- `ask_volume_N`: Volume at Nth ask level (N=1 to 10)
- `bid_price_N`: Nth bid price (optional, can be inferred)
- `ask_price_N`: Nth ask price (optional, can be inferred)

### Order Events

Additional order event data can be provided in a separate CSV:

```csv
timestamp,event_type,order_id,price,volume,side
1609459200000,placement,ORD001,100.50,1000,1
1609459200010,cancellation,ORD001,100.50,1000,1
```

Where:

- `event_type`: 'placement', 'cancellation', 'execution', 'modification'
- `side`: 1 for buy, -1 for sell

## Data Sources

1. **LOBSTER Database**: High-quality reconstructed LOB data
   - Website: https://lobsterdata.com
   - Provides Level 3 data for NASDAQ stocks

2. **CME Group**: Futures market data
   - Historical Market Data: https://www.cmegroup.com/market-data/datamine-historical-data.html
   - E-mini S&P 500 and other futures contracts

3. **Cryptocurrency Exchanges**: Via APIs
   - Binance: https://github.com/binance/binance-spot-api-docs
   - Coinbase Pro: https://docs.pro.coinbase.com

## Synthetic Data

If real data is not available, use the Adversarial Backtest Framework:

```python
from utils.data_generation import AdversarialBacktestFramework
import pandas as pd

# Generate baseline
lob_data = pd.DataFrame({
    'timestamp': np.arange(10000),
    'best_bid': 100 + np.cumsum(np.random.randn(10000) * 0.01),
    'best_ask': 100.05 + np.cumsum(np.random.randn(10000) * 0.01)
})

# Inject spoofing patterns
framework = AdversarialBacktestFramework()
sequences, labels, metadata = framework.generate_labeled_dataset(
    lob_data,
    num_samples=1000,
    spoofing_ratio=0.5
)
```

## Data Preprocessing

1. **Normalization**: Z-score normalization per feature
2. **Windowing**: Fixed-length sequences (default: 100 events)
3. **Time Encoding**: Compute time-since-last-event
4. **Feature Extraction**: 47-dimensional feature vectors

See `utils/feature_engineering.py` for implementation.

## Privacy and Compliance

- Ensure data usage complies with exchange terms of service
- Anonymize trader IDs in real data
- Follow MiFID II and MAR regulations for EU data
- Obtain necessary licenses for commercial use
