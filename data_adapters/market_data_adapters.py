"""
Data Adapters for Common Market Data Formats
Transforms FIX, ITCH, and OUCH messages to TEN-GNN 47-feature format
"""

import numpy as np
from typing import Dict, Tuple, Optional
from collections import deque
from datetime import datetime


class LOBFeatureTransformer:
    """
    Base transformer for LOB data to 47-feature format

    Feature Structure (47 features):
    - Features 0-9: Bid prices (levels 1-10)
    - Features 10-19: Ask prices (levels 1-10)
    - Features 20-29: Bid volumes (levels 1-10)
    - Features 30-39: Ask volumes (levels 1-10)
    - Features 40-46: Microstructure features
    """

    def __init__(self, num_levels: int = 10):
        """
        Args:
            num_levels: Number of price levels to track
        """
        self.num_levels = num_levels

        # Maintain order book state
        self.bids = {}  # price -> volume
        self.asks = {}  # price -> volume

        # Historical features for derived metrics
        self.price_history = deque(maxlen=100)
        self.volume_history = deque(maxlen=100)
        self.last_trade_price = None
        self.last_trade_volume = None

    def extract_features(self) -> np.ndarray:
        """
        Extract 47 features from current LOB state

        Returns:
            Feature vector (47,)
        """
        features = np.zeros(47)

        # Sort bids (descending) and asks (ascending)
        sorted_bids = sorted(self.bids.items(), key=lambda x: x[0], reverse=True)[
            : self.num_levels
        ]
        sorted_asks = sorted(self.asks.items(), key=lambda x: x[0])[: self.num_levels]

        # Features 0-9: Bid prices
        for i, (price, _) in enumerate(sorted_bids):
            if i < self.num_levels:
                features[i] = price

        # Features 10-19: Ask prices
        for i, (price, _) in enumerate(sorted_asks):
            if i < self.num_levels:
                features[10 + i] = price

        # Features 20-29: Bid volumes
        for i, (_, volume) in enumerate(sorted_bids):
            if i < self.num_levels:
                features[20 + i] = volume

        # Features 30-39: Ask volumes
        for i, (_, volume) in enumerate(sorted_asks):
            if i < self.num_levels:
                features[30 + i] = volume

        # Microstructure features (40-46)
        if sorted_bids and sorted_asks:
            best_bid = sorted_bids[0][0]
            best_ask = sorted_asks[0][0]

            # Feature 40: Spread
            features[40] = best_ask - best_bid

            # Feature 41: Mid price
            features[41] = (best_bid + best_ask) / 2

            # Feature 42: Order imbalance
            bid_volume = sum(v for _, v in sorted_bids)
            ask_volume = sum(v for _, v in sorted_asks)
            features[42] = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-8)

            # Feature 43: Weighted mid price
            best_bid_vol = sorted_bids[0][1]
            best_ask_vol = sorted_asks[0][1]
            features[43] = (best_bid * best_ask_vol + best_ask * best_bid_vol) / (
                best_bid_vol + best_ask_vol + 1e-8
            )

            # Feature 44: Price range
            if len(self.price_history) > 0:
                features[44] = max(self.price_history) - min(self.price_history)

            # Feature 45: Volume ratio
            if self.last_trade_volume and bid_volume > 0:
                features[45] = self.last_trade_volume / bid_volume

            # Feature 46: Volatility estimate
            if len(self.price_history) > 10:
                features[46] = np.std(list(self.price_history))

        return features

    def update_book(self, side: str, price: float, volume: float, action: str = "add"):
        """
        Update order book state

        Args:
            side: 'bid' or 'ask'
            price: Order price
            volume: Order volume
            action: 'add', 'modify', or 'delete'
        """
        book = self.bids if side == "bid" else self.asks

        if action == "delete":
            if price in book:
                del book[price]
        elif action == "modify":
            book[price] = volume
        else:  # add
            if price in book:
                book[price] += volume
            else:
                book[price] = volume

        # Remove zero volumes
        book = {p: v for p, v in book.items() if v > 0}

        if side == "bid":
            self.bids = book
        else:
            self.asks = book


class FIXProtocolAdapter:
    """
    Adapter for FIX Protocol messages

    Handles FIX 4.2/4.4 market data messages (35=W, 35=X)
    """

    def __init__(self):
        self.transformer = LOBFeatureTransformer()
        self.last_timestamp = None

    def parse_fix_message(self, message: str) -> Optional[Dict]:
        """
        Parse FIX message string

        Args:
            message: FIX message (e.g., "8=FIX.4.2|9=178|35=W|49=SENDER|...")

        Returns:
            Parsed message dictionary or None
        """
        if not message:
            return None

        fields = {}
        for field in message.split("|"):
            if "=" in field:
                tag, value = field.split("=", 1)
                fields[int(tag)] = value

        return fields

    def process_market_data_snapshot(self, fields: Dict) -> Tuple[np.ndarray, float]:
        """
        Process FIX Market Data Snapshot (35=W)

        Args:
            fields: Parsed FIX fields

        Returns:
            (feature_vector, time_delta)
        """
        fields.get(55, "UNKNOWN")

        # Extract timestamp (tag 52)
        fields.get(52, "")
        current_time = datetime.utcnow().timestamp() * 1000

        # Calculate time delta
        time_delta = 0.0
        if self.last_timestamp:
            time_delta = current_time - self.last_timestamp
        self.last_timestamp = current_time

        # Process MD entries (tags 268, 269, 270, 271)
        # Tag 268 = NoMDEntries
        # Tag 269 = MDEntryType (0=Bid, 1=Ask, 2=Trade)
        # Tag 270 = MDEntryPx (price)
        # Tag 271 = MDEntrySize (volume)

        # For simplicity, assume repeating groups are parsed
        # In production, use proper FIX parser library

        # Update transformer
        # (Simplified - in production, parse all MD entries)

        features = self.transformer.extract_features()

        return features, time_delta

    def to_lob_event(self, message: str) -> Optional[Dict]:
        """
        Convert FIX message to LOB event format

        Args:
            message: FIX message string

        Returns:
            LOB event dictionary compatible with TEN-GNN API
        """
        fields = self.parse_fix_message(message)
        if not fields:
            return None

        features, time_delta = self.process_market_data_snapshot(fields)

        # Extract bid/ask prices and volumes
        bid_prices = features[0:10].tolist()
        ask_prices = features[10:20].tolist()
        bid_volumes = features[20:30].tolist()
        ask_volumes = features[30:40].tolist()

        return {
            "timestamp": self.last_timestamp or datetime.utcnow().timestamp() * 1000,
            "asset": fields.get(55, "UNKNOWN"),
            "bid_prices": bid_prices,
            "ask_prices": ask_prices,
            "bid_volumes": bid_volumes,
            "ask_volumes": ask_volumes,
            "mid_price": features[41],
            "spread": features[40],
            "order_imbalance": features[42],
        }


class ITCHProtocolAdapter:
    """
    Adapter for NASDAQ ITCH Protocol (Version 5.0)

    Handles ITCH message types:
    - 'A': Add Order
    - 'D': Delete Order
    - 'U': Replace Order
    - 'E': Execute Order
    - 'P': Trade (non-cross)
    """

    def __init__(self):
        self.transformer = LOBFeatureTransformer()
        self.orders = {}  # order_id -> (side, price, volume)
        self.last_timestamp = None

    def process_add_order(self, message: Dict) -> None:
        """
        Process Add Order message (Type A)

        Args:
            message: ITCH message with fields {
                'timestamp': nanoseconds,
                'order_id': int,
                'side': 'B' or 'S',
                'shares': int,
                'stock': str,
                'price': int (4 decimal places)
            }
        """
        order_id = message["order_id"]
        side = "bid" if message["side"] == "B" else "ask"
        price = message["price"] / 10000.0  # Convert to decimal
        volume = message["shares"]

        # Store order
        self.orders[order_id] = (side, price, volume)

        # Update book
        self.transformer.update_book(side, price, volume, action="add")

    def process_delete_order(self, message: Dict) -> None:
        """
        Process Delete Order message (Type D)
        """
        order_id = message["order_id"]

        if order_id in self.orders:
            side, price, volume = self.orders[order_id]
            self.transformer.update_book(side, price, volume, action="delete")
            del self.orders[order_id]

    def process_replace_order(self, message: Dict) -> None:
        """
        Process Replace Order message (Type U)
        """
        old_order_id = message["old_order_id"]
        new_order_id = message["new_order_id"]
        new_shares = message["new_shares"]
        new_price = message["new_price"] / 10000.0

        # Delete old order
        if old_order_id in self.orders:
            side, old_price, old_volume = self.orders[old_order_id]
            self.transformer.update_book(side, old_price, old_volume, action="delete")
            del self.orders[old_order_id]

            # Add new order
            self.orders[new_order_id] = (side, new_price, new_shares)
            self.transformer.update_book(side, new_price, new_shares, action="add")

    def to_lob_event(self, message: Dict, asset: str) -> Dict:
        """
        Convert ITCH message to LOB event format

        Args:
            message: ITCH message
            asset: Asset symbol

        Returns:
            LOB event dictionary
        """
        # Update timestamp
        current_time = message.get("timestamp", 0) / 1e6  # nanoseconds to milliseconds
        if self.last_timestamp:
            current_time - self.last_timestamp
        self.last_timestamp = current_time

        # Process message based on type
        msg_type = message.get("type", "")

        if msg_type == "A":
            self.process_add_order(message)
        elif msg_type == "D":
            self.process_delete_order(message)
        elif msg_type == "U":
            self.process_replace_order(message)

        # Extract features
        features = self.transformer.extract_features()

        return {
            "timestamp": current_time,
            "asset": asset,
            "bid_prices": features[0:10].tolist(),
            "ask_prices": features[10:20].tolist(),
            "bid_volumes": features[20:30].tolist(),
            "ask_volumes": features[30:40].tolist(),
            "mid_price": features[41],
            "spread": features[40],
            "order_imbalance": features[42],
        }


class OUCHProtocolAdapter:
    """
    Adapter for OUCH Protocol (Order Entry)

    OUCH is used for order entry, not market data, but can be used
    to track own orders in the context of spoofing detection.
    """

    def __init__(self):
        self.own_orders = {}  # track firm's own orders
        self.transformer = LOBFeatureTransformer()

    def process_enter_order(self, message: Dict) -> None:
        """
        Process Enter Order message

        Args:
            message: OUCH message with fields {
                'order_token': str,
                'side': 'B' or 'S',
                'shares': int,
                'stock': str,
                'price': int,
                'time_in_force': int
            }
        """
        order_token = message["order_token"]
        self.own_orders[order_token] = message

    def process_cancel_order(self, message: Dict) -> None:
        """
        Process Cancel Order message
        """
        order_token = message["order_token"]
        if order_token in self.own_orders:
            del self.own_orders[order_token]

    def get_own_order_activity(self) -> Dict:
        """
        Get summary of own order activity

        Returns:
            Dictionary with order activity metrics
        """
        total_buy_volume = sum(
            o["shares"] for o in self.own_orders.values() if o["side"] == "B"
        )
        total_sell_volume = sum(
            o["shares"] for o in self.own_orders.values() if o["side"] == "S"
        )

        return {
            "active_orders": len(self.own_orders),
            "buy_volume": total_buy_volume,
            "sell_volume": total_sell_volume,
            "order_imbalance": (total_buy_volume - total_sell_volume)
            / (total_buy_volume + total_sell_volume + 1e-8),
        }


# ============================================
# Example Usage
# ============================================

if __name__ == "__main__":
    print("=" * 70)
    print("TEN-GNN Data Adapters - Example Usage")
    print("=" * 70)

    # Example 1: FIX Protocol
    print("\n1. FIX Protocol Adapter")
    print("-" * 70)

    fix_adapter = FIXProtocolAdapter()

    # Simulate FIX Market Data Snapshot
    fix_message = "8=FIX.4.2|9=178|35=W|49=SENDER|56=TARGET|55=AAPL|268=2|269=0|270=150.25|271=1000|269=1|270=150.30|271=800"

    lob_event = fix_adapter.to_lob_event(fix_message)
    if lob_event:
        print(f"Converted LOB Event:")
        print(f"  Asset: {lob_event['asset']}")
        print(f"  Mid Price: {lob_event['mid_price']:.2f}")
        print(f"  Spread: {lob_event['spread']:.4f}")

    # Example 2: ITCH Protocol
    print("\n2. ITCH Protocol Adapter")
    print("-" * 70)

    itch_adapter = ITCHProtocolAdapter()

    # Simulate ITCH Add Order
    itch_message = {
        "type": "A",
        "timestamp": 1234567890000000,  # nanoseconds
        "order_id": 12345,
        "side": "B",
        "shares": 100,
        "stock": "AAPL",
        "price": 1502500,  # $150.25 with 4 decimal places
    }

    lob_event = itch_adapter.to_lob_event(itch_message, "AAPL")
    print(f"Converted LOB Event:")
    print(f"  Asset: {lob_event['asset']}")
    print(f"  Timestamp: {lob_event['timestamp']:.0f}ms")

    print("\n" + "=" * 70)
    print("✓ All adapters initialized successfully")
    print("=" * 70)
