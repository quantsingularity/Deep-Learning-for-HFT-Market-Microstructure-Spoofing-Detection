-- PostgreSQL initialization script for alert storage

CREATE TABLE IF NOT EXISTS alerts (
    id SERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    asset VARCHAR(20) NOT NULL,
    prediction INTEGER NOT NULL,
    confidence FLOAT NOT NULL,
    inference_time_ms FLOAT NOT NULL,
    mid_price FLOAT,
    spread FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_alerts_timestamp ON alerts(timestamp);
CREATE INDEX idx_alerts_asset ON alerts(asset);
CREATE INDEX idx_alerts_created_at ON alerts(created_at);

-- Performance metrics table
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    timestamp BIGINT NOT NULL,
    mean_latency_ms FLOAT NOT NULL,
    p95_latency_ms FLOAT NOT NULL,
    throughput FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User settings
CREATE TABLE IF NOT EXISTS settings (
    key VARCHAR(100) PRIMARY KEY,
    value TEXT NOT NULL,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default settings
INSERT INTO settings (key, value) VALUES
    ('confidence_threshold', '0.8'),
    ('alert_cooldown_ms', '1000'),
    ('max_alerts_per_hour', '100')
ON CONFLICT (key) DO NOTHING;
