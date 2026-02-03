# MMSim: Market Manipulation Simulation in Python

**Note: This project is currently in development**

## Features

- [ ] Market manipulation simulation
- [ ] Predicting MM with persistence entropy

A Python based market manipulation simulation strongly inspired by [braintruffle's video](https://www.youtube.com/watch?v=-jF9gW2r_bk)

## Installation

```bash
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

## Usage

The main application demonstrates our topological data analysis pipeline for market manipulation detection:

```python
# Main workflow from src/main.py
def main():
    # 1. Import historical price data
    data = import_price_historical("AAPL")
    
    # 2. Extract closing prices and embed in higher dimensional space
    ts = data["Close"].values
    ts_embedded = persistence_homology.embed_time_series(ts)
    
    # 3. Reduce dimension if needed for visualization
    if ts_embedded.shape[1] > 3:
        ts_embedded = persistence_homology.reduce_dimension(ts_embedded)
    
    # 4. Visualize the embedded time series
    persistence_homology.plot_pcd(ts_embedded, "embedded_ts_AAPL")
    
    # 5. Compute persistent homology using Vietoris-Rips filtration
    persistence_diagram = persistence_homology.vietoris_rips_transform(ts_embedded, symbol="AAPL")
    
    # 6. Calculate persistence entropy to detect manipulation patterns
    persistence_entropy = persistence_homology.persistence_entropy(persistence_diagram)
```

The goal of TDA here is to transform time series data into a topological space where manipulation patterns become more apparent, then use persistence entropy to quantify these patterns for detection.

```bash
# Run application
uv run python src/main.py

# Run tests
uv run python -m pytest
```

## Topological Data Analysis

This project uses topological data analysis (TDA) to identify market manipulation patterns. TDA helps us understand the "shape" of market data by:

- **Persistent Homology**: Detecting persistent features in price movements that may indicate manipulation
- **Persistence Entropy**: Measuring complexity of topological features to predict market manipulation

By analyzing the topological structure of financial time series, we can spot anomalies and manipulation tactics that traditional statistical methods might miss, such as coordinated buying patterns or artificial price ceilings.
