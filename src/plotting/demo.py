"""Example usage of dynamic plotting module"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from simulation import create_sample_market_data, MarketSimulator
from plotting import SimpleDynamicPlotter, MarketPlotter, TDAPlotter
from plotting.dynamic_plotter import SimpleDynamicPlotter


def demo_market_plotting():
    """Demonstrate market plotting functionality"""

    # Create sample market data
    data = create_sample_market_data(n_days=30)

    # Initialize plotter
    market_plotter = MarketPlotter()

    # Create candlestick chart
    candlestick_fig = market_plotter.create_candlestick_chart(
        data, title="Sample Market Data"
    )
    candlestick_fig.show()

    # Create volume profile
    volume_profile_fig = market_plotter.create_volume_profile(data)
    volume_profile_fig.show()

    # Create order flow analysis
    order_flow_fig = market_plotter.create_order_flow_chart(data)
    order_flow_fig.show()

    # Create volatility chart
    volatility_fig = market_plotter.create_volatility_chart(data)
    volatility_fig.show()


def demo_tda_plotting():
    """Demonstrate TDA plotting functionality"""

    # Create sample TDA data
    plotter = TDAPlotter()

    # Generate sample point cloud
    n_points = 100
    t = np.linspace(0, 4 * np.pi, n_points)
    point_cloud = np.column_stack(
        [
            np.cos(t) * np.linspace(0.1, 1, n_points),
            np.sin(t) * np.linspace(0.1, 1, n_points),
            t / (4 * np.pi),
        ]
    )

    # Create persistence diagram (mock)
    persistence_diagram = [
        np.array([[0.1, 0.8], [0.2, 0.6], [0.15, 0.9]]),  # H0
        np.array([[0.3, 0.7], [0.25, 0.8]]),  # H1
        np.array([[0.4, 0.6]]),  # H2
    ]

    # Create entropy values
    entropy_values = [
        np.linspace(0.5, 1.5, 50),  # H0 entropy
        np.linspace(0.3, 1.2, 50),  # H1 entropy
        np.linspace(0.2, 0.8, 50),  # H2 entropy
    ]

    # Create plots
    point_cloud_fig = plotter.create_point_cloud_plot(point_cloud)
    point_cloud_fig.show()

    persistence_fig = plotter.create_persistence_diagram([persistence_diagram])
    persistence_fig.show()

    entropy_fig = plotter.create_entropy_plot([entropy_values])
    entropy_fig.show()

    comprehensive_fig = plotter.create_comprehensive_tda_plot(
        point_cloud, [persistence_diagram], [entropy_values]
    )
    comprehensive_fig.show()


def demo_dynamic_plotting():
    """Demonstrate dynamic plotting functionality"""

    from topology import persistence_homology

    # Create simulator
    simulator = MarketSimulator()

    class MockTDAAnalyzer:
        def embed_time_series(self, ts):
            n = len(ts)
            if n >= 3:
                embedded = np.array([ts[i : i + 3] for i in range(n - 3)])
            else:
                embedded = ts.reshape(-1, 1)
            return embedded

        def reduce_dimension(self, data):
            if data.shape[1] > 3:
                return data[:, :3]
            return data

        def vietoris_rips_transform(self, data, symbol):
            # Mock persistence diagram
            return [np.random.rand(5, 2) for _ in range(3)]

        def persistence_entropy(self, diagram):
            # Mock entropy values
            return [np.random.rand(3)]

    tda_analyzer = MockTDAAnalyzer()

    # Use simplified dynamic plotter
    plotter = SimpleDynamicPlotter()

    # Create comprehensive view with mock data
    market_data = simulator.create_ohlcv_data(n_days=5, n_intraday=10)

    # Generate simple point cloud
    n_points = 50
    point_cloud = np.random.randn(n_points, 3)

    # Mock persistence data
    persistence_data = [
        np.array([[0.1, 0.8], [0.2, 0.6]]),
        np.array([[0.3, 0.7]]),
        np.array([[0.4, 0.6]]),
    ]

    entropy_values = [0.5, 0.3, 0.2]

    # Create comprehensive plot
    fig = plotter.create_comprehensive_view(
        market_data, point_cloud, persistence_data, entropy_values
    )
    fig.show()

    return plotter


if __name__ == "__main__":
    print("Starting plotting demonstrations...")

    print("\n1. Market Plotting Demo")
    demo_market_plotting()

    print("\n2. TDA Plotting Demo")
    demo_tda_plotting()

    print("\n3. Dynamic Plotting Demo")
    try:
        dynamic_plotter = demo_dynamic_plotting()
        print("Dynamic plotter started. Close browser window to continue...")
    except Exception as e:
        print(f"Dynamic plotting demo failed: {e}")

    print("\nPlotting demonstrations completed!")
