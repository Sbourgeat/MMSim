import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import warnings


class MarketPlotter:
    """Specialized plotting for market data visualization"""

    def __init__(self, theme: str = "plotly_dark"):
        self.theme = theme

    def create_candlestick_chart(
        self, data: pd.DataFrame, title: str = "Price Chart", show_volume: bool = True
    ) -> go.Figure:
        """Create interactive candlestick chart with volume"""

        if show_volume:
            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.03,
                subplot_titles=(title, "Volume"),
                row_width=[0.2, 0.7],
            )

            # Candlestick chart
            candlestick = go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="OHLC",
                increasing_line_color="#00ff00",
                decreasing_line_color="red",
            )
            fig.add_trace(candlestick, row=1, col=1)

            # Volume bars
            volume_colors = np.where(data["Close"] >= data["Open"], "#00ff00", "red")
            volume_bars = go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                marker_color=volume_colors,
                opacity=0.6,
            )
            fig.add_trace(volume_bars, row=2, col=1)

            # Update layout
            fig.update_layout(
                template=self.theme,
                title=title,
                xaxis_rangeslider_visible=False,
                yaxis=dict(title="Price ($)"),
                yaxis2=dict(title="Volume"),
            )

        else:
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=data.index,
                        open=data["Open"],
                        high=data["High"],
                        low=data["Low"],
                        close=data["Close"],
                        name="OHLC",
                    )
                ]
            )

            fig.update_layout(
                template=self.theme,
                title=title,
                xaxis_rangeslider_visible=False,
                yaxis=dict(title="Price ($)"),
            )

        return fig

    def create_volume_profile(self, data: pd.DataFrame, bins: int = 50) -> go.Figure:
        """Create volume profile chart showing volume at different price levels"""

        # Calculate volume at each price level
        price_levels = np.linspace(data["Low"].min(), data["High"].max(), bins)
        volume_profile = np.zeros(bins)

        for i in range(len(price_levels) - 1):
            mask = (data["Low"] <= price_levels[i + 1]) & (
                data["High"] >= price_levels[i]
            )
            volume_profile[i] = data.loc[mask, "Volume"].sum()

        fig = go.Figure()

        # Create volume profile bars
        fig.add_trace(
            go.Bar(
                x=volume_profile,
                y=price_levels[:-1],
                orientation="h",
                name="Volume Profile",
                marker_color="lightblue",
                opacity=0.7,
            )
        )

        fig.update_layout(
            template=self.theme,
            title="Volume Profile",
            xaxis=dict(title="Volume"),
            yaxis=dict(title="Price ($)", autorange="reversed"),
        )

        return fig

    def create_order_flow_chart(
        self, data: pd.DataFrame, window: int = 20
    ) -> go.Figure:
        """Create order flow analysis chart"""

        fig = make_subplots(
            rows=3,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Price", "Volume", "Order Flow Imbalance"),
        )

        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="white"),
            ),
            row=1,
            col=1,
        )

        # Volume
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                marker_color="lightblue",
                opacity=0.6,
            ),
            row=2,
            col=1,
        )

        # Order flow imbalance (simplified)
        price_change = data["Close"].diff().fillna(0)
        volume_weighted_price_change = price_change * data["Volume"]
        order_flow_imbalance = volume_weighted_price_change.rolling(window).sum()

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=order_flow_imbalance,
                mode="lines",
                name="Order Flow Imbalance",
                line=dict(color="orange"),
            ),
            row=3,
            col=1,
        )

        # Add zero line
        fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

        fig.update_layout(template=self.theme, title="Order Flow Analysis", height=800)

        return fig

    def create_manipulation_detection_plot(
        self, data: pd.DataFrame, alerts: List[Dict[str, Any]]
    ) -> go.Figure:
        """Create chart with manipulation detection alerts"""

        fig = go.Figure()

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
                increasing_line_color="#00ff00",
                decreasing_line_color="red",
            )
        )

        # Add manipulation alerts
        for alert in alerts:
            if "timestamp" in alert and "price" in alert:
                fig.add_trace(
                    go.Scatter(
                        x=[alert["timestamp"]],
                        y=[alert["price"]],
                        mode="markers",
                        marker=dict(symbol="triangle-up", size=15, color="red"),
                        name=f"Alert: {alert.get('type', 'Unknown')}",
                        text=[alert.get("description", "")],
                        hovertemplate="%{text}<extra></extra>",
                    )
                )

        fig.update_layout(
            template=self.theme,
            title="Market Manipulation Detection",
            xaxis_rangeslider_visible=False,
            yaxis=dict(title="Price ($)"),
        )

        return fig

    def create_real_time_chart(self) -> go.Figure:
        """Create template for real-time chart updates"""

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Real-time Price", "Real-time Volume"),
            row_heights=[0.7, 0.3],
        )

        # Initialize empty traces
        fig.add_trace(
            go.Scatter(
                x=[], y=[], mode="lines", name="Price", line=dict(color="white")
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(x=[], y=[], name="Volume", marker_color="lightblue"), row=2, col=1
        )

        fig.update_layout(
            template=self.theme,
            title="Real-time Market Data",
            xaxis_rangeslider_visible=False,
            yaxis=dict(title="Price ($)"),
            yaxis2=dict(title="Volume"),
        )

        return fig

    def update_real_time_chart(
        self, fig: go.Figure, new_price: float, new_volume: int, timestamp: str
    ):
        """Update real-time chart with new data"""

        # Update price trace
        price_trace = fig.select_traces(row=1, col=1)[0]
        price_trace.x = tuple(list(price_trace.x) + [timestamp])
        price_trace.y = tuple(list(price_trace.y) + [new_price])

        # Update volume trace
        volume_trace = fig.select_traces(row=2, col=1)[0]
        volume_trace.x = tuple(list(volume_trace.x) + [timestamp])
        volume_trace.y = tuple(list(volume_trace.y) + [new_volume])

        return fig

    def create_volatility_chart(
        self, data: pd.DataFrame, window: int = 20
    ) -> go.Figure:
        """Create volatility analysis chart"""

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=("Price", "Volatility"),
        )

        # Price chart
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["Close"],
                mode="lines",
                name="Close Price",
                line=dict(color="white"),
            ),
            row=1,
            col=1,
        )

        # Calculate volatility
        returns = data["Close"].pct_change().fillna(0)
        volatility = returns.rolling(window).std() * np.sqrt(252)  # Annualized

        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=volatility,
                mode="lines",
                name="Volatility",
                line=dict(color="orange"),
            ),
            row=2,
            col=1,
        )

        fig.update_layout(
            template=self.theme,
            title="Volatility Analysis",
            yaxis=dict(title="Price ($)"),
            yaxis2=dict(title="Volatility (%)"),
        )

        return fig


# Utility functions for market plotting
def create_manipulation_alerts_demo(data: pd.DataFrame) -> List[Dict[str, Any]]:
    """Create sample manipulation alerts for demonstration"""
    alerts = []

    # Simulate some manipulation points
    alert_indices = np.random.choice(len(data), size=3, replace=False)

    for idx in alert_indices:
        alerts.append(
            {
                "timestamp": data.index[idx],
                "price": data["Close"].iloc[idx],
                "type": "Volume Spike",
                "description": "Unusual volume detected",
            }
        )

    return alerts


def export_chart_to_html(fig: go.Figure, filename: str, auto_open: bool = False):
    """Export chart to HTML file"""
    fig.write_html(filename, include_plotlyjs="cdn", auto_open=auto_open)


def export_chart_to_image(fig: go.Figure, filename: str, format: str = "png"):
    """Export chart to image file (requires kaleido)"""
    try:
        fig.write_image(filename, format=format)
    except Exception as e:
        warnings.warn(f"Failed to export image: {e}. Install kaleido for image export.")


if __name__ == "__main__":
    # Test market plotting functions

    # Generate sample data
    from simulation import create_sample_market_data

    sample_data = create_sample_market_data(n_days=50)

    # Create plots
    plotter = MarketPlotter()

    # Candlestick chart
    candlestick_fig = plotter.create_candlestick_chart(sample_data)
    candlestick_fig.show()

    # Volume profile
    volume_profile_fig = plotter.create_volume_profile(sample_data)
    volume_profile_fig.show()

    # Order flow
    order_flow_fig = plotter.create_order_flow_chart(sample_data)
    order_flow_fig.show()

    # Manipulation detection demo
    alerts = create_manipulation_alerts_demo(sample_data)
    manipulation_fig = plotter.create_manipulation_detection_plot(sample_data, alerts)
    manipulation_fig.show()

    print("Market plotting tests completed successfully!")
