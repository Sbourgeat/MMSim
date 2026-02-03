"""Simplified dynamic plotting module for market simulation and TDA visualization"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Tuple, Any, Callable
import time
import threading
from queue import Queue, Empty
from dataclasses import dataclass
import warnings


@dataclass
class PlotConfig:
    """Configuration for dynamic plotting"""

    width: int = 1200
    height: int = 800
    update_interval: int = 100
    max_points: int = 1000
    theme: str = "plotly_dark"


class SimpleDynamicPlotter:
    """Simplified dynamic plotting system"""

    def __init__(self, config: Optional[PlotConfig] = None):
        self.config = config or PlotConfig()
        self.fig = None
        self.data_queue = Queue()
        self.is_running = False

    def create_market_plot(self, data: pd.DataFrame) -> go.Figure:
        """Create market price and volume plot"""

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=("Price", "Volume"),
            row_heights=[0.7, 0.3],
        )

        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        # Volume bars
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

        fig.update_layout(
            template=self.config.theme,
            title="Market Data",
            height=self.config.height,
            xaxis_rangeslider_visible=False,
        )

        return fig

    def create_point_cloud_plot(self, point_cloud: np.ndarray) -> go.Figure:
        """Create 3D point cloud visualization"""

        if point_cloud.shape[1] < 3:
            # 2D plot if fewer than 3 dimensions
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=point_cloud[:, 0],
                    y=point_cloud[:, 1],
                    mode="markers",
                    marker=dict(size=5, color="cyan"),
                    name="Points",
                )
            )
            fig.update_layout(
                template=self.config.theme,
                title="Point Cloud",
                xaxis_title="Dim 1",
                yaxis_title="Dim 2",
            )
        else:
            # 3D plot
            fig = go.Figure()
            fig.add_trace(
                go.Scatter3d(
                    x=point_cloud[:, 0],
                    y=point_cloud[:, 1],
                    z=point_cloud[:, 2],
                    mode="markers",
                    marker=dict(size=3, color="lightblue"),
                    name="Points",
                )
            )
            fig.update_layout(
                template=self.config.theme,
                title="Point Cloud",
                scene=dict(
                    xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"
                ),
            )

        return fig

    def create_persistence_diagram(self, persistence_data: List) -> go.Figure:
        """Create persistence diagram"""

        fig = go.Figure()

        # Plot different homology dimensions
        colors = ["red", "green", "blue"]
        labels = ["H0", "H1", "H2"]

        for i, (color, label) in enumerate(zip(colors, labels)):
            if i < len(persistence_data) and len(persistence_data[i]) > 0:
                points = persistence_data[i]
                fig.add_trace(
                    go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode="markers",
                        marker=dict(color=color, size=6),
                        name=label,
                    )
                )

        # Add diagonal line
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color="gray", dash="dash"),
                name="Diagonal",
            )
        )

        fig.update_layout(
            template=self.config.theme,
            title="Persistence Diagram",
            xaxis_title="Birth",
            yaxis_title="Death",
        )

        return fig

    def create_entropy_plot(self, entropy_values: List[float]) -> go.Figure:
        """Create entropy evolution plot"""

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(len(entropy_values))),
                y=entropy_values,
                mode="lines+markers",
                name="Entropy",
                line=dict(color="orange", width=2),
            )
        )

        fig.update_layout(
            template=self.config.theme,
            title="Persistence Entropy",
            xaxis_title="Time Step",
            yaxis_title="Entropy",
        )

        return fig

    def create_comprehensive_view(
        self,
        market_data: pd.DataFrame,
        point_cloud: np.ndarray,
        persistence_data: List,
        entropy_values: List[float],
    ) -> go.Figure:
        """Create comprehensive view with all components"""

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Market Data",
                "Point Cloud",
                "Persistence Diagram",
                "Persistence Entropy",
            ),
            specs=[
                [{"secondary_y": True}, {"type": "scatter3d"}],
                [{"type": "scatter"}, {"type": "scatter"}],
            ],
        )

        # Market data
        fig.add_trace(
            go.Candlestick(
                x=market_data.index,
                open=market_data["Open"],
                high=market_data["High"],
                low=market_data["Low"],
                close=market_data["Close"],
                name="Price",
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Bar(
                x=market_data.index,
                y=market_data["Volume"],
                name="Volume",
                marker_color="rgba(158,158,158,0.5)",
            ),
            row=1,
            col=1,
            secondary_y=True,
        )

        # Point cloud
        if point_cloud.shape[1] >= 3:
            fig.add_trace(
                go.Scatter3d(
                    x=point_cloud[:, 0],
                    y=point_cloud[:, 1],
                    z=point_cloud[:, 2],
                    mode="markers",
                    marker=dict(size=3, color="lightblue"),
                    name="Points",
                ),
                row=1,
                col=2,
            )

        # Persistence diagram
        colors = ["red", "green", "blue"]
        labels = ["H0", "H1", "H2"]

        for i, (color, label) in enumerate(zip(colors, labels)):
            if i < len(persistence_data) and len(persistence_data[i]) > 0:
                points = persistence_data[i]
                fig.add_trace(
                    go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode="markers",
                        marker=dict(color=color, size=4),
                        name=label,
                    ),
                    row=2,
                    col=1,
                )

        # Entropy
        fig.add_trace(
            go.Scatter(
                x=list(range(len(entropy_values))),
                y=entropy_values,
                mode="lines+markers",
                name="Entropy",
                line=dict(color="orange", width=2),
            ),
            row=2,
            col=2,
        )

        fig.update_layout(
            template=self.config.theme,
            title="Market Manipulation Detection with TDA",
            height=self.config.height,
            showlegend=True,
        )

        return fig

    def add_manipulation_alert(self, fig: go.Figure, timestamp: str, alert_type: str):
        """Add manipulation alert to plot"""

        # Add vertical line for alert
        fig.add_vline(
            x=timestamp,
            line_width=2,
            line_dash="dot",
            line_color="red",
            annotation_text=alert_type,
        )


class SimpleMarketPlotter:
    """Simplified market plotting utilities"""

    def __init__(self, theme: str = "plotly_dark"):
        self.theme = theme

    def create_candlestick(self, data: pd.DataFrame) -> go.Figure:
        """Create candlestick chart"""

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=data.index,
                    open=data["Open"],
                    high=data["High"],
                    low=data["Low"],
                    close=data["Close"],
                    name="Price",
                )
            ]
        )

        fig.update_layout(
            template=self.theme, title="Price Chart", xaxis_rangeslider_visible=False
        )

        return fig

    def create_volume_chart(self, data: pd.DataFrame) -> go.Figure:
        """Create volume chart"""

        fig = go.Figure(
            data=[
                go.Bar(
                    x=data.index,
                    y=data["Volume"],
                    name="Volume",
                    marker_color="lightblue",
                )
            ]
        )

        fig.update_layout(template=self.theme, title="Volume", yaxis_title="Volume")

        return fig


class SimpleTDAPlotter:
    """Simplified TDA plotting utilities"""

    def __init__(self, theme: str = "plotly_dark"):
        self.theme = theme

    def create_point_cloud_2d(self, point_cloud: np.ndarray) -> go.Figure:
        """Create 2D point cloud"""

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=point_cloud[:, 0],
                y=point_cloud[:, 1],
                mode="markers",
                marker=dict(size=5, color="cyan"),
                name="Points",
            )
        )

        fig.update_layout(
            template=self.theme,
            title="2D Point Cloud",
            xaxis_title="Dim 1",
            yaxis_title="Dim 2",
        )

        return fig

    def create_entropy_evolution(self, entropy_data: List[List[float]]) -> go.Figure:
        """Create entropy evolution for all dimensions"""

        fig = go.Figure()

        colors = ["red", "green", "blue"]
        labels = ["H0 Entropy", "H1 Entropy", "H2 Entropy"]

        for i, (color, label) in enumerate(zip(colors, labels)):
            if i < len(entropy_data) and len(entropy_data[i]) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(entropy_data[i]))),
                        y=entropy_data[i],
                        mode="lines",
                        name=label,
                        line=dict(color=color, width=2),
                    )
                )

        fig.update_layout(
            template=self.theme,
            title="Persistence Entropy Evolution",
            xaxis_title="Time Step",
            yaxis_title="Entropy",
        )

        return fig
