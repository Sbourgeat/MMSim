"""Interactive market visualization with sliding window selector"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def create_interactive_price_plot(
    data: pd.DataFrame, title: str = "Market Price Evolution"
) -> go.Figure:
    """
    Create interactive price plot with:
    - Main candlestick chart with volume
    - Sliding window selector below
    - Range slider for zooming
    """

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=(title, "Window Selector"),
    )

    # Main chart - Row 1: Candlestick with volume
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
            increasing_fillcolor="#00ff00",
            decreasing_fillcolor="red",
        ),
        row=1,
        col=1,
    )

    # Add volume bars on secondary y-axis
    colors = [
        "#00ff00" if data["Close"].iloc[i] >= data["Open"].iloc[i] else "red"
        for i in range(len(data))
    ]

    fig.add_trace(
        go.Bar(
            x=data.index,
            y=data["Volume"],
            name="Volume",
            marker_color=colors,
            opacity=0.3,
            yaxis="y2",
        ),
        row=1,
        col=1,
    )

    # Window selector - Row 2: Simple line chart for selecting range
    fig.add_trace(
        go.Scatter(
            x=data.index,
            y=data["Close"],
            mode="lines",
            line=dict(color="blue", width=1),
            name="Price Overview",
            fill="tozeroy",
            fillcolor="rgba(0, 100, 255, 0.2)",
        ),
        row=2,
        col=1,
    )

    # Add moving average to window selector
    if len(data) >= 20:
        ma20 = data["Close"].rolling(window=20).mean()
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=ma20,
                mode="lines",
                line=dict(color="orange", width=2, dash="dash"),
                name="MA20",
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    # Update layout with range slider and buttons
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>Interactive Analysis with Window Selection</sup>",
            x=0.5,
            font=dict(size=18),
        ),
        height=800,
        width=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        # Add range slider for interactivity
        xaxis=dict(
            rangeslider=dict(
                visible=True,
                thickness=0.05,
                bgcolor="rgba(200, 200, 200, 0.2)",
                bordercolor="gray",
                borderwidth=1,
            ),
            rangeselector=dict(
                buttons=list(
                    [
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=3, label="3m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(step="all", label="All"),
                    ]
                ),
                bgcolor="rgba(255, 255, 255, 0.8)",
                bordercolor="gray",
                borderwidth=1,
                x=0.85,
                y=1.02,
            ),
        ),
        # Secondary y-axis for volume
        yaxis=dict(
            title="Price ($)",
            side="left",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.2)",
        ),
        yaxis2=dict(
            title="Volume",
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, data["Volume"].max() * 3],
        ),
        # Update mode for better interactivity
        dragmode="zoom",
        hovermode="x unified",
        # Template
        template="plotly_dark",
        # Margins
        margin=dict(t=100, b=80, l=60, r=60),
    )

    # Update axes
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", row=1, col=1)

    fig.update_xaxes(
        title_text="Date",
        showgrid=True,
        gridcolor="rgba(128, 128, 128, 0.2)",
        rangeslider=dict(thickness=0.15),
        row=2,
        col=1,
    )

    fig.update_yaxes(
        title_text="Price ($)",
        showgrid=True,
        gridcolor="rgba(128, 128, 128, 0.2)",
        row=1,
        col=1,
    )

    fig.update_yaxes(
        title_text="Price",
        showgrid=True,
        gridcolor="rgba(128, 128, 128, 0.2)",
        row=2,
        col=1,
    )

    return fig


def create_multi_scenario_interactive_plot(
    results: Dict[str, Dict], specs: Dict
) -> go.Figure:
    """
    Create interactive plot comparing all scenarios with sliding window
    """
    fig = make_subplots(
        rows=2,
        cols=2,
        shared_xaxes=False,
        vertical_spacing=0.1,
        horizontal_spacing=0.08,
        subplot_titles=("Control Market", "Pump & Dump", "Spoofing", "Layering"),
    )

    colors = {
        "control": "blue",
        "pump_and_dump": "red",
        "spoofing": "green",
        "layering": "orange",
    }

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for idx, (key, pos) in enumerate(
        zip(["control", "pump_and_dump", "spoofing", "layering"], positions)
    ):
        if key not in results:
            continue

        data = results[key]["data"]
        color = colors.get(key, "blue")
        display_name = specs[key].display if key in specs else key

        row, col = pos

        # Add candlestick
        fig.add_trace(
            go.Candlestick(
                x=list(range(len(data))),
                open=data["Open"],
                high=data["High"],
                low=data["Low"],
                close=data["Close"],
                name=f"{display_name} Price",
                increasing_line_color=color,
                decreasing_line_color="red" if key != "control" else "darkred",
            ),
            row=row,
            col=col,
        )

        # Add volume
        volume_colors = [
            color if data["Close"].iloc[i] >= data["Open"].iloc[i] else "darkgray"
            for i in range(len(data))
        ]
        fig.add_trace(
            go.Bar(
                x=list(range(len(data))),
                y=data["Volume"],
                name=f"{display_name} Volume",
                marker_color=volume_colors,
                opacity=0.4,
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        # Add moving average
        if len(data) >= 10:
            ma10 = data["Close"].rolling(window=10).mean()
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(data))),
                    y=ma10,
                    mode="lines",
                    line=dict(color="yellow", width=1.5, dash="dot"),
                    name=f"MA10",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

    fig.update_layout(
        title=dict(
            text="Interactive Multi-Scenario Comparison<br><sup>Use range sliders to explore different time windows</sup>",
            x=0.5,
            font=dict(size=16),
        ),
        height=900,
        width=1400,
        showlegend=False,
        template="plotly_dark",
        dragmode="zoom",
    )

    # Add range sliders to all subplots
    fig.update_xaxes(rangeslider=dict(visible=True, thickness=0.05))

    # Update all y-axes
    for i in range(1, 3):
        for j in range(1, 3):
            fig.update_yaxes(title_text="Price ($)", row=i, col=j)

    return fig


if __name__ == "__main__":
    # Test the interactive plot
    from simulation import create_sample_market_data

    print("Creating interactive price plot...")
    data = create_sample_market_data(n_days=100)
    fig = create_interactive_price_plot(data, "Test Market Data")
    fig.show()
    print("Interactive plot displayed!")
