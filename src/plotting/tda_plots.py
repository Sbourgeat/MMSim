import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import warnings
from typing import Dict, List, Optional, Tuple, Any


class TDAPlotter:
    """Specialized plotting for Topological Data Analysis visualization"""

    def __init__(self, theme: str = "plotly_dark"):
        self.theme = theme

    def create_point_cloud_plot(
        self,
        point_cloud: np.ndarray,
        title: str = "Point Cloud",
        color_by_time: bool = True,
    ) -> go.Figure:
        """Create 3D point cloud visualization"""

        if point_cloud.shape[1] < 3:
            warnings.warn("Point cloud has fewer than 3 dimensions, creating 2D plot")
            return self._create_2d_point_cloud(point_cloud, title)

        # Color configuration
        if color_by_time:
            colors = list(range(len(point_cloud)))
            colorscale = "Viridis"
        else:
            colors = point_cloud[:, 0]  # Color by first dimension
            colorscale = "Plasma"

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=point_cloud[:, 0],
                    y=point_cloud[:, 1],
                    z=point_cloud[:, 2],
                    mode="markers",
                    marker=dict(
                        size=4,
                        color=colors,
                        colorscale=colorscale,
                        opacity=0.8,
                        colorbar=dict(title="Time" if color_by_time else "Value"),
                    ),
                    name="Embedded Points",
                )
            ]
        )

        fig.update_layout(
            template=self.theme,
            title=title,
            scene=dict(xaxis_title="Dim 1", yaxis_title="Dim 2", zaxis_title="Dim 3"),
        )

        return fig

    def _create_2d_point_cloud(self, point_cloud: np.ndarray, title: str) -> go.Figure:
        """Create 2D point cloud for lower dimensional data"""

        if point_cloud.shape[1] < 2:
            warnings.warn("Point cloud has only 1 dimension, creating line plot")
            return self._create_1d_point_cloud(point_cloud, title)

        colors = list(range(len(point_cloud)))

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=point_cloud[:, 0],
                    y=point_cloud[:, 1],
                    mode="markers",
                    marker=dict(
                        size=6,
                        color=colors,
                        colorscale="Viridis",
                        opacity=0.8,
                        colorbar=dict(title="Time"),
                    ),
                    name="Embedded Points",
                )
            ]
        )

        fig.update_layout(
            template=self.theme, title=title, xaxis_title="Dim 1", yaxis_title="Dim 2"
        )

        return fig

    def _create_1d_point_cloud(self, point_cloud: np.ndarray, title: str) -> go.Figure:
        """Create line plot for 1D data"""

        fig = go.Figure(
            data=[
                go.Scatter(
                    x=list(range(len(point_cloud))),
                    y=point_cloud.flatten(),
                    mode="lines+markers",
                    line=dict(color="cyan", width=2),
                    marker=dict(size=4),
                    name="Time Series",
                )
            ]
        )

        fig.update_layout(
            template=self.theme, title=title, xaxis_title="Time", yaxis_title="Value"
        )

        return fig

    def create_persistence_diagram(
        self, persistence_diagram: np.ndarray, title: str = "Persistence Diagram"
    ) -> go.Figure:
        """Create persistence diagram visualization"""

        fig = go.Figure()

        # Plot different homology dimensions
        colors = ["red", "green", "blue"]
        labels = ["H0 (Components)", "H1 (Loops)", "H2 (Voids)"]
        sizes = [8, 6, 6]  # Different sizes for different dimensions

        for i in range(min(3, len(persistence_diagram[0]))):
            if len(persistence_diagram[0][i]) > 0:
                points = persistence_diagram[0][i]
                fig.add_trace(
                    go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode="markers",
                        marker=dict(color=colors[i], size=sizes[i], opacity=0.8),
                        name=labels[i],
                        hovertemplate="<b>%{fullData.name}</b><br>"
                        + "Birth: %{x:.4f}<br>"
                        + "Death: %{y:.4f}<br>"
                        + "Persistence: %{y:.4f - x:.4f}<extra></extra>",
                    )
                )

        # Add diagonal line
        max_val = max(
            [
                points.max() if len(points) > 0 else 0
                for points in persistence_diagram[0]
            ]
            + [1]
        )

        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(color="gray", dash="dash", width=1),
                name="Diagonal",
                hoverinfo="skip",
            )
        )

        fig.update_layout(
            template=self.theme,
            title=title,
            xaxis=dict(title="Birth", range=[0, max_val]),
            yaxis=dict(title="Death", range=[0, max_val]),
            showlegend=True,
        )

        return fig

    def create_entropy_plot(
        self,
        entropy_values: List[List[float]],
        time_points: Optional[List[int]] = None,
        title: str = "Persistence Entropy Over Time",
    ) -> go.Figure:
        """Create entropy evolution plot"""

        if time_points is None:
            time_points = list(range(len(entropy_values[0])))

        fig = go.Figure()

        # Plot entropy for each homology dimension
        labels = ["H0 Entropy", "H1 Entropy", "H2 Entropy"]
        colors = ["red", "green", "blue"]

        for i in range(min(3, len(entropy_values))):
            if len(entropy_values[i]) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=time_points,
                        y=entropy_values[i],
                        mode="lines+markers",
                        line=dict(color=colors[i], width=2),
                        marker=dict(size=4),
                        name=labels[i],
                    )
                )

        fig.update_layout(
            template=self.theme,
            title=title,
            xaxis=dict(title="Time Step"),
            yaxis=dict(title="Entropy"),
            showlegend=True,
        )

        return fig

    def create_bottleneck_distance_plot(
        self,
        diagrams: List[np.ndarray],
        labels: List[str],
        title: str = "Bottleneck Distances",
    ) -> go.Figure:
        """Create bottleneck distance comparison plot"""

        # Calculate bottleneck distances between diagrams
        n_diagrams = len(diagrams)
        distance_matrix = np.zeros((n_diagrams, n_diagrams))

        for i in range(n_diagrams):
            for j in range(i + 1, n_diagrams):
                # Simplified bottleneck distance calculation
                # In practice, you'd use a proper TDA library
                dist = self._approximate_bottleneck_distance(diagrams[i], diagrams[j])
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist

        fig = go.Figure(
            data=go.Heatmap(
                z=distance_matrix,
                x=labels,
                y=labels,
                colorscale="Viridis",
                colorbar=dict(title="Bottleneck Distance"),
            )
        )

        fig.update_layout(template=self.theme, title=title)

        return fig

    def _approximate_bottleneck_distance(
        self, diag1: np.ndarray, diag2: np.ndarray
    ) -> float:
        """Approximate bottleneck distance calculation (simplified)"""
        # This is a simplified version - real implementation would use proper TDA algorithms
        if len(diag1[0]) == 0 or len(diag2[0]) == 0:
            return float("inf")

        # Use simple average persistence difference as approximation
        avg_persistence_1 = (
            np.mean(diag1[0][:, 1] - diag1[0][:, 0]) if len(diag1[0]) > 0 else 0
        )
        avg_persistence_2 = (
            np.mean(diag2[0][:, 1] - diag2[0][:, 0]) if len(diag2[0]) > 0 else 0
        )

        return abs(avg_persistence_1 - avg_persistence_2)

    def create_comprehensive_tda_plot(
        self,
        point_cloud: np.ndarray,
        persistence_diagram: np.ndarray,
        entropy_values: List[List[float]],
        title: str = "Comprehensive TDA Analysis",
    ) -> go.Figure:
        """Create comprehensive TDA visualization with all components"""

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Point Cloud",
                "Persistence Diagram",
                "Persistence Entropy",
                "Feature Statistics",
            ),
            specs=[
                [{"type": "scatter3d"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "bar"}],
            ],
        )

        # Point Cloud
        if point_cloud.shape[1] >= 3:
            colors = list(range(len(point_cloud)))
            fig.add_trace(
                go.Scatter3d(
                    x=point_cloud[:, 0],
                    y=point_cloud[:, 1],
                    z=point_cloud[:, 2],
                    mode="markers",
                    marker=dict(
                        size=3, color=colors, colorscale="Viridis", opacity=0.8
                    ),
                    name="Points",
                ),
                row=1,
                col=1,
            )

        # Persistence Diagram
        for i in range(min(3, len(persistence_diagram[0]))):
            if len(persistence_diagram[0][i]) > 0:
                points = persistence_diagram[0][i]
                colors = ["red", "green", "blue"]
                labels = ["H0", "H1", "H2"]
                fig.add_trace(
                    go.Scatter(
                        x=points[:, 0],
                        y=points[:, 1],
                        mode="markers",
                        marker=dict(color=colors[i], size=6),
                        name=labels[i],
                    ),
                    row=1,
                    col=2,
                )

        # Entropy Plot
        time_points = list(range(len(entropy_values[0])))
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=entropy_values[0],
                mode="lines+markers",
                name="H0 Entropy",
                line=dict(color="red"),
            ),
            row=2,
            col=1,
        )

        # Feature Statistics
        feature_counts = [
            len(persistence_diagram[0][i])
            for i in range(min(3, len(persistence_diagram[0])))
        ]
        fig.add_trace(
            go.Bar(
                x=["H0", "H1", "H2"],
                y=feature_counts,
                name="Feature Count",
                marker_color=["red", "green", "blue"],
            ),
            row=2,
            col=2,
        )

        fig.update_layout(template=self.theme, title=title, height=800, showlegend=True)

        return fig

    def create_manipulation_detection_plot(
        self,
        entropy_history: List[float],
        threshold: float,
        alerts: List[Dict[str, Any]],
        title: str = "Manipulation Detection",
    ) -> go.Figure:
        """Create plot showing manipulation detection based on entropy"""

        fig = go.Figure()

        # Entropy time series
        time_points = list(range(len(entropy_history)))
        fig.add_trace(
            go.Scatter(
                x=time_points,
                y=entropy_history,
                mode="lines+markers",
                name="Persistence Entropy",
                line=dict(color="orange", width=2),
            )
        )

        # Threshold line
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Manipulation Threshold",
        )

        # Mark detected manipulations
        for alert in alerts:
            if "time_index" in alert:
                fig.add_vline(
                    x=alert["time_index"],
                    line_color="red",
                    line_width=2,
                    annotation_text=alert.get("type", "Alert"),
                )

        fig.update_layout(
            template=self.theme,
            title=title,
            xaxis=dict(title="Time Step"),
            yaxis=dict(title="Entropy"),
            showlegend=True,
        )

        return fig


# Utility functions for TDA plotting
def create_sample_tda_data() -> Tuple[np.ndarray, np.ndarray, List[List[float]]]:
    """Create sample TDA data for testing"""

    # Create sample point cloud
    n_points = 100
    t = np.linspace(0, 4 * np.pi, n_points)

    # 3D point cloud following a spiral pattern
    point_cloud = np.column_stack(
        [
            np.cos(t) * np.linspace(0.1, 1, n_points),
            np.sin(t) * np.linspace(0.1, 1, n_points),
            t / (4 * np.pi),
        ]
    )

    # Create sample persistence diagram
    persistence_diagram = [
        np.array([[0.1, 0.8], [0.2, 0.6], [0.15, 0.9]]),  # H0
        np.array([[0.3, 0.7], [0.25, 0.8]]),  # H1
        np.array([[0.4, 0.6]]),  # H2
    ]

    # Create sample entropy values
    entropy_values = [
        np.linspace(0.5, 1.5, 50),  # H0 entropy
        np.linspace(0.3, 1.2, 50),  # H1 entropy
        np.linspace(0.2, 0.8, 50),  # H2 entropy
    ]

    return point_cloud, persistence_diagram, entropy_values


if __name__ == "__main__":
    # Test TDA plotting functions
    plotter = TDAPlotter()

    # Create sample data
    point_cloud, persistence_diagram, entropy_values = create_sample_tda_data()

    # Test all plot types
    point_cloud_fig = plotter.create_point_cloud_plot(point_cloud)
    point_cloud_fig.show()

    persistence_fig = plotter.create_persistence_diagram(persistence_diagram)
    persistence_fig.show()

    entropy_fig = plotter.create_entropy_plot(entropy_values)
    entropy_fig.show()

    comprehensive_fig = plotter.create_comprehensive_tda_plot(
        point_cloud, persistence_diagram, entropy_values
    )
    comprehensive_fig.show()

    print("TDA plotting tests completed successfully!")
