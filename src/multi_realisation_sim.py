import numpy as np
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
from scipy import stats
from sklearn.metrics import roc_curve, auc, f1_score
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import precision_score, recall_score
from simulation import MarketSimulator, MarketParams
from main import (
    pump_dump_intraday,
    spoofing_intraday,
    layering_intraday,
    aggregate_daily,
    analyze_market_data,
)


@dataclass
class SimulationResult:
    """Store results from a single simulation run"""

    scenario: str
    entropy_H0: float
    entropy_H1: float
    entropy_H2: float
    embedding_dim: int
    embedding_delay: int
    n_features_H0: int
    n_features_H1: int
    n_features_H2: int
    price_path: np.ndarray  # Store the actual price path for visualization


def run_monte_carlo_simulations(
    scenario_name: str,
    n_simulations: int = 500,
    n_days: int = 200,
    n_intraday: int = 390,
    transform_func=None,
) -> tuple[list[SimulationResult], list[np.ndarray]]:
    """
    Run multiple simulations to build probability distribution

    Args:
        scenario_name: Name of the scenario (control, pump_dump, etc.)
        n_simulations: Number of Monte Carlo runs
        n_days: Days per simulation
        n_intraday: Intraday steps per day
        transform_func: Intraday transformation function (from main.py)

    Returns:
        Tuple of (list of SimulationResult, list of price paths)
    """
    results = []
    price_paths = []

    for seed in tqdm(
        range(n_simulations), desc=f"{scenario_name:15s}", unit=" sim", ncols=80
    ):
        # Create simulator with unique seed - THIS IS THE FIX!
        params = MarketParams()
        simulator = MarketSimulator(params, seed=seed)  # CRITICAL: Pass seed here!

        # Generate intraday data
        rng = np.random.default_rng(seed)
        prices, volumes = simulator.create_intraday_series(
            n_days=n_days, n_intraday=n_intraday
        )

        # Apply manipulation if specified
        if transform_func is not None:
            prices, volumes = transform_func(prices, volumes, n_days, n_intraday, rng)

        # Aggregate to daily
        df = aggregate_daily(prices, volumes, n_days, n_intraday)

        # Store price path for visualization
        price_paths.append(df["Close"].values.copy())

        # Analyze with TDA
        result = analyze_single_simulation(df, scenario_name, seed)
        results.append(result)

    return results, price_paths


def analyze_single_simulation(
    data: pd.DataFrame, scenario: str, seed: int
) -> SimulationResult:
    """Analyze a single simulation and extract metrics"""
    analysis = analyze_market_data(data)

    embedded = analysis["embedded"]
    diagram = analysis["persistence"]
    entropy = analysis["entropy"]

    # Count features in each dimension
    n_features = [len(diagram[0][i]) if len(diagram[0]) > i else 0 for i in range(3)]

    return SimulationResult(
        scenario=scenario,
        entropy_H0=float(entropy[0]),
        entropy_H1=float(entropy[1]),
        entropy_H2=float(entropy[2]),
        embedding_dim=embedded.shape[1],
        embedding_delay=embedded.shape[1],  # Original dimension before reduction
        n_features_H0=n_features[0],
        n_features_H1=n_features[1],
        n_features_H2=n_features[2],
        price_path=data["Close"].values.copy(),
    )


def plot_price_realizations(
    price_paths: list[np.ndarray], scenario_name: str, color: str = "blue"
) -> go.Figure:
    """
    Create a plot showing all price realizations with the mean path

    Args:
        price_paths: List of price paths from all simulations
        scenario_name: Name of the scenario for the title
        color: Color for the mean line
    """
    fig = go.Figure()

    # Convert to array for easier manipulation
    paths_array = np.array(price_paths)
    n_days = paths_array.shape[1]
    time_axis = np.arange(n_days)

    # Calculate mean and percentiles
    mean_path = np.mean(paths_array, axis=0)
    p05 = np.percentile(paths_array, 5, axis=0)
    p25 = np.percentile(paths_array, 25, axis=0)
    p75 = np.percentile(paths_array, 75, axis=0)
    p95 = np.percentile(paths_array, 95, axis=0)

    # Plot individual realizations (with low opacity)
    n_show = min(50, len(price_paths))  # Show at most 50 paths to avoid clutter
    indices = np.random.choice(len(price_paths), n_show, replace=False)

    for idx in indices:
        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=price_paths[idx],
                mode="lines",
                line=dict(color="gray", width=0.5),
                opacity=0.2,
                showlegend=False,
                hoverinfo="skip",
            )
        )

    # Plot mean path (prominent)
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=mean_path,
            mode="lines",
            name=f"Mean Path ({scenario_name})",
            line=dict(color=color, width=3),
            showlegend=True,
        )
    )

    # Add confidence bands
    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=p95,
            mode="lines",
            line=dict(color=color, width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=time_axis,
            y=p05,
            mode="lines",
            line=dict(color=color, width=0),
            fill="tonexty",
            fillcolor=f"rgba({int(color == 'red') * 255},{int(color == 'green') * 255},{int(color == 'blue') * 255}, 0.2)",
            name="5th-95th percentile",
            showlegend=True,
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Price Evolution: {scenario_name}<br><sub>{len(price_paths)} Realizations with Mean Path</sub>",
        xaxis_title="Trading Days",
        yaxis_title="Price ($)",
        height=600,
        width=1000,
        template="plotly_white",
        showlegend=True,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )

    return fig


def calculate_roc_and_f1(
    control_results: list[SimulationResult],
    manipulation_results: list[SimulationResult],
) -> tuple[dict, go.Figure]:
    """
    Calculate ROC curve, AUC, and F1 scores for detection

    Returns:
        Tuple of (metrics dict, ROC curve figure)
    """
    # Extract features (H0 and H1 entropy)
    control_features = np.array(
        [
            [r.entropy_H0, r.entropy_H1]
            for r in control_results
            if r.entropy_H0 > -0.5 and r.entropy_H1 > -0.5
        ]
    )

    manip_features = np.array(
        [
            [r.entropy_H0, r.entropy_H1]
            for r in manipulation_results
            if r.entropy_H0 > -0.5 and r.entropy_H1 > -0.5
        ]
    )

    # Create labels: 0 for control, 1 for manipulation
    X = np.vstack([control_features, manip_features])
    y_true = np.array([0] * len(control_features) + [1] * len(manip_features))

    # Calculate distance from control mean as score
    mean_control = np.mean(control_features, axis=0)
    scores = np.linalg.norm(X - mean_control, axis=1)

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # Find optimal threshold using Youden's index
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    # Predictions at optimal threshold
    y_pred = (scores >= optimal_threshold).astype(int)

    # F1 Scores
    f1 = f1_score(y_true, y_pred)

    # Precision and Recall


    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    metrics = {
        "auc": roc_auc,
        "f1_score": f1,
        "precision": precision,
        "recall": recall,
        "optimal_threshold": optimal_threshold,
        "tpr_at_optimal": tpr[optimal_idx],
        "fpr_at_optimal": fpr[optimal_idx],
    }

    # Create ROC curve plot
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC Curve (AUC = {roc_auc:.3f})",
            line=dict(color="blue", width=2),
        )
    )

    # Add diagonal (random classifier)
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random (AUC = 0.5)",
            line=dict(color="gray", dash="dash"),
        )
    )

    # Mark optimal point
    fig.add_trace(
        go.Scatter(
            x=[fpr[optimal_idx]],
            y=[tpr[optimal_idx]],
            mode="markers",
            name=f"Optimal Threshold",
            marker=dict(color="red", size=10, symbol="star"),
        )
    )

    fig.update_layout(
        title=f"ROC Curve for Manipulation Detection<br><sub>F1={f1:.3f}, Precision={precision:.3f}, Recall={recall:.3f}</sub>",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=600,
        width=700,
        template="plotly_white",
        legend=dict(yanchor="bottom", y=0.01, xanchor="right", x=0.99),
    )

    return metrics, fig


def perform_statistical_tests(
    control_results: list[SimulationResult],
    manipulation_results: list[SimulationResult],
) -> dict:
    """
    Perform statistical tests to compare distributions

    Tests:
    - T-test for mean differences
    - Kolmogorov-Smirnov test for distribution differences
    - Mann-Whitney U test (non-parametric)
    - Effect size (Cohen's d)
    """

    def extract_entropy(results, dim):
        """Extract entropy values for a specific dimension"""
        if dim == 0:
            return [r.entropy_H0 for r in results]
        elif dim == 1:
            return [r.entropy_H1 for r in results]
        else:
            return [r.entropy_H2 for r in results]

    tests = {}

    for dim, dim_name in [(0, "H0"), (1, "H1"), (2, "H2")]:
        control_data = np.array(extract_entropy(control_results, dim))
        manip_data = np.array(extract_entropy(manipulation_results, dim))

        # Remove any NaN or -1 values (indicating no features)
        control_clean = control_data[control_data > -0.5]
        manip_clean = manip_data[manip_data > -0.5]

        if len(control_clean) < 10 or len(manip_clean) < 10:
            continue

        # T-test
        t_stat, t_pval = stats.ttest_ind(control_clean, manip_clean)

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.ks_2samp(control_clean, manip_clean)

        # Mann-Whitney U test
        u_stat, u_pval = stats.mannwhitneyu(
            control_clean, manip_clean, alternative="two-sided"
        )

        # Effect size (Cohen's d)
        pooled_std = np.sqrt(
            (np.std(control_clean, ddof=1) ** 2 + np.std(manip_clean, ddof=1) ** 2) / 2
        )
        cohens_d = (
            (np.mean(manip_clean) - np.mean(control_clean)) / pooled_std
            if pooled_std > 0
            else 0
        )

        tests[dim_name] = {
            "control_mean": np.mean(control_clean),
            "control_std": np.std(control_clean),
            "manipulation_mean": np.mean(manip_clean),
            "manipulation_std": np.std(manip_clean),
            "t_statistic": t_stat,
            "t_pvalue": t_pval,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pval,
            "u_statistic": u_stat,
            "u_pvalue": u_pval,
            "cohens_d": cohens_d,
            "significant": t_pval < 0.05 or ks_pval < 0.05,
        }

    return tests


def create_distribution_plots(
    control_results: list[SimulationResult],
    manipulation_results: list[SimulationResult],
    scenario_name: str,
) -> go.Figure:
    """Create comprehensive visualization of entropy distributions"""
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=(
            "H0 Distribution (Histogram)",
            "H0 Distribution (Box Plot)",
            "H0 vs H1 Scatter",
            "H1 Distribution (Histogram)",
            "H1 Distribution (Box Plot)",
            "Cumulative H0",
            "H2 Distribution (Histogram)",
            "H2 Distribution (Box Plot)",
            "Cumulative H1",
        ),
    )

    dimensions = [
        (
            [r.entropy_H0 for r in control_results],
            [r.entropy_H0 for r in manipulation_results],
            0,
        ),
        (
            [r.entropy_H1 for r in control_results],
            [r.entropy_H1 for r in manipulation_results],
            1,
        ),
        (
            [r.entropy_H2 for r in control_results],
            [r.entropy_H2 for r in manipulation_results],
            2,
        ),
    ]

    colors = {"control": "blue", "manipulation": "red"}

    for dim_idx, (control_data, manip_data, row_offset) in enumerate(dimensions):
        row = dim_idx + 1

        # Clean data
        control_clean = np.array([x for x in control_data if x > -0.5])
        manip_clean = np.array([x for x in manip_data if x > -0.5])

        if len(control_clean) < 5:
            continue

        # Histogram
        fig.add_trace(
            go.Histogram(
                x=control_clean,
                name="Control",
                opacity=0.6,
                marker_color=colors["control"],
                nbinsx=30,
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )

        fig.add_trace(
            go.Histogram(
                x=manip_clean,
                name=scenario_name,
                opacity=0.6,
                marker_color=colors["manipulation"],
                nbinsx=30,
                showlegend=(row == 1),
            ),
            row=row,
            col=1,
        )

        # Box plot
        fig.add_trace(
            go.Box(
                y=control_clean,
                name="Control",
                marker_color=colors["control"],
                showlegend=False,
            ),
            row=row,
            col=2,
        )

        fig.add_trace(
            go.Box(
                y=manip_clean,
                name=scenario_name,
                marker_color=colors["manipulation"],
                showlegend=False,
            ),
            row=row,
            col=2,
        )

    # H0 vs H1 scatter
    fig.add_trace(
        go.Scatter(
            x=[r.entropy_H0 for r in control_results if r.entropy_H0 > -0.5],
            y=[r.entropy_H1 for r in control_results if r.entropy_H1 > -0.5],
            mode="markers",
            name="Control",
            marker=dict(color="blue", size=5, opacity=0.5),
        ),
        row=1,
        col=3,
    )

    fig.add_trace(
        go.Scatter(
            x=[r.entropy_H0 for r in manipulation_results if r.entropy_H0 > -0.5],
            y=[r.entropy_H1 for r in manipulation_results if r.entropy_H1 > -0.5],
            mode="markers",
            name=scenario_name,
            marker=dict(color="red", size=5, opacity=0.5),
        ),
        row=1,
        col=3,
    )

    fig.update_layout(
        title=f"Statistical Analysis: Control vs {scenario_name}<br><sub>Distribution Comparison of Persistence Entropies</sub>",
        height=900,
        width=1200,
        showlegend=True,
        barmode="overlay",
    )

    return fig


def main():
    print("=" * 80)
    print("STATISTICAL ENTROPY ANALYSIS FOR MANIPULATION DETECTION")
    print("=" * 80)

    N_SIMULATIONS = 5
    N_DAYS = 200
    N_INTRADAY = 390

    scenarios = {
        "pump_dump": {
            "name": "Pump and Dump",
            "transform": pump_dump_intraday,
            "color": "red",
        },
        "spoofing": {
            "name": "Spoofing",
            "transform": spoofing_intraday,
            "color": "green",
        },
        "layering": {
            "name": "Layering",
            "transform": layering_intraday,
            "color": "orange",
        },
    }

    all_price_paths = {}
    all_results = {}

    print(f"\n1. Running {N_SIMULATIONS} control (normal market) simulations...")
    control_results, control_paths = run_monte_carlo_simulations(
        "control",
        n_simulations=N_SIMULATIONS,
        n_days=N_DAYS,
        n_intraday=N_INTRADAY,
        transform_func=None,
    )
    all_price_paths["control"] = control_paths
    print(f"   Completed: {len(control_results)} simulations")

    print("   Displaying price realizations...")
    fig_control = plot_price_realizations(
        control_paths, "Control Market (Normal)", "blue"
    )
    fig_control.show()

    for key, scenario in scenarios.items():
        print(f"\n2. Running {N_SIMULATIONS} {scenario['name']} simulations...")
        results, paths = run_monte_carlo_simulations(
            key,
            n_simulations=N_SIMULATIONS,
            n_days=N_DAYS,
            n_intraday=N_INTRADAY,
            transform_func=scenario["transform"],
        )
        all_results[key] = results
        all_price_paths[key] = paths
        print(f"   Completed: {len(results)} simulations")


        print(f"   Displaying {scenario['name']} price realizations...")
        fig_manip = plot_price_realizations(paths, scenario["name"], scenario["color"])
        fig_manip.show()

    print("\n3. Performing statistical analysis...")
    print("-" * 80)

    summary_data = []

    for key, scenario in scenarios.items():
        print(f"\n{scenario['name'].upper()}")
        print("-" * 40)

        tests = perform_statistical_tests(control_results, all_results[key])

        for dim, results in tests.items():
            print(f"\n{dim} Entropy:")
            print(
                f"  Control:     μ={results['control_mean']:.4f}, σ={results['control_std']:.4f}"
            )
            print(
                f"  Manipulation: μ={results['manipulation_mean']:.4f}, σ={results['manipulation_std']:.4f}"
            )
            print(
                f"  T-test:      t={results['t_statistic']:.3f}, p={results['t_pvalue']:.4f}"
            )
            print(
                f"  KS-test:     D={results['ks_statistic']:.3f}, p={results['ks_pvalue']:.4f}"
            )
            print(f"  Effect size: d={results['cohens_d']:.3f}")

            if results["significant"]:
                print(f"  *** STATISTICALLY SIGNIFICANT DIFFERENCE ***")

        # Calculate ROC and F1
        roc_metrics, roc_fig = calculate_roc_and_f1(control_results, all_results[key])
        roc_fig.show()
        print(
            f"   ROC AUC: {roc_metrics['auc']:.3f}, F1: {roc_metrics['f1_score']:.3f}"
        )

        summary_data.append(
            {
                "Scenario": scenario["name"],
                "AUC": roc_metrics["auc"],
                "F1 Score": roc_metrics["f1_score"],
                "Precision": roc_metrics["precision"],
                "Recall": roc_metrics["recall"],
                "H0 p-value": tests.get("H0", {}).get("t_pvalue", 1.0),
                "H1 p-value": tests.get("H1", {}).get("t_pvalue", 1.0),
                "Effect Size": tests.get("H0", {}).get("cohens_d", 0.0),
            }
        )

    print("\n4. Generating entropy distribution plots...")

    for key, scenario in scenarios.items():
        fig = create_distribution_plots(
            control_results, all_results[key], scenario["name"]
        )
        fig.show()
        print(f"   Displayed: {scenario['name']} entropy distribution analysis")

    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Detection Performance Metrics")
    print("=" * 80)
    print(
        f"{'Scenario':<20} {'AUC':<8} {'F1':<8} {'Precision':<10} {'Recall':<8} {'H0 p-val':<10} {'H1 p-val':<10}"
    )
    print("-" * 80)

    for row in summary_data:
        sig_marker = (
            "***" if row["H0 p-value"] < 0.05 or row["H1 p-value"] < 0.05 else "   "
        )
        print(
            f"{row['Scenario']:<20} {row['AUC']:<8.3f} {row['F1 Score']:<8.3f} "
            f"{row['Precision']:<10.3f} {row['Recall']:<8.3f} "
            f"{row['H0 p-value']:<10.4f} {row['H1 p-value']:<10.4f} {sig_marker}"
        )

    print("=" * 80)
    print("*** = Statistically significant difference detected")
    print("=" * 80)


if __name__ == "__main__":
    main()
