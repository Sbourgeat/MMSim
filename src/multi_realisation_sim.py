import numpy as np
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulation import MarketSimulator, MarketParams
from topology import persistence_homology
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


def run_monte_carlo_simulations(
    scenario_name: str,
    n_simulations: int = 500,
    n_days: int = 200,
    n_intraday: int = 390,
    transform_func=None,
) -> list[SimulationResult]:
    """
    Run multiple simulations to build probability distribution

    Args:
        scenario_name: Name of the scenario (control, pump_dump, etc.)
        n_simulations: Number of Monte Carlo runs
        n_days: Days per simulation
        n_intraday: Intraday steps per day
        transform_func: Intraday transformation function (from main.py)
    """
    results = []

    for seed in tqdm(
        range(n_simulations), desc=f"{scenario_name:15s}", unit=" sim", ncols=80
    ):
        # Create simulator
        params = MarketParams()
        simulator = MarketSimulator(params)

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

        # Analyze with TDA
        result = analyze_single_simulation(df, scenario_name, seed)
        results.append(result)

    return results


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
    )


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


def calculate_detection_metrics(
    control_results: list[SimulationResult],
    manipulation_results: list[SimulationResult],
) -> dict:
    """Calculate classification metrics for detection capability"""
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

    mean_control = np.mean(control_features, axis=0)
    mean_manip = np.mean(manip_features, axis=0)
    threshold = (mean_control + mean_manip) / 2

    control_correct = np.sum(
        np.linalg.norm(control_features - mean_control, axis=1)
        < np.linalg.norm(control_features - mean_manip, axis=1)
    )
    manip_correct = np.sum(
        np.linalg.norm(manip_features - mean_manip, axis=1)
        < np.linalg.norm(manip_features - mean_control, axis=1)
    )

    control_accuracy = control_correct / len(control_features)
    manip_accuracy = manip_correct / len(manip_features)

    return {
        "control_accuracy": control_accuracy,
        "manipulation_accuracy": manip_accuracy,
        "overall_accuracy": (control_correct + manip_correct)
        / (len(control_features) + len(manip_features)),
        "threshold_H0": threshold[0],
        "threshold_H1": threshold[1],
        "control_center": mean_control.tolist(),
        "manipulation_center": mean_manip.tolist(),
    }


def main():
    """Main execution: Run full statistical analysis"""
    print("=" * 80)
    print("STATISTICAL ENTROPY ANALYSIS FOR MANIPULATION DETECTION")
    print("=" * 80)

    N_SIMULATIONS = 10
    N_DAYS = 200
    N_INTRADAY = 390

    scenarios = {
        "pump_dump": {"name": "Pump and Dump", "transform": pump_dump_intraday},
        "spoofing": {"name": "Spoofing", "transform": spoofing_intraday},
        "layering": {"name": "Layering", "transform": layering_intraday},
    }

    # Step 1: Run control simulations
    print(f"\n1. Running {N_SIMULATIONS} control (normal market) simulations...")
    control_results = run_monte_carlo_simulations(
        "control",
        n_simulations=N_SIMULATIONS,
        n_days=N_DAYS,
        n_intraday=N_INTRADAY,
        transform_func=None,
    )
    print(f"   Completed: {len(control_results)} simulations")

    all_results = {}

    # Step 2: Run manipulation simulations
    for key, scenario in scenarios.items():
        print(f"\n2. Running {N_SIMULATIONS} {scenario['name']} simulations...")
        results = run_monte_carlo_simulations(
            key,
            n_simulations=N_SIMULATIONS,
            n_days=N_DAYS,
            n_intraday=N_INTRADAY,
            transform_func=scenario["transform"],
        )
        all_results[key] = results
        print(f"   Completed: {len(results)} simulations")

    # Step 3: Statistical tests
    print("\n3. Performing statistical tests...")
    print("-" * 80)

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

        metrics = calculate_detection_metrics(control_results, all_results[key])
        print(f"\nDetection Accuracy:")
        print(f"  Control correctly classified: {metrics['control_accuracy']:.1%}")
        print(
            f"  Manipulation correctly classified: {metrics['manipulation_accuracy']:.1%}"
        )
        print(f"  Overall accuracy: {metrics['overall_accuracy']:.1%}")

    # Step 4: Visualizations
    print("\n4. Generating distribution plots...")

    for key, scenario in scenarios.items():
        fig = create_distribution_plots(
            control_results, all_results[key], scenario["name"]
        )
        fig.show()
        print(f"   Displayed: {scenario['name']} distribution analysis")

    # Step 5: Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nKey Findings:")
    for key, scenario in scenarios.items():
        tests = perform_statistical_tests(control_results, all_results[key])
        significant_dims = [dim for dim, res in tests.items() if res["significant"]]
        if significant_dims:
            print(
                f"  • {scenario['name']}: Detectable in {', '.join(significant_dims)}"
            )
        else:
            print(f"  • {scenario['name']}: No significant differences found")



if __name__ == "__main__":
    main()
