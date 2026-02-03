from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulation.market_simulator import MarketSimulator, MarketParams
from topology import persistence_homology


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    display: str
    color: str
    params: MarketParams
    intraday_transform: (
        Callable[
            [np.ndarray, np.ndarray, int, int, np.random.Generator],
            tuple[np.ndarray, np.ndarray],
        ]
        | None
    )


def analyze_market_data(df: pd.DataFrame) -> Dict:
    ts = df["Close"].to_numpy()
    embedded = persistence_homology.embed_time_series(ts)

    if embedded.shape[1] > 3:
        embedded = persistence_homology.reduce_dimension(embedded)

    diagram = persistence_homology.vietoris_rips_transform(embedded, symbol="MARKET")
    entropy_raw = persistence_homology.persistence_entropy(diagram)

    ent = np.asarray(entropy_raw, dtype=float)
    if ent.ndim == 2:
        ent = ent[0]
    ent = ent.reshape(-1)
    if ent.size < 3:
        ent = np.pad(ent, (0, 3 - ent.size), constant_values=np.nan)
    ent = ent[:3]

    return {
        "data": df,
        "embedded": embedded,
        "persistence": diagram,
        "entropy": ent,
        "entropy_H0": float(ent[0]),
        "entropy_H1": float(ent[1]),
        "entropy_H2": float(ent[2]),
    }


def aggregate_daily(
    prices: np.ndarray, volumes: np.ndarray, n_days: int, n_intraday: int
) -> pd.DataFrame:
    rows = []
    for d in range(n_days):
        s = d * n_intraday
        e = (d + 1) * n_intraday
        p = prices[s:e]
        v = volumes[s:e]
        rows.append(
            {
                "Open": float(p[0]),
                "High": float(np.max(p)),
                "Low": float(np.min(p)),
                "Close": float(p[-1]),
                "Volume": int(np.sum(v)),
            }
        )
    df = pd.DataFrame(rows)
    start_date = pd.Timestamp("2024-01-01")
    df.index = [start_date + pd.Timedelta(days=i) for i in range(len(df))]
    return df


# Intraday manipulation transforms


def pump_dump_intraday(
    prices: np.ndarray,
    volumes: np.ndarray,
    n_days: int,
    n_intraday: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    total = n_days * n_intraday
    start = (n_days // 3) * n_intraday
    end = (2 * n_days // 3) * n_intraday
    mid = (start + end) // 2

    logp = np.log(prices)

    # Pump and dump drift per step
    pump_drift = 0.00002
    dump_drift = -0.00003

    # Add drift with noise
    logp[start:mid] += pump_drift * np.arange(mid - start)
    logp[mid:end] += dump_drift * np.arange(end - mid)

    # Add a few crash impulses during dump
    crash_k = max(1, (end - mid) // 3000)
    crash_idx = rng.choice(np.arange(mid, end), size=crash_k, replace=False)
    logp[crash_idx] += rng.normal(-0.03, 0.01, size=crash_k)

    prices2 = np.exp(logp)

    ramp = np.ones(total)
    ramp[start:mid] *= np.linspace(1.2, 3.0, mid - start)
    ramp[mid:end] *= np.linspace(2.0, 4.0, end - mid)
    volumes2 = np.clip(volumes * ramp, 1, None).astype(int)

    return prices2, volumes2


def spoofing_intraday(
    prices: np.ndarray,
    volumes: np.ndarray,
    n_days: int,
    n_intraday: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    total = n_days * n_intraday
    logp = np.log(prices)

    # short bursts that revert
    n_bursts = max(5, total // 8000)
    idx = rng.choice(np.arange(100, total - 200), size=n_bursts, replace=False)

    for i in idx:
        horizon = 40
        amp = abs(rng.normal(0.012, 0.004))
        decay = np.exp(-np.linspace(0.0, 3.0, horizon))
        sign = rng.choice([-1.0, 1.0])
        logp[i : i + horizon] += sign * amp * decay

        mult = rng.uniform(3.0, 8.0)
        volumes[i : i + horizon] = np.clip(
            volumes[i : i + horizon] * mult, 1, None
        ).astype(int)

    return np.exp(logp), volumes


def layering_intraday(
    prices: np.ndarray,
    volumes: np.ndarray,
    n_days: int,
    n_intraday: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    total = n_days * n_intraday
    logp = np.log(prices)

    # noisy oscillatory pressure via AR(1)
    x = 0.0
    phi = 0.97
    sigma = 0.00025

    for t in range(1, total):
        x = phi * x + rng.normal(0.0, sigma)
        logp[t] += x
        volumes[t] = int(max(1, volumes[t] * (1.0 + 40.0 * abs(x))))

    return np.exp(logp), volumes


def run_scenario(
    spec: ScenarioSpec, n_days: int, n_intraday: int, rng: np.random.Generator
) -> Dict:
    sim = MarketSimulator(spec.params, seed=42)
    prices, volumes = sim.create_intraday_series(n_days=n_days, n_intraday=n_intraday)

    if spec.intraday_transform is not None:
        prices, volumes = spec.intraday_transform(
            prices, volumes, n_days, n_intraday, rng
        )

    df = aggregate_daily(prices, volumes, n_days, n_intraday)
    return analyze_market_data(df)


def create_report_figure(
    results: Dict[str, Dict], specs: Dict[str, ScenarioSpec]
) -> go.Figure:
    keys = list(specs.keys())
    display = [specs[k].display for k in keys]

    fig = make_subplots(
        rows=3,
        cols=4,
        subplot_titles=(
            "Control",
            "Pump and Dump",
            "Spoofing",
            "Layering",
            "Entropy H0",
            "Entropy H1",
            "H0 vs H1",
            "Detection Score",
            "Control Embedding",
            "Pump Dump Embedding",
            "Spoofing Embedding",
            "Layering Embedding",
        ),
        specs=[
            [{"type": "scatter"}] * 4,
            [{"type": "bar"}, {"type": "bar"}, {"type": "scatter"}, {"type": "bar"}],
            [{"type": "scatter3d"}] * 4,
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.05,
    )

    # Row 1: Close
    for c, k in enumerate(keys, start=1):
        df = results[k]["data"]
        fig.add_trace(
            go.Scatter(
                x=np.arange(len(df)),
                y=df["Close"],
                mode="lines",
                line=dict(color=specs[k].color, width=2),
                showlegend=False,
            ),
            row=1,
            col=c,
        )

    # Row 2: Entropy bars
    h0 = [results[k]["entropy_H0"] for k in keys]
    h1 = [results[k]["entropy_H1"] for k in keys]

    fig.add_trace(go.Bar(x=display, y=h0, name="H0"), row=2, col=1)
    fig.add_trace(go.Bar(x=display, y=h1, name="H1"), row=2, col=2)

    # Scatter H0 H1
    for k in keys:
        fig.add_trace(
            go.Scatter(
                x=[results[k]["entropy_H0"]],
                y=[results[k]["entropy_H1"]],
                mode="markers",
                marker=dict(size=12, color=specs[k].color),
                name=specs[k].display,
                showlegend=True,
            ),
            row=2,
            col=3,
        )

    # Score as z normalized distance from control estimated by control only (placeholder)
    # Better: bootstrap control, but this is a working baseline.
    ctrl = results["control"]
    d0 = [results[k]["entropy_H0"] - ctrl["entropy_H0"] for k in keys]
    d1 = [results[k]["entropy_H1"] - ctrl["entropy_H1"] for k in keys]
    score = [float(np.sqrt(d0[i] ** 2 + d1[i] ** 2)) for i in range(len(keys))]

    fig.add_trace(go.Bar(x=display, y=score, name="Score"), row=2, col=4)

    # Row 3: embedding clouds
    for c, k in enumerate(keys, start=1):
        emb = results[k]["embedded"]
        if emb.shape[1] == 1:
            emb3 = np.c_[emb[:, 0], np.zeros(len(emb)), np.zeros(len(emb))]
        elif emb.shape[1] == 2:
            emb3 = np.c_[emb[:, 0], emb[:, 1], np.zeros(len(emb))]
        else:
            emb3 = emb[:, :3]

        fig.add_trace(
            go.Scatter3d(
                x=emb3[:, 0],
                y=emb3[:, 1],
                z=emb3[:, 2],
                mode="markers",
                marker=dict(size=3, color=np.arange(len(emb3)), colorscale="Viridis"),
                showlegend=False,
            ),
            row=3,
            col=c,
        )

    fig.update_layout(
        title=dict(text="Market Manipulation Detection Report", x=0.5),
        height=1100,
        width=1600,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        margin=dict(t=80, b=60, l=40, r=40),
    )

    for a in fig.layout.annotations:
        a.font = dict(size=11)

    return fig


def main() -> None:
    n_days = 200
    n_intraday = 390
    rng = np.random.default_rng(7)

    scenario_list = [
        ScenarioSpec(
            key="control",
            display="Control",
            color="blue",
            params=MarketParams(initial_price=150.0, volatility=0.02, drift=0.0001),
            intraday_transform=None,
        ),
        ScenarioSpec(
            key="pump_and_dump",
            display="Pump and Dump",
            color="red",
            params=MarketParams(initial_price=150.0, volatility=0.02, drift=0.0001),
            intraday_transform=pump_dump_intraday,
        ),
        ScenarioSpec(
            key="spoofing",
            display="Spoofing",
            color="green",
            params=MarketParams(initial_price=150.0, volatility=0.02, drift=0.0001),
            intraday_transform=spoofing_intraday,
        ),
        ScenarioSpec(
            key="layering",
            display="Layering",
            color="orange",
            params=MarketParams(initial_price=150.0, volatility=0.02, drift=0.0001),
            intraday_transform=layering_intraday,
        ),
    ]
    specs = {s.key: s for s in scenario_list}

    results: Dict[str, Dict] = {}
    for s in scenario_list:
        results[s.key] = run_scenario(s, n_days=n_days, n_intraday=n_intraday, rng=rng)
        print(
            s.key,
            results[s.key]["entropy_H0"],
            results[s.key]["entropy_H1"],
            results[s.key]["entropy_H2"],
        )

    fig = create_report_figure(results, specs)
    fig.show()

    # Print summary table
    print("\n" + "=" * 80)
    print("DETECTION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Scenario':<20} {'H0':<12} {'H1':<12} {'H2':<12} {'Status':<15}")
    print("-" * 80)

    # Get control entropy values as baseline
    control_h0 = results["control"]["entropy_H0"]
    control_h1 = results["control"]["entropy_H1"]
    control_h2 = results["control"]["entropy_H2"]

    for key, spec in specs.items():
        if key == "control":
            continue
        h0 = results[key]["entropy_H0"]
        h1 = results[key]["entropy_H1"]
        h2 = results[key]["entropy_H2"]

        # Calculate deviations from control
        h0_dev = abs(h0 - control_h0)
        h1_dev = abs(h1 - control_h1)
        h2_dev = abs(h2 - control_h2)
        total_dev = h0_dev + h1_dev + h2_dev

        # Determine status
        if total_dev > 0.5:
            status = "DETECTED"
        elif total_dev > 0.2:
            status = "SUSPICIOUS"
        else:
            status = "NORMAL"

        print(f"{spec.display:<20} {h0:<12.4f} {h1:<12.4f} {h2:<12.4f} {status:<15}")

    print("=" * 80)
    print(
        f"\nControl Baseline - H0: {control_h0:.4f}, H1: {control_h1:.4f}, H2: {control_h2:.4f}"
    )
    print("\nDetection Thresholds:")
    print("  Total Deviation > 0.5: MANIPULATION DETECTED")
    print("  Total Deviation > 0.2: SUSPICIOUS ACTIVITY")
    print("  Total Deviation < 0.2: NORMAL")
    print("=" * 80)


if __name__ == "__main__":
    main()
