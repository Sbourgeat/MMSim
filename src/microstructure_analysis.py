import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, f1_score

from simulation import MarketSimulator, MarketParams
from topology import persistence_homology
from main import pump_dump_intraday, spoofing_intraday, layering_intraday, aggregate_daily


@dataclass
class SimulationResult:
    scenario: str
    entropy_H0: float
    entropy_H1: float
    entropy_H2: float
    embedding_dim: int
    n_features_H0: int
    n_features_H1: int
    n_features_H2: int
    embedded: np.ndarray
    price_path: np.ndarray
    rv_daily: np.ndarray
    log_range_daily: np.ndarray
    log_volume_daily: np.ndarray

    @property
    def avg_realized_vol(self) -> float:
        return float(np.mean(self.rv_daily))

    @property
    def avg_log_range(self) -> float:
        return float(np.mean(self.log_range_daily))

    @property
    def avg_log_volume(self) -> float:
        return float(np.mean(self.log_volume_daily))


def realized_variance_from_intraday(prices: np.ndarray) -> float:
    p = np.asarray(prices, dtype=float)
    if p.size < 2:
        return 0.0
    lr = np.diff(np.log(np.maximum(p, 1e-12)))
    return float(np.sum(lr * lr))


def compute_microstructure_series(
    intraday_prices: np.ndarray,
    daily_df: pd.DataFrame,
    n_intraday: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_days = len(daily_df)

    log_close = np.log(daily_df["Close"].to_numpy(dtype=float))
    returns = np.diff(log_close, prepend=log_close[0])
    returns[0] = 0.0

    high = daily_df["High"].to_numpy(dtype=float)
    low = np.maximum(daily_df["Low"].to_numpy(dtype=float), 1e-12)
    log_range = np.log(high / low)

    rv = np.zeros(n_days, dtype=float)
    for d in range(n_days):
        s = d * n_intraday
        e = (d + 1) * n_intraday
        rv[d] = realized_variance_from_intraday(intraday_prices[s:e])

    log_vol = np.log1p(daily_df["Volume"].to_numpy(dtype=float))
    return returns, log_range, rv, log_vol


def microstructure_scalar_signal(
    returns: np.ndarray,
    log_range: np.ndarray,
    rv: np.ndarray,
    log_vol: np.ndarray,
    weights: Optional[np.ndarray] = None,
) -> np.ndarray:
    X = np.vstack([returns, log_range, rv, log_vol]).T
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    w = np.array([0.6, 1.0, 1.2, 0.4], dtype=float) if weights is None else np.asarray(weights, dtype=float)
    return X @ w


def analyze_with_tda(ts: np.ndarray, symbol: str) -> Dict:
    ts = np.asarray(ts, dtype=float)

    embedded = persistence_homology.embed_time_series(ts)

    if embedded.ndim != 2:
        raise ValueError("Embedded output must be 2D")

    if embedded.shape[1] > 3:
        embedded = persistence_homology.reduce_dimension(embedded, n_components=3)

    diagram = persistence_homology.vietoris_rips_transform(embedded, symbol=symbol)

    ent_raw = persistence_homology.persistence_entropy(diagram)
    ent = np.asarray(ent_raw, dtype=float)
    if ent.ndim == 2:
        ent = ent[0]
    ent = ent.reshape(-1)
    if ent.size < 3:
        ent = np.pad(ent, (0, 3 - ent.size), constant_values=np.nan)
    ent = ent[:3]

    nfeat = [len(diagram[0][i]) if len(diagram[0]) > i else 0 for i in range(3)]

    return {
        "embedded": embedded,
        "persistence": diagram,
        "entropy": ent,
        "entropy_H0": float(ent[0]),
        "entropy_H1": float(ent[1]),
        "entropy_H2": float(ent[2]),
        "n_features": nfeat,
    }


def run_monte_carlo(
    scenario_key: str,
    n_simulations: int,
    n_days: int,
    n_intraday: int,
    transform_func=None,
) -> List[SimulationResult]:
    results: List[SimulationResult] = []

    for seed in tqdm(range(n_simulations), desc=f"{scenario_key:15s}", unit=" sim", ncols=80):
        simulator = MarketSimulator(MarketParams(), seed=seed)
        rng = np.random.default_rng(seed)

        prices, volumes = simulator.create_intraday_series(n_days=n_days, n_intraday=n_intraday)

        if transform_func is not None:
            prices, volumes = transform_func(prices, volumes, n_days, n_intraday, rng)

        df = aggregate_daily(prices, volumes, n_days, n_intraday)

        ret, lrng, rv, lvol = compute_microstructure_series(prices, df, n_intraday)
        ts = microstructure_scalar_signal(ret, lrng, rv, lvol)

        analysis = analyze_with_tda(ts, symbol=scenario_key)

        results.append(
            SimulationResult(
                scenario=scenario_key,
                entropy_H0=analysis["entropy_H0"],
                entropy_H1=analysis["entropy_H1"],
                entropy_H2=analysis["entropy_H2"],
                embedding_dim=int(analysis["embedded"].shape[1]),
                n_features_H0=int(analysis["n_features"][0]),
                n_features_H1=int(analysis["n_features"][1]),
                n_features_H2=int(analysis["n_features"][2]),
                embedded=np.asarray(analysis["embedded"], dtype=float),
                price_path=df["Close"].to_numpy(dtype=float).copy(),
                rv_daily=rv,
                log_range_daily=lrng,
                log_volume_daily=lvol,
            )
        )

    return results


def _run_level_features(results: List[SimulationResult]) -> np.ndarray:
    return np.array(
        [
            [
                r.entropy_H0,
                r.entropy_H1,
                r.entropy_H2,
                r.avg_realized_vol,
                r.avg_log_range,
                r.avg_log_volume,
            ]
            for r in results
        ],
        dtype=float,
    )


def _best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    thresholds = np.unique(y_score)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = float(f1), float(t)
    return best_t, best_f1


def evaluate_binary_detector_cv(
    control_results: List[SimulationResult],
    manip_results: List[SimulationResult],
    n_splits: int = 5,
    seed: int = 0,
) -> Dict:
    X0 = _run_level_features(control_results)
    X1 = _run_level_features(manip_results)

    X = np.vstack([X0, X1])
    y = np.hstack([np.zeros(len(X0), dtype=int), np.ones(len(X1), dtype=int)])

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
        ]
    )

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y_score = np.zeros_like(y, dtype=float)

    for tr, te in cv.split(X, y):
        model.fit(X[tr], y[tr])
        y_score[te] = model.predict_proba(X[te])[:, 1]

    fpr, tpr, _ = roc_curve(y, y_score)
    auc_val = float(auc(fpr, tpr))
    best_t, best_f1 = _best_f1_threshold(y, y_score)

    return {
        "y": y,
        "y_score": y_score,
        "fpr": fpr,
        "tpr": tpr,
        "auc": auc_val,
        "best_threshold": best_t,
        "best_f1": best_f1,
        "n_control": int(len(X0)),
        "n_manip": int(len(X1)),
    }


def roc_f1_summary_table(all_eval: Dict[str, Dict], scenarios_meta: Dict[str, Dict]) -> go.Figure:
    rows = []
    for key, ev in all_eval.items():
        rows.append(
            [
                scenarios_meta[key]["name"],
                ev["n_control"],
                ev["n_manip"],
                f"{ev['auc']:.3f}",
                f"{ev['best_f1']:.3f}",
                f"{ev['best_threshold']:.4f}",
            ]
        )

    header = ["Scenario", "N control", "N manip", "AUC", "Best F1", "Best threshold"]
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=header, align="center"),
                cells=dict(values=list(map(list, zip(*rows))), align="center"),
            )
        ]
    )
    fig.update_layout(title="Detection Metrics (Cross validated)", height=360, width=980)
    return fig


def plot_rocs(all_eval: Dict[str, Dict], scenarios_meta: Dict[str, Dict]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash"), name="Random"))

    for key, ev in all_eval.items():
        name = scenarios_meta[key]["name"]
        fig.add_trace(
            go.Scatter(
                x=ev["fpr"],
                y=ev["tpr"],
                mode="lines",
                name=f"{name} (AUC={ev['auc']:.3f})",
            )
        )

    fig.update_layout(
        title="ROC Curves (Control vs Manipulation)",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=520,
        width=900,
    )
    return fig


def plot_persistence_entropy_comparison(
    control_results: List[SimulationResult],
    all_results: Dict[str, List[SimulationResult]],
    scenarios_meta: Dict[str, Dict],
) -> go.Figure:
    order = ["control"] + list(scenarios_meta.keys())
    labels = {"control": "Control"}
    colors = {"control": "blue"}
    for k, meta in scenarios_meta.items():
        labels[k] = meta["name"]
        colors[k] = meta.get("color", "red")

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Persistence Entropy H0", "Persistence Entropy H1", "Persistence Entropy H2"),
        horizontal_spacing=0.06,
    )

    def vals(results: List[SimulationResult], dim: int) -> np.ndarray:
        if dim == 0:
            arr = np.array([r.entropy_H0 for r in results], float)
        elif dim == 1:
            arr = np.array([r.entropy_H1 for r in results], float)
        else:
            arr = np.array([r.entropy_H2 for r in results], float)
        return arr[np.isfinite(arr)]

    for col, dim in enumerate([0, 1, 2], start=1):
        for scen in order:
            res = control_results if scen == "control" else all_results[scen]
            y = vals(res, dim)
            fig.add_trace(
                go.Violin(
                    y=y,
                    name=labels[scen],
                    legendgroup=labels[scen],
                    showlegend=(col == 1),
                    box_visible=True,
                    meanline_visible=True,
                    points="all",
                    jitter=0.25,
                    scalemode="width",
                    line_color=colors[scen],
                    fillcolor=colors[scen],
                    opacity=0.55,
                ),
                row=1,
                col=col,
            )

    fig.update_layout(title="Persistence Entropy Comparison", height=520, width=1350)
    return fig


def plot_embedding_point_clouds(
    control_results: List[SimulationResult],
    all_results: Dict[str, List[SimulationResult]],
    scenarios_meta: Dict[str, Dict],
    which_run: int = 0,
) -> go.Figure:
    order = ["control"] + list(scenarios_meta.keys())
    titles = ["Control"] + [scenarios_meta[k]["name"] for k in scenarios_meta.keys()]

    fig = make_subplots(
        rows=1,
        cols=4,
        specs=[[{"type": "scatter3d"}] * 4],
        subplot_titles=tuple(titles),
        horizontal_spacing=0.03,
    )

    def emb_for(scen: str) -> np.ndarray:
        res = control_results if scen == "control" else all_results[scen]
        idx = min(which_run, len(res) - 1)
        E = np.asarray(res[idx].embedded, float)
        if E.shape[1] == 3:
            return E
        if E.shape[1] == 2:
            return np.column_stack([E[:, 0], E[:, 1], np.zeros(E.shape[0])])
        return np.column_stack([E[:, 0], np.zeros(E.shape[0]), np.zeros(E.shape[0])])

    for j, scen in enumerate(order, start=1):
        E = emb_for(scen)
        c = np.arange(E.shape[0])
        fig.add_trace(
            go.Scatter3d(
                x=E[:, 0],
                y=E[:, 1],
                z=E[:, 2],
                mode="markers",
                marker=dict(size=2.8, color=c, colorscale="Viridis", opacity=0.8),
                showlegend=False,
            ),
            row=1,
            col=j,
        )

    fig.update_layout(title="Embedding Point Clouds", height=520, width=1500)
    return fig


def main() -> None:
    N_SIMULATIONS = 50
    N_DAYS = 200
    N_INTRADAY = 390

    scenarios = {
        "pump_dump": {"name": "Pump and Dump", "transform": pump_dump_intraday, "color": "red"},
        "spoofing": {"name": "Spoofing", "transform": spoofing_intraday, "color": "green"},
        "layering": {"name": "Layering", "transform": layering_intraday, "color": "orange"},
    }

    control_results = run_monte_carlo(
        "control",
        n_simulations=N_SIMULATIONS,
        n_days=N_DAYS,
        n_intraday=N_INTRADAY,
        transform_func=None,
    )

    all_results: Dict[str, List[SimulationResult]] = {}
    for key, meta in scenarios.items():
        all_results[key] = run_monte_carlo(
            key,
            n_simulations=N_SIMULATIONS,
            n_days=N_DAYS,
            n_intraday=N_INTRADAY,
            transform_func=meta["transform"],
        )

    plot_persistence_entropy_comparison(control_results, all_results, scenarios).show()
    plot_embedding_point_clouds(control_results, all_results, scenarios, which_run=0).show()

    all_eval: Dict[str, Dict] = {}
    for key in scenarios.keys():
        all_eval[key] = evaluate_binary_detector_cv(control_results, all_results[key], n_splits=5, seed=0)

    roc_f1_summary_table(all_eval, scenarios).show()
    plot_rocs(all_eval, scenarios).show()


if __name__ == "__main__":
    main()
