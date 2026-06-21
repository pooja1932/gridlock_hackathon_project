from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from parking_intelligence_pipeline import DEFAULT_INPUT, OUT, build_hotspots, load_and_clean


FEATURES = ["history", "last_month", "trend", "station", "peak_junction"]
WEIGHT_STEP = 0.20


def minmax_array(values: pd.Series) -> pd.Series:
    values = values.astype(float).replace([np.inf, -np.inf], np.nan).fillna(0)
    lo = values.min()
    hi = values.max()
    if hi == lo:
        return pd.Series(np.zeros(len(values)), index=values.index)
    return 100 * (values - lo) / (hi - lo)


def dcg(relevance: np.ndarray) -> float:
    discounts = np.log2(np.arange(2, len(relevance) + 2))
    return float(np.sum(relevance / discounts))


def ndcg_at_k(predicted: list[str], actual_scores: dict[str, float], k: int) -> float:
    ranked = predicted[:k]
    rel = np.array([actual_scores.get(grid_id, 0.0) for grid_id in ranked], dtype=float)
    ideal = np.array(sorted(actual_scores.values(), reverse=True)[:k], dtype=float)
    ideal_score = dcg(ideal)
    return 0.0 if ideal_score == 0 else dcg(rel) / ideal_score


def precision_at_k(predicted: list[str], actual_rank: list[str], k: int) -> float:
    return len(set(predicted[:k]) & set(actual_rank[:k])) / k


def spearman_without_scipy(left: pd.Series, right: pd.Series) -> float:
    return float(left.rank(method="average").corr(right.rank(method="average"), method="pearson"))


def weight_grid(step: float = WEIGHT_STEP) -> list[dict[str, float]]:
    return [
        {"history": 1.00, "last_month": 0.00, "trend": 0.00, "station": 0.00, "peak_junction": 0.00},
        {"history": 0.70, "last_month": 0.30, "trend": 0.00, "station": 0.00, "peak_junction": 0.00},
        {"history": 0.55, "last_month": 0.35, "trend": 0.10, "station": 0.00, "peak_junction": 0.00},
        {"history": 0.50, "last_month": 0.30, "trend": 0.10, "station": 0.10, "peak_junction": 0.00},
        {"history": 0.45, "last_month": 0.35, "trend": 0.10, "station": 0.05, "peak_junction": 0.05},
        {"history": 0.40, "last_month": 0.35, "trend": 0.10, "station": 0.10, "peak_junction": 0.05},
        {"history": 0.35, "last_month": 0.45, "trend": 0.10, "station": 0.05, "peak_junction": 0.05},
        {"history": 0.35, "last_month": 0.35, "trend": 0.15, "station": 0.10, "peak_junction": 0.05},
        {"history": 0.30, "last_month": 0.45, "trend": 0.15, "station": 0.05, "peak_junction": 0.05},
        {"history": 0.30, "last_month": 0.40, "trend": 0.10, "station": 0.10, "peak_junction": 0.10},
        {"history": 0.25, "last_month": 0.50, "trend": 0.10, "station": 0.10, "peak_junction": 0.05},
        {"history": 0.25, "last_month": 0.45, "trend": 0.15, "station": 0.10, "peak_junction": 0.05},
    ]


def safe_score_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["grid_id", "score"])
    h = build_hotspots(df)
    return h


def build_prediction_features(df: pd.DataFrame, train_months: list[str]) -> pd.DataFrame:
    last_month = train_months[-1]
    prev_month = train_months[-2] if len(train_months) >= 2 else None

    hist = safe_score_frame(df.loc[df["month"].isin(train_months)])
    last = safe_score_frame(df.loc[df["month"].eq(last_month)])
    prev = safe_score_frame(df.loc[df["month"].eq(prev_month)]) if prev_month else pd.DataFrame()

    base = hist[
        [
            "grid_id",
            "latitude",
            "longitude",
            "police_station",
            "junction_name",
            "congestion_impact_score",
            "peak_hour_weight_score",
            "junction_proximity_score",
            "repeat_hotspot_density_score",
        ]
    ].rename(columns={"congestion_impact_score": "history"})

    last_score = last[["grid_id", "congestion_impact_score"]].rename(
        columns={"congestion_impact_score": "last_month"}
    )
    base = base.merge(last_score, on="grid_id", how="left")
    if not prev.empty:
        prev_score = prev[["grid_id", "congestion_impact_score"]].rename(
            columns={"congestion_impact_score": "prev_month"}
        )
        base = base.merge(prev_score, on="grid_id", how="left")
    else:
        base["prev_month"] = 0.0

    station = (
        hist.groupby("police_station", as_index=False)["congestion_impact_score"]
        .mean()
        .rename(columns={"congestion_impact_score": "station"})
    )
    base = base.merge(station, on="police_station", how="left")
    base["last_month"] = base["last_month"].fillna(0)
    base["prev_month"] = base["prev_month"].fillna(0)
    base["station"] = base["station"].fillna(base["history"].median())
    base["trend_raw"] = base["last_month"] - base["prev_month"]
    base["trend"] = minmax_array(base["trend_raw"])
    base["peak_junction"] = (
        0.55 * base["peak_hour_weight_score"] + 0.45 * base["junction_proximity_score"]
    )
    for col in FEATURES:
        base[col] = minmax_array(base[col])
    return base


def rank_with_weights(features: pd.DataFrame, weights: dict[str, float]) -> pd.DataFrame:
    out = features.copy()
    out["model_priority_score"] = sum(out[col] * weight for col, weight in weights.items())
    return out.sort_values("model_priority_score", ascending=False).reset_index(drop=True)


def prepare_windows(df: pd.DataFrame) -> list[dict[str, object]]:
    months = sorted(df["month"].dropna().unique())
    windows = []
    for i in range(1, len(months)):
        train_months = months[:i]
        test_month = months[i]
        features = build_prediction_features(df, train_months)
        actual = build_hotspots(df.loc[df["month"].eq(test_month)])
        windows.append(
            {
                "features": features,
                "actual": actual,
                "actual_rank": actual["grid_id"].tolist(),
                "actual_scores": dict(zip(actual["grid_id"], actual["congestion_impact_score"])),
                "actual_series": actual.set_index("grid_id")["congestion_impact_score"],
                "meta": {
            "test_month": test_month,
            "train_months": ",".join(train_months),
                },
            }
        )
    return windows


def evaluate_prepared(windows: list[dict[str, object]], weights: dict[str, float], k_values=(20, 50, 100)) -> pd.DataFrame:
    rows = []
    for window in windows:
        features = window["features"]
        predicted = rank_with_weights(features, weights)
        predicted_rank = predicted["grid_id"].tolist()
        pred_series = predicted.set_index("grid_id")["model_priority_score"]
        actual_series = window["actual_series"]
        common = pred_series.index.intersection(actual_series.index)
        row = {
            **window["meta"],
            "common_hotspots": int(len(common)),
            "spearman_score_correlation": round(
                spearman_without_scipy(pred_series.loc[common], actual_series.loc[common]), 4
            )
            if len(common) >= 3
            else None,
        }
        for k in k_values:
            row[f"precision_at_{k}"] = round(precision_at_k(predicted_rank, window["actual_rank"], k), 4)
            row[f"ndcg_at_{k}"] = round(ndcg_at_k(predicted_rank, window["actual_scores"], k), 4)
        rows.append(row)
    return pd.DataFrame(rows)


def score_weights(results: pd.DataFrame) -> float:
    return (
        0.45 * results["ndcg_at_50"].mean()
        + 0.25 * results["precision_at_50"].mean()
        + 0.20 * results["spearman_score_correlation"].fillna(0).mean()
        + 0.10 * results["precision_at_100"].mean()
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune and export the ParkSense hybrid hotspot ranking model."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Path to the police violation CSV.",
    )
    return parser.parse_args()


def main(input_path: Path = DEFAULT_INPUT) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(input_path)
    windows = prepare_windows(df)

    candidates = []
    for weights in weight_grid():
        result = evaluate_prepared(windows, weights)
        candidates.append(
            {
                **weights,
                "objective": score_weights(result),
                "precision_at_50": float(result["precision_at_50"].mean()),
                "ndcg_at_50": float(result["ndcg_at_50"].mean()),
                "spearman": float(result["spearman_score_correlation"].fillna(0).mean()),
            }
        )
    search = pd.DataFrame(candidates).sort_values("objective", ascending=False).reset_index(drop=True)
    best_weights = search.loc[0, FEATURES].to_dict()
    best_results = evaluate_prepared(windows, best_weights)

    final_features = build_prediction_features(df, sorted(df["month"].dropna().unique()))
    final_rank = rank_with_weights(final_features, best_weights)
    final_rank.insert(0, "model_rank", np.arange(1, len(final_rank) + 1))

    search.head(30).to_csv(OUT / "model_weight_search_top30.csv", index=False)
    best_results.to_csv(OUT / "hybrid_model_backtest.csv", index=False)
    final_rank.to_csv(OUT / "hybrid_model_ranked_hotspots.csv", index=False)

    summary = {
        "model": "hybrid learned ranking model",
        "features": FEATURES,
        "best_weights": best_weights,
        "backtest_windows": int(len(best_results)),
        "metrics_mean": {
            "precision_at_20": round(float(best_results["precision_at_20"].mean()), 4),
            "precision_at_50": round(float(best_results["precision_at_50"].mean()), 4),
            "precision_at_100": round(float(best_results["precision_at_100"].mean()), 4),
            "ndcg_at_20": round(float(best_results["ndcg_at_20"].mean()), 4),
            "ndcg_at_50": round(float(best_results["ndcg_at_50"].mean()), 4),
            "ndcg_at_100": round(float(best_results["ndcg_at_100"].mean()), 4),
            "spearman_score_correlation": round(
                float(best_results["spearman_score_correlation"].fillna(0).mean()), 4
            ),
        },
        "outputs": [
            "model_weight_search_top30.csv",
            "hybrid_model_backtest.csv",
            "hybrid_model_ranked_hotspots.csv",
        ],
    }
    (OUT / "hybrid_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    args = parse_args()
    main(args.input)
