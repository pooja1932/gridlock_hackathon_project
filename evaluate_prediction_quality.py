from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from parking_intelligence_pipeline import DEFAULT_INPUT, OUT, build_hotspots, load_and_clean


def dcg(relevance: np.ndarray) -> float:
    if len(relevance) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(relevance) + 2))
    return float(np.sum(relevance / discounts))


def ndcg_at_k(predicted: list[str], actual_scores: dict[str, float], k: int) -> float:
    ranked = predicted[:k]
    rel = np.array([actual_scores.get(grid_id, 0.0) for grid_id in ranked], dtype=float)
    ideal = np.array(sorted(actual_scores.values(), reverse=True)[:k], dtype=float)
    ideal_score = dcg(ideal)
    if ideal_score == 0:
        return 0.0
    return dcg(rel) / ideal_score


def spearman_without_scipy(left: pd.Series, right: pd.Series) -> float:
    left_rank = left.rank(method="average")
    right_rank = right.rank(method="average")
    return float(left_rank.corr(right_rank, method="pearson"))


def backtest_next_month(df: pd.DataFrame, k_values: tuple[int, ...] = (20, 50, 100)) -> pd.DataFrame:
    months = sorted(df["month"].dropna().unique())
    rows = []

    for i in range(1, len(months)):
        train_months = months[:i]
        test_month = months[i]
        train = df.loc[df["month"].isin(train_months)].copy()
        test = df.loc[df["month"].eq(test_month)].copy()
        if train.empty or test.empty:
            continue

        predicted_hotspots = build_hotspots(train)
        actual_hotspots = build_hotspots(test)

        predicted_rank = predicted_hotspots["grid_id"].tolist()
        actual_rank = actual_hotspots["grid_id"].tolist()
        actual_scores = dict(
            zip(actual_hotspots["grid_id"], actual_hotspots["congestion_impact_score"])
        )

        train_scores = predicted_hotspots.set_index("grid_id")["congestion_impact_score"]
        test_scores = actual_hotspots.set_index("grid_id")["congestion_impact_score"]
        common = train_scores.index.intersection(test_scores.index)
        spearman = (
            spearman_without_scipy(train_scores.loc[common], test_scores.loc[common])
            if len(common) >= 3
            else np.nan
        )

        row = {
            "train_months": ",".join(train_months),
            "test_month": test_month,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "predicted_hotspots": int(len(predicted_hotspots)),
            "actual_hotspots": int(len(actual_hotspots)),
            "common_hotspots": int(len(common)),
            "spearman_score_correlation": round(float(spearman), 4)
            if pd.notna(spearman)
            else None,
        }

        for k in k_values:
            pred_top = set(predicted_rank[:k])
            actual_top = set(actual_rank[:k])
            hits = len(pred_top & actual_top)
            row[f"precision_at_{k}"] = round(hits / k, 4)
            row[f"hit_count_at_{k}"] = hits
            row[f"ndcg_at_{k}"] = round(ndcg_at_k(predicted_rank, actual_scores, k), 4)

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(DEFAULT_INPUT)
    results = backtest_next_month(df)
    results.to_csv(OUT / "prediction_quality_backtest.csv", index=False)

    metric_cols = [
        c
        for c in results.columns
        if c.startswith("precision_at_")
        or c.startswith("ndcg_at_")
        or c == "spearman_score_correlation"
    ]
    summary = {
        "method": "walk-forward next-month hotspot backtest",
        "backtest_windows": int(len(results)),
        "metrics_mean": {
            c: round(float(results[c].dropna().mean()), 4) for c in metric_cols
        },
        "metrics_by_month": results.to_dict(orient="records"),
        "interpretation": {
            "precision_at_k": "Share of predicted top-k enforcement hotspots that also appeared in the actual top-k next month.",
            "ndcg_at_k": "Ranking quality where higher-ranked actual high-impact hotspots are rewarded more.",
            "spearman_score_correlation": "Rank correlation between historical hotspot score and next-month hotspot score for repeated grid cells.",
        },
    }
    (OUT / "prediction_quality_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
