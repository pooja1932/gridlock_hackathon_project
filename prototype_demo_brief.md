# ParkSense AI Demo Brief

## One-Line Pitch

ParkSense AI converts parking violation records into an illegal parking hotspot and enforcement priority system, helping traffic police decide where to act, when to act, and what action will reduce road blockage fastest.

## 60-Second Explanation

Illegal and spillover parking does not affect every road equally. A parked scooter inside a wide residential road is different from a double-parked car near a market junction during peak hour. ParkSense AI captures this difference.

The prototype ingests anonymized police violation records and creates a city-level hotspot intelligence layer. Every hotspot receives a congestion-impact score based on violation type, vehicle footprint, junction sensitivity, peak-hour timing, repeat density, and police-station jurisdiction.

The current dataset does not contain live traffic speed or density, so the prototype does not claim to use measured traffic speed. It ranks enforcement priority using available fields: violation frequency, repeat density, junction proximity, peak timing, and vehicle blockage weight. In production, real speed, density, CCTV dwell-time, or road-segment occupancy data can be added.

The output is not only a heatmap. It is an enforcement plan: top hotspots, responsible police station, best patrol hours, and recommended field action.

## Demo Flow

1. Open `outputs/parksense_command_center.html`.
2. Start with the KPI cards: records used, hotspots found, high-priority zones, top impact score.
3. Show the hotspot command map and filter by risk band.
4. Click the top hotspot: Safina Plaza Junction, Shivajinagar.
5. Explain why it ranks high:
   - 4,411 parking violations,
   - nearly half in peak hours,
   - junction-sensitive zone,
   - high repeat density.
6. Open the top hotspot table and station priority table.
7. Show the enforcement shift plan:
   - where to send patrols,
   - what hours to target,
   - what action is recommended.

## What Makes It Strong

- It is operational, not only analytical.
- It prioritizes by congestion impact, not raw violation count.
- It gives police-station level deployment priorities.
- It includes a validated hybrid ranking model that blends long-term hotspot history with last-month persistence.
- It can integrate later with camera feeds, towing logs, traffic speed APIs, and event schedules.

## Technical Components

- Data cleaning and parking violation filtering.
- Spatial hotspot binning using latitude and longitude.
- Time feature extraction from violation timestamps.
- Vehicle footprint scoring using PCU-style weights.
- Violation severity scoring.
- Junction and peak-hour weighting.
- Enforcement priority score:
  `0.40 x violation frequency + 0.25 x repeat hotspot density + 0.15 x junction proximity + 0.10 x peak-hour factor + 0.10 x vehicle blockage weight`.
- Patrol recommendation rules.
- Dashboard and CSV exports.

## Prediction Quality

This prototype is a hotspot-prioritization model, so standard classification accuracy is not the best metric. The useful question is:

> If we use historical violation data, how well can we identify the next month's high-impact parking congestion hotspots?

I used walk-forward month-wise validation:

1. Train/rank hotspots using all months before the test month.
2. Generate predicted top-k enforcement hotspots.
3. Compare them with the actual top-k high-impact hotspots in the next month.
4. Repeat this from December 2023 through April 2024.

Average validation results for the dataset-backed enforcement priority formula:

- Top-20 hotspot precision: 55.0%
- Top-50 hotspot precision: 66.4%
- Top-100 hotspot precision: 68.0%
- NDCG@20: 93.58%
- NDCG@50: 93.24%
- NDCG@100: 92.58%
- Spearman score correlation: 0.7803

Hybrid ranking model:

- Best learned blend: 70% long-term hotspot history + 30% last-month activity.
- Top-20 hotspot precision: 57.0%
- Top-50 hotspot precision: 68.4%
- Top-100 hotspot precision: 69.6%
- NDCG@50: 93.64%
- Spearman score correlation: 0.7723

This hybrid model is used as an additional model-performance layer because it improves the practical top-k enforcement hit rate while remaining simple enough to explain and reproduce.

Interpretation:

- Precision@50 means around 66% of the predicted top 50 enforcement hotspots also appear in the actual next-month top 50.
- NDCG@50 above 93% means the model ranks the most important zones near the top, which is the key need for enforcement planning.
- Spearman correlation around 0.78 shows strong stability between historical hotspot score and future hotspot score.

## Submission Assets

- `parking_intelligence_pipeline.py`: full data pipeline.
- `evaluate_prediction_quality.py`: month-wise validation and hotspot prediction quality metrics.
- `optimize_hotspot_ranking_model.py`: hybrid ranking model and model selection.
- `build_command_center_dashboard.py`: command-center dashboard builder.
- `outputs/hotspot_priority.csv`: ranked hotspot output.
- `outputs/station_priority_summary.csv`: station-level prioritization.
- `outputs/enforcement_shift_plan.csv`: deployment plan.
- `outputs/prediction_quality_backtest.csv`: validation metrics by test month.
- `outputs/prediction_quality_summary.json`: summarized validation results.
- `outputs/hybrid_model_backtest.csv`: validation metrics for the hybrid ranking model.
- `outputs/hybrid_model_ranked_hotspots.csv`: model-ranked enforcement hotspot list.
- `outputs/hybrid_model_summary.json`: final hybrid model summary.
- `outputs/parksense_command_center.html`: main prototype demo.
- `parksense_round2_concept_note.md`: written concept note.
