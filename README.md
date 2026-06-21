# ParkSense AI

ParkSense AI is a Round 2 prototype for the **Poor Visibility on Parking-Induced Congestion** theme in Flipkart GridLock Hackathon 2.0.

The system converts geo-tagged police violation records into illegal-parking hotspots, ranks them by enforcement priority, and produces a judge-facing command center dashboard for traffic police teams.

## What The Prototype Does

- Filters illegal parking records using the violation description and offence code.
- Builds map-ready hotspot clusters from latitude and longitude.
- Scores each hotspot using repeat violations, peak-hour behavior, junction sensitivity, vehicle blockage weight, and police-station density.
- Generates station-level enforcement priorities and suggested deployment windows.
- Validates hotspot ranking quality with walk-forward month-based backtesting.
- Exports a standalone HTML dashboard that can be opened without a server.

## Dataset

Place the provided police violation CSV anywhere on your machine and pass it with `--input`.

Expected useful columns include:

- `latitude`, `longitude`
- `location`
- `vehicle_type`
- `violation_1`
- `offence_code`
- `created_date`
- `police_station`
- `junction_name`
- `validation`

The prototype does not require a live traffic-speed feed. It estimates a parking-induced congestion risk score from the provided violation data using explainable proxy features.

## Setup

Use Python 3.8+.

```bash
pip install -r requirements.txt
```

## Run

From this repository folder:

```powershell
python parking_intelligence_pipeline.py --input "C:\path\to\jan to may police violation_anonymized791b166.csv"
python evaluate_prediction_quality.py --input "C:\path\to\jan to may police violation_anonymized791b166.csv"
python optimize_hotspot_ranking_model.py --input "C:\path\to\jan to may police violation_anonymized791b166.csv"
python build_command_center_dashboard.py
```

Open the main prototype:

```text
outputs/parksense_command_center.html
```

## Main Outputs

- `outputs/hotspot_priority.csv`  
  Ranked illegal-parking hotspots with impact score and suggested action.

- `outputs/station_priority_summary.csv`  
  Police-station level prioritization for enforcement planning.

- `outputs/enforcement_shift_plan.csv`  
  Suggested deployment windows for the highest-priority hotspots.

- `outputs/month_hour_profile.csv`  
  Monthly and hourly violation pattern summary.

- `outputs/prediction_quality_summary.json`  
  Walk-forward validation summary for hotspot predictability.

- `outputs/hybrid_model_summary.json`  
  Validation metrics and selected weights for the hybrid ranking model.

- `outputs/hybrid_model_ranked_hotspots.csv`  
  Final model-ranked hotspot list.

- `outputs/parksense_command_center.html`  
  Standalone interactive dashboard for demo and judging.

## Model And Validation

The ranking layer is intentionally explainable. It combines:

- Historical hotspot impact
- Most recent month behavior
- Hotspot trend signal
- Police-station risk context
- Peak-hour and junction sensitivity

The model is evaluated with walk-forward month-based validation: previous months predict the next month. The included validation report tracks Precision@20/50/100, NDCG@20/50/100, and Spearman rank correlation.

Current validation summary from the prepared outputs:

- Precision@50: about `0.684`
- Precision@100: about `0.696`
- NDCG@50: about `0.936`
- Spearman rank correlation: about `0.772`

## Demo Flow

1. Open `outputs/parksense_command_center.html`.
2. Start with the Dashboard Map to show hotspot concentration.
3. Move to Hotspot Analysis to explain the impact score.
4. Show Station Profile for enforcement ownership.
5. Show Peak Hour Impact for deployment timing.
6. End with Record Management to prove the dashboard is traceable to source records.

## Reproducibility Note

All generated files are built from the input police violation CSV using the scripts in this repository. The dataset path is not hardcoded; pass it with `--input` so reviewers can reproduce the outputs on their machine.
