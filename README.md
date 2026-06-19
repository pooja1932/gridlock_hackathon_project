# Round 2 Parking Intelligence Prototype

This folder contains a working prototype for the theme:

**Poor Visibility on Parking-Induced Congestion**

## How to Run

From the workspace root:

```powershell
python .\round2_parking_intelligence\parking_intelligence_pipeline.py
```

The script reads:

```text
C:\Users\luhar\Downloads\jan to may police violation_anonymized791b166.csv
```

and writes results to:

```text
round2_parking_intelligence\outputs
```

## Output Files

- `hotspot_priority.csv`  
  Ranked illegal parking hotspots with congestion-impact score and suggested action.

- `station_priority_summary.csv`  
  Police-station level prioritization for deployment planning.

- `enforcement_shift_plan.csv`  
  Top hotspots with suggested enforcement hours.

- `month_hour_profile.csv`  
  Hourly and monthly trend profile.

- `cleaned_sample_records.csv`  
  Small cleaned sample for inspection.

- `parksense_dashboard.html`  
  Lightweight dashboard preview with hotspot map and tables.

- `parksense_command_center.html`  
  Main judge-facing prototype dashboard with hotspot filters, station priorities,
  operational impact panel, hourly impact profile, shift plan, and sidebar-based
  views for Dashboard Map, Hotspot Analysis, Station Profile, Peak Hour Impact,
  and Record Management.

- `run_summary.json`  
  Pipeline run summary.

- `hybrid_model_summary.json`  
  Summary of the learned hybrid ranking model and validation metrics.

- `hybrid_model_ranked_hotspots.csv`  
  Model-ranked enforcement hotspot list.

- `hybrid_model_backtest.csv`  
  Month-wise backtest results for the hybrid ranking model.

## Main Pitch

ParkSense AI turns parking violation records into an operational enforcement-priority engine. It does not only count violations; it estimates the congestion impact of each hotspot using location, junction sensitivity, time of day, vehicle footprint, violation type, and repeat density.

## Recommended Demo Flow

1. Show the operational problem: patrol-based enforcement is reactive.
2. Open `outputs/parksense_command_center.html`.
3. Open `hotspot_priority.csv` and show the top-ranked hotspots.
4. Explain the `congestion_impact_score`.
5. Show `enforcement_shift_plan.csv` to prove the system recommends action, not just analysis.
6. End with how this can plug into cameras, towing teams, and traffic-speed feeds.

## Optional Rebuild

If the output CSVs already exist and only the command-center dashboard needs to be rebuilt:

```powershell
python .\round2_parking_intelligence\build_command_center_dashboard.py
```

To rebuild the validation and hybrid ranking layer:

```powershell
python .\round2_parking_intelligence\evaluate_prediction_quality.py
python .\round2_parking_intelligence\optimize_hotspot_ranking_model.py
python .\round2_parking_intelligence\build_command_center_dashboard.py
```
