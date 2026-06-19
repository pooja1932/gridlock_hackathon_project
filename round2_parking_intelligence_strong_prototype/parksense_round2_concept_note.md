# ParkSense AI: Parking-Induced Congestion Intelligence

## Problem Chosen

Poor Visibility on Parking-Induced Congestion.

On-street illegal parking and spillover parking around commercial streets, metro stations, market areas, and event zones reduce usable carriageway width. Today, enforcement is mostly patrol-driven, so teams react after congestion has already formed. ParkSense AI converts historical violation data into a prioritized enforcement map.

## Core Idea

ParkSense AI detects illegal parking hotspots and ranks them for enforcement action. The current prototype uses the provided parking violation data directly: geo-location, violation type, offence code, timestamp, vehicle type, police station, and junction name.

If live speed, density, CCTV, or road-network feeds are available later, those signals can be added as extra modules without changing the dashboard flow.

Instead of ranking areas only by violation count, the system combines:

- repeated violation density,
- junction proximity,
- peak-hour occurrence,
- vehicle footprint,
- type of parking violation,
- police-station jurisdiction.

This produces a `congestion_impact_score` for every spatial hotspot and recommends where enforcement should be focused first.

## Dataset Used

The prototype uses the anonymized police violation dataset from November 2023 to April 2024.

Key fields used:

- latitude and longitude,
- created timestamp,
- police station,
- junction name,
- violation type,
- vehicle type,
- vehicle number anonymized id.

The dataset contains 298,450 records. After filtering for parking-related violations, 298,445 records are used.

## Feature Engineering

1. Time features:
   - hour of day,
   - month,
   - weekday,
   - peak-hour flag.

2. Spatial features:
   - 100m approximate grid cell,
   - junction proximity flag,
   - police-station zone.

3. Violation severity:
   - double parking, main-road parking, road-crossing parking, and traffic-light/zebra-cross parking receive higher severity.

4. Vehicle footprint:
   - scooters and motorcycles get low obstruction weight,
   - cars and autos get medium weight,
   - buses, tankers, lorries, and HGVs get high obstruction weight.

5. Repeat behaviour:
   - repeated incidents per active day are used to separate persistent hotspots from one-time spikes.

## Congestion Impact Score

Each parking event receives an `incident_impact`:

`incident_impact = violation_severity x vehicle_footprint x junction_weight x peak_hour_weight`

Each hotspot then receives a 0-100 enforcement priority score using this prototype formula:

`Priority Score = 0.40 x Violation Frequency + 0.25 x Repeat Hotspot Density + 0.15 x Junction Proximity + 0.10 x Peak-Hour Factor + 0.10 x Vehicle Blockage Weight`

Component definitions:

- Violation Frequency: normalized illegal/no-parking violation count in the spatial grid.
- Repeat Hotspot Density: repeat violations per active day in that location.
- Junction Proximity: whether violations occur near named junctions.
- Peak-Hour Factor: share of violations during morning/evening operational peaks.
- Vehicle Blockage Weight: heavier vehicles receive higher blockage weight than two-wheelers.

This makes the output operationally useful: a location with fewer but more damaging peak-hour junction violations can outrank a location with many low-impact violations.

## Prototype Outputs

The pipeline creates:

- `hotspot_priority.csv`: ranked hotspot list with impact score and suggested action.
- `station_priority_summary.csv`: police-station level prioritization.
- `enforcement_shift_plan.csv`: top hotspots with best enforcement hours.
- `month_hour_profile.csv`: temporal profile for planning.
- `parksense_dashboard.html`: lightweight dashboard preview.

## Validation Method

The system predicts and ranks future enforcement hotspots, so I evaluate it as a ranking/forecasting problem rather than a simple binary classifier.

Validation uses walk-forward next-month testing:

1. Use historical months as training data.
2. Rank hotspots using the congestion-impact score.
3. Compare predicted top-k hotspots with the actual top-k hotspots observed in the next month.
4. Repeat for each available month from December 2023 to April 2024.

Average backtest performance for the dataset-backed enforcement priority formula:

- Top-20 hotspot precision: 55.0%
- Top-50 hotspot precision: 66.4%
- Top-100 hotspot precision: 68.0%
- NDCG@50: 93.24%
- Spearman score correlation: 0.7803

This means the model is strong at placing the most operationally important hotspots near the top of the enforcement list, which is more useful for field deployment than ordinary row-level accuracy.

## Hybrid Ranking Model

Along with the transparent enforcement priority formula, I added a hybrid ranking model for next-month hotspot prioritization. The model searches a small set of explainable ranking blends and selects the one that performs best in walk-forward validation.

Selected blend:

- 70% long-term hotspot history,
- 30% last-month hotspot activity.

Hybrid model validation:

- Top-20 hotspot precision: 57.0%
- Top-50 hotspot precision: 68.4%
- Top-100 hotspot precision: 69.6%
- NDCG@50: 93.64%
- Spearman score correlation: 0.7723

This is useful operationally because enforcement teams usually act on a limited top-k list. Improving top-50 and top-100 precision means the recommended patrol zones are more likely to remain relevant in the next planning period.

## Initial Findings

Top hotspot from the data:

- Safina Plaza Junction, Shivajinagar
- 4,411 violations
- 49.3% during peak hours
- congestion impact score: 86.49
- recommended action: tow-away patrol, edge barricading, peak-hour officer.

Other high-priority zones include:

- Elite Junction, Upparpet,
- KR Market Junction, City Market,
- Sagar Theatre Junction, Upparpet,
- Central Street Junction, Shivajinagar.

Top police-station priority zones:

- Upparpet,
- Shivajinagar,
- Malleshwaram,
- City Market,
- HAL Old Airport.

## Enforcement Recommendation Logic

Critical hotspot:

- tow-away patrol,
- temporary no-parking cones or edge barricades,
- peak-hour officer deployment,
- repeat-offender tracking.

High hotspot:

- focused patrol during top two impact hours,
- daily hotspot monitoring,
- targeted warnings and towing.

Medium hotspot:

- warning campaign,
- weekend or event-day enforcement.

Watch hotspot:

- keep in dashboard for weekly review.

## Why This Can Win

The solution is not just a heatmap. It gives a decision layer:

- where to enforce,
- when to enforce,
- why that location matters,
- which police station should act,
- what operational action is recommended.

This directly addresses the challenge: poor visibility, reactive enforcement, and difficulty prioritizing zones.

## Next Phase

If real-time traffic speed or camera feed data is added, the same framework can be upgraded into a live command-center system:

- live illegal parking detection,
- traffic-speed drop correlation,
- predicted congestion recovery after towing,
- officer route optimization,
- before-after impact reporting.
