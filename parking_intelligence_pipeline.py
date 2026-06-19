from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = Path(r"C:\Users\luhar\Downloads\jan to may police violation_anonymized791b166.csv")
OUT = ROOT / "outputs"


VIOLATION_WEIGHTS = {
    "NO PARKING": 1.00,
    "WRONG PARKING": 1.05,
    "PARKING IN A MAIN ROAD": 1.40,
    "DOUBLE PARKING": 1.55,
    "PARKING NEAR ROAD CROSSING": 1.60,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 1.65,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 1.45,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 1.35,
    "PARKING OTHER THAN BUS STOP": 1.25,
    "PARKING ON FOOTPATH": 1.10,
}

VEHICLE_PCU = {
    "MOPED": 0.35,
    "SCOOTER": 0.40,
    "MOTOR CYCLE": 0.40,
    "PASSENGER AUTO": 0.75,
    "GOODS AUTO": 0.85,
    "CAR": 1.00,
    "JEEP": 1.00,
    "VAN": 1.10,
    "MAXI-CAB": 1.20,
    "LGV": 1.35,
    "TEMPO": 1.45,
    "MINI LORRY": 1.60,
    "LORRY/GOODS VEHICLE": 1.90,
    "HGV": 2.10,
    "TANKER": 2.20,
    "BUS (BMTC/KSRTC)": 2.50,
    "PRIVATE BUS": 2.50,
    "TOURIST BUS": 2.50,
    "SCHOOL VEHICLE": 2.25,
    "FACTORY BUS": 2.40,
}


def parse_list(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = ast.literal_eval(value)
    except Exception:
        return [value.strip()]
    if isinstance(parsed, list):
        return [str(x).strip().upper() for x in parsed if str(x).strip()]
    return [str(parsed).strip().upper()]


def minmax(series: pd.Series) -> pd.Series:
    lo = series.min()
    hi = series.max()
    if pd.isna(lo) or pd.isna(hi) or hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def risk_band(score: float) -> str:
    if score >= 80:
        return "Critical"
    if score >= 60:
        return "High"
    if score >= 40:
        return "Medium"
    return "Watch"


def enforcement_action(score: float) -> str:
    if score >= 80:
        return "Tow-away patrol + barricade edge + peak-hour officer"
    if score >= 60:
        return "Focused patrol during top 2 peak hours"
    if score >= 40:
        return "Warning drive + repeat-offender monitoring"
    return "Observe through weekly dashboard"


def load_and_clean(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["created_at_ist"] = pd.to_datetime(
        df["created_datetime"], errors="coerce", utc=True
    ).dt.tz_convert("Asia/Kolkata")
    df = df.dropna(subset=["created_at_ist", "latitude", "longitude"]).copy()

    df["violation_items"] = df["violation_type"].map(parse_list)
    df["parking_violation"] = df["violation_items"].map(
        lambda items: any(("PARKING" in item) or (item in {"NO PARKING", "WRONG PARKING"}) for item in items)
    )
    df = df.loc[df["parking_violation"]].copy()

    df["date"] = df["created_at_ist"].dt.date.astype(str)
    df["month"] = df["created_at_ist"].dt.to_period("M").astype(str)
    df["weekday"] = df["created_at_ist"].dt.day_name()
    df["hour"] = df["created_at_ist"].dt.hour
    df["is_peak_hour"] = df["hour"].isin([8, 9, 10, 17, 18, 19, 20]).astype(int)
    df["is_weekend"] = df["created_at_ist"].dt.dayofweek.isin([5, 6]).astype(int)
    df["near_junction"] = (
        df["junction_name"].fillna("").str.strip().ne("")
        & df["junction_name"].fillna("").str.upper().ne("NO JUNCTION")
    ).astype(int)

    df["violation_severity"] = df["violation_items"].map(
        lambda items: max([VIOLATION_WEIGHTS.get(item, 0.75) for item in items] or [0.75])
    )
    df["vehicle_pcu"] = df["vehicle_type"].fillna("OTHERS").str.upper().map(VEHICLE_PCU).fillna(1.0)
    df["junction_weight"] = np.where(df["near_junction"].eq(1), 1.25, 1.00)
    df["peak_weight"] = np.where(df["is_peak_hour"].eq(1), 1.20, 1.00)

    # About 100-110m bins around Bengaluru; enough for patrol-level hotspot ranking.
    df["grid_lat"] = df["latitude"].round(3)
    df["grid_lon"] = df["longitude"].round(3)
    df["grid_id"] = df["grid_lat"].astype(str) + "," + df["grid_lon"].astype(str)

    df["incident_impact"] = (
        df["violation_severity"]
        * df["vehicle_pcu"]
        * df["junction_weight"]
        * df["peak_weight"]
    )
    return df


def build_hotspots(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("grid_id", as_index=False).agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        grid_lat=("grid_lat", "first"),
        grid_lon=("grid_lon", "first"),
        violation_count=("id", "count"),
        unique_vehicles=("vehicle_number", "nunique"),
        impact_sum=("incident_impact", "sum"),
        impact_avg=("incident_impact", "mean"),
        peak_hour_share=("is_peak_hour", "mean"),
        junction_share=("near_junction", "mean"),
        active_days=("date", "nunique"),
        police_station=("police_station", lambda x: x.mode().iat[0] if not x.mode().empty else ""),
        junction_name=("junction_name", lambda x: x.mode().iat[0] if not x.mode().empty else ""),
        top_vehicle_type=("vehicle_type", lambda x: x.mode().iat[0] if not x.mode().empty else ""),
    )

    grouped["repeat_density"] = grouped["violation_count"] / grouped["active_days"].clip(lower=1)
    grouped["parking_violation_frequency_score"] = 100 * minmax(
        np.log1p(grouped["violation_count"])
    )
    grouped["repeat_hotspot_density_score"] = 100 * minmax(grouped["repeat_density"])
    grouped["junction_proximity_score"] = 100 * grouped["junction_share"]
    grouped["peak_hour_weight_score"] = 100 * grouped["peak_hour_share"]
    grouped["vehicle_blockage_score"] = 100 * minmax(grouped["impact_avg"])
    grouped["parking_violation_density_score"] = grouped["parking_violation_frequency_score"]
    grouped["estimated_speed_drop_score"] = 100 * minmax(
        np.log1p(grouped["impact_sum"]) * (0.65 + 0.35 * grouped["peak_hour_share"])
    )
    grouped["road_occupancy_score"] = grouped["vehicle_blockage_score"]
    grouped["road_importance_score"] = grouped["junction_proximity_score"]
    grouped["congestion_impact_score"] = (
        0.40 * grouped["parking_violation_frequency_score"]
        + 0.25 * grouped["repeat_hotspot_density_score"]
        + 0.15 * grouped["junction_proximity_score"]
        + 0.10 * grouped["peak_hour_weight_score"]
        + 0.10 * grouped["vehicle_blockage_score"]
    )
    grouped["score_formula"] = (
        "0.40*ViolationFrequency + 0.25*RepeatHotspotDensity "
        "+ 0.15*JunctionProximity + 0.10*PeakHourFactor "
        "+ 0.10*VehicleBlockageWeight"
    )
    grouped["risk_band"] = grouped["congestion_impact_score"].map(risk_band)
    grouped["recommended_action"] = grouped["congestion_impact_score"].map(enforcement_action)
    grouped = grouped.sort_values("congestion_impact_score", ascending=False).reset_index(drop=True)
    grouped.insert(0, "priority_rank", np.arange(1, len(grouped) + 1))
    return grouped


def build_station_summary(df: pd.DataFrame, hotspots: pd.DataFrame) -> pd.DataFrame:
    station = df.groupby("police_station", as_index=False).agg(
        violation_count=("id", "count"),
        impact_sum=("incident_impact", "sum"),
        peak_hour_share=("is_peak_hour", "mean"),
        junction_share=("near_junction", "mean"),
        unique_hotspots=("grid_id", "nunique"),
    )
    critical = hotspots.loc[hotspots["risk_band"].isin(["Critical", "High"])]
    high_counts = critical.groupby("police_station").size().rename("high_priority_hotspots")
    station = station.merge(high_counts, on="police_station", how="left")
    station["high_priority_hotspots"] = station["high_priority_hotspots"].fillna(0).astype(int)
    station["station_priority_score"] = 100 * (
        0.45 * minmax(np.log1p(station["impact_sum"]))
        + 0.25 * minmax(np.log1p(station["violation_count"]))
        + 0.20 * minmax(station["high_priority_hotspots"])
        + 0.10 * station["peak_hour_share"]
    )
    return station.sort_values("station_priority_score", ascending=False).reset_index(drop=True)


def build_shift_plan(df: pd.DataFrame, hotspots: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    top_ids = hotspots.head(top_n)["grid_id"]
    hourly = (
        df.loc[df["grid_id"].isin(top_ids)]
        .groupby(["grid_id", "hour"], as_index=False)
        .agg(events=("id", "count"), impact=("incident_impact", "sum"))
    )
    rows = []
    meta = hotspots.set_index("grid_id")
    for grid_id, part in hourly.groupby("grid_id"):
        top_hours = part.sort_values(["impact", "events"], ascending=False).head(3)
        h = meta.loc[grid_id]
        rows.append(
            {
                "priority_rank": int(h["priority_rank"]),
                "grid_id": grid_id,
                "latitude": h["latitude"],
                "longitude": h["longitude"],
                "police_station": h["police_station"],
                "junction_name": h["junction_name"],
                "risk_band": h["risk_band"],
                "congestion_impact_score": round(float(h["congestion_impact_score"]), 2),
                "best_enforcement_hours": ", ".join(f"{int(x):02d}:00" for x in top_hours["hour"]),
                "recommended_action": h["recommended_action"],
            }
        )
    return pd.DataFrame(rows).sort_values("priority_rank")


def write_dashboard(hotspots: pd.DataFrame, station: pd.DataFrame, shift_plan: pd.DataFrame) -> None:
    top_points = hotspots.head(120).copy()
    points = top_points[
        [
            "priority_rank",
            "latitude",
            "longitude",
            "police_station",
            "junction_name",
            "violation_count",
            "congestion_impact_score",
            "risk_band",
            "recommended_action",
        ]
    ].round({"latitude": 6, "longitude": 6, "congestion_impact_score": 2})
    data_json = json.dumps(points.to_dict(orient="records"))
    top_table = hotspots.head(15)[
        [
            "priority_rank",
            "police_station",
            "junction_name",
            "violation_count",
            "congestion_impact_score",
            "risk_band",
            "recommended_action",
        ]
    ].round(2).to_html(index=False)
    station_table = station.head(10).round(2).to_html(index=False)
    shift_table = shift_plan.head(20).round(2).to_html(index=False)

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>ParkSense AI - Parking Congestion Intelligence</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; color: #14213d; background: #f7f8fa; }}
    h1, h2 {{ margin-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin: 18px 0; }}
    .card {{ background: white; border: 1px solid #d9dee7; padding: 14px; border-radius: 8px; }}
    .metric {{ font-size: 28px; font-weight: 700; color: #0b5cad; }}
    table {{ border-collapse: collapse; width: 100%; background: white; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #d9dee7; padding: 8px; font-size: 13px; vertical-align: top; }}
    th {{ background: #edf2f7; text-align: left; }}
    #map {{ height: 520px; background: white; border: 1px solid #d9dee7; border-radius: 8px; position: relative; overflow: hidden; }}
    .dot {{ position: absolute; border-radius: 50%; opacity: .72; transform: translate(-50%, -50%); border: 1px solid white; }}
    .legend span {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 6px; }}
  </style>
</head>
<body>
  <h1>ParkSense AI</h1>
  <p>Illegal parking hotspot detection and congestion-impact prioritization for targeted enforcement.</p>
  <div class="grid">
    <div class="card"><div class="metric">{int(hotspots['violation_count'].sum()):,}</div><div>parking violations analysed</div></div>
    <div class="card"><div class="metric">{len(hotspots):,}</div><div>spatial hotspots</div></div>
    <div class="card"><div class="metric">{int((hotspots['risk_band'] == 'Critical').sum()):,}</div><div>critical hotspots</div></div>
    <div class="card"><div class="metric">{station['police_station'].nunique():,}</div><div>police stations covered</div></div>
  </div>

  <h2>Hotspot Map Preview</h2>
  <p class="legend"><span style="background:#d62828"></span>Critical <span style="background:#f77f00"></span>High <span style="background:#fcbf49"></span>Medium <span style="background:#2a9d8f"></span>Watch</p>
  <div id="map"></div>

  <h2>Top Enforcement Hotspots</h2>
  {top_table}

  <h2>Station Priority Summary</h2>
  {station_table}

  <h2>Suggested Shift Plan</h2>
  {shift_table}

<script>
const points = {data_json};
const map = document.getElementById('map');
const minLat = Math.min(...points.map(p => p.latitude));
const maxLat = Math.max(...points.map(p => p.latitude));
const minLon = Math.min(...points.map(p => p.longitude));
const maxLon = Math.max(...points.map(p => p.longitude));
function color(band) {{
  return band === 'Critical' ? '#d62828' : band === 'High' ? '#f77f00' : band === 'Medium' ? '#fcbf49' : '#2a9d8f';
}}
for (const p of points) {{
  const x = 5 + 90 * ((p.longitude - minLon) / Math.max(0.000001, maxLon - minLon));
  const y = 95 - 90 * ((p.latitude - minLat) / Math.max(0.000001, maxLat - minLat));
  const d = document.createElement('div');
  const size = Math.max(8, Math.min(34, 7 + p.congestion_impact_score / 4));
  d.className = 'dot';
  d.style.left = x + '%';
  d.style.top = y + '%';
  d.style.width = size + 'px';
  d.style.height = size + 'px';
  d.style.background = color(p.risk_band);
  d.title = `#${{p.priority_rank}} ${{p.police_station}} | ${{p.junction_name}} | score ${{p.congestion_impact_score}} | ${{p.recommended_action}}`;
  map.appendChild(d);
}}
</script>
</body>
</html>"""
    (OUT / "parksense_dashboard.html").write_text(html, encoding="utf-8")


def main(input_path: Path = DEFAULT_INPUT) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_and_clean(input_path)
    hotspots = build_hotspots(df)
    station = build_station_summary(df, hotspots)
    shift_plan = build_shift_plan(df, hotspots)

    df.head(5000).to_csv(OUT / "cleaned_sample_records.csv", index=False)
    hotspots.to_csv(OUT / "hotspot_priority.csv", index=False)
    station.to_csv(OUT / "station_priority_summary.csv", index=False)
    shift_plan.to_csv(OUT / "enforcement_shift_plan.csv", index=False)
    (
        df.groupby(["month", "hour"], as_index=False)
        .agg(violations=("id", "count"), impact=("incident_impact", "sum"))
        .to_csv(OUT / "month_hour_profile.csv", index=False)
    )
    write_dashboard(hotspots, station, shift_plan)

    summary = {
        "input_rows": int(pd.read_csv(input_path, usecols=["id"]).shape[0]),
        "parking_rows_used": int(len(df)),
        "hotspots": int(len(hotspots)),
        "critical_hotspots": int((hotspots["risk_band"] == "Critical").sum()),
        "top_hotspot": hotspots.head(1).to_dict(orient="records")[0],
        "outputs": [
            "hotspot_priority.csv",
            "station_priority_summary.csv",
            "enforcement_shift_plan.csv",
            "month_hour_profile.csv",
            "cleaned_sample_records.csv",
            "parksense_dashboard.html",
        ],
    }
    (OUT / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
