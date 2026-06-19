from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"


def records(df: pd.DataFrame) -> str:
    return json.dumps(df.to_dict(orient="records"), ensure_ascii=False)


def main() -> None:
    hotspots = pd.read_csv(OUT / "hotspot_priority.csv")
    stations = pd.read_csv(OUT / "station_priority_summary.csv")
    shifts = pd.read_csv(OUT / "enforcement_shift_plan.csv")
    profile = pd.read_csv(OUT / "month_hour_profile.csv")
    hybrid_summary_path = OUT / "hybrid_model_summary.json"
    hybrid_rank_path = OUT / "hybrid_model_ranked_hotspots.csv"
    hybrid_backtest_path = OUT / "hybrid_model_backtest.csv"
    hybrid_summary = (
        json.loads(hybrid_summary_path.read_text(encoding="utf-8"))
        if hybrid_summary_path.exists()
        else {"metrics_mean": {}, "best_weights": {}}
    )
    hybrid_rank = (
        pd.read_csv(hybrid_rank_path)
        if hybrid_rank_path.exists()
        else pd.DataFrame()
    )
    hybrid_backtest = (
        pd.read_csv(hybrid_backtest_path)
        if hybrid_backtest_path.exists()
        else pd.DataFrame()
    )

    top_hotspots = hotspots.head(160).copy()
    top_hotspots["congestion_impact_score"] = top_hotspots["congestion_impact_score"].round(2)
    for col in [
        "parking_violation_density_score",
        "parking_violation_frequency_score",
        "repeat_hotspot_density_score",
        "junction_proximity_score",
        "estimated_speed_drop_score",
        "road_occupancy_score",
        "peak_hour_weight_score",
        "road_importance_score",
        "vehicle_blockage_score",
    ]:
        if col in top_hotspots.columns:
            top_hotspots[col] = top_hotspots[col].round(1)
    top_hotspots["peak_hour_share"] = (100 * top_hotspots["peak_hour_share"]).round(1)
    top_hotspots["junction_share"] = (100 * top_hotspots["junction_share"]).round(1)
    top_hotspots = top_hotspots[
        [
            "priority_rank",
            "grid_id",
            "latitude",
            "longitude",
            "police_station",
            "junction_name",
            "violation_count",
            "unique_vehicles",
            "peak_hour_share",
            "junction_share",
            "congestion_impact_score",
            "parking_violation_density_score",
            "parking_violation_frequency_score",
            "repeat_hotspot_density_score",
            "junction_proximity_score",
            "estimated_speed_drop_score",
            "road_occupancy_score",
            "peak_hour_weight_score",
            "road_importance_score",
            "vehicle_blockage_score",
            "risk_band",
            "recommended_action",
        ]
    ]

    station_view = stations.head(20).copy()
    station_view["station_priority_score"] = station_view["station_priority_score"].round(2)
    station_view["peak_hour_share"] = (100 * station_view["peak_hour_share"]).round(1)
    station_view = station_view[
        [
            "police_station",
            "violation_count",
            "high_priority_hotspots",
            "unique_hotspots",
            "peak_hour_share",
            "station_priority_score",
        ]
    ]

    shift_view = shifts.head(40).copy()
    shift_view["congestion_impact_score"] = shift_view["congestion_impact_score"].round(2)
    shift_view = shift_view[
        [
            "priority_rank",
            "police_station",
            "junction_name",
            "risk_band",
            "congestion_impact_score",
            "best_enforcement_hours",
            "recommended_action",
        ]
    ]

    hourly = profile.groupby("hour", as_index=False).agg(
        violations=("violations", "sum"),
        impact=("impact", "sum"),
    )
    hourly["impact"] = hourly["impact"].round(2)

    hybrid_view = hybrid_rank.head(20).copy()
    if not hybrid_view.empty:
        hybrid_view["model_priority_score"] = hybrid_view["model_priority_score"].round(2)
        hybrid_view["history"] = hybrid_view["history"].round(2)
        hybrid_view["last_month"] = hybrid_view["last_month"].round(2)
        hybrid_view = hybrid_view[
            [
                "model_rank",
                "police_station",
                "junction_name",
                "model_priority_score",
                "history",
                "last_month",
            ]
        ]
    backtest_view = hybrid_backtest.copy()
    if not backtest_view.empty:
        for col in [
            "precision_at_20",
            "precision_at_50",
            "precision_at_100",
            "ndcg_at_50",
            "spearman_score_correlation",
        ]:
            backtest_view[col] = backtest_view[col].round(4)
        backtest_view = backtest_view[
            [
                "test_month",
                "precision_at_20",
                "precision_at_50",
                "precision_at_100",
                "ndcg_at_50",
                "spearman_score_correlation",
            ]
        ]

    critical = int((hotspots["risk_band"] == "Critical").sum())
    high = int((hotspots["risk_band"] == "High").sum())
    medium = int((hotspots["risk_band"] == "Medium").sum())
    total_violations = int(hotspots["violation_count"].sum())
    total_hotspots = int(len(hotspots))
    top_score = float(hotspots["congestion_impact_score"].max())
    top_station = station_view.iloc[0]["police_station"]
    top_junction = top_hotspots.iloc[0]["junction_name"]
    model_metrics = hybrid_summary.get("metrics_mean", {})
    model_precision50 = 100 * float(model_metrics.get("precision_at_50", 0))
    model_ndcg50 = 100 * float(model_metrics.get("ndcg_at_50", 0))
    model_spearman = float(model_metrics.get("spearman_score_correlation", 0))
    model_weights = hybrid_summary.get("best_weights", {})

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>ParkSense AI Command Center</title>
  <style>
    :root {{
      --ink: #172033;
      --muted: #647083;
      --line: #d8dee8;
      --page: #eef3f8;
      --panel: #ffffff;
      --nav: #101827;
      --blue: #0b5cad;
      --cyan: #06a6b8;
      --green: #138a50;
      --amber: #e69f00;
      --orange: #e76f00;
      --red: #c62828;
      --violet: #6f42c1;
      --shadow: 0 14px 36px rgba(22, 32, 51, .08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Inter, Segoe UI, Arial, sans-serif;
      color: var(--ink);
      background:
        linear-gradient(180deg, #f7f9fd 0%, #eef3f8 46%, #e8eef6 100%);
      letter-spacing: 0;
    }}
    .app-shell {{ min-height: 100vh; display: grid; grid-template-columns: 276px 1fr; }}
    aside {{
      background:
        linear-gradient(180deg, #101827 0%, #132033 62%, #0c1422 100%);
      color: white;
      padding: 22px 18px;
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      border-right: 1px solid rgba(255,255,255,.08);
    }}
    .brand {{ display: flex; gap: 12px; align-items: center; margin-bottom: 26px; }}
    .mark {{
      width: 42px;
      height: 42px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: linear-gradient(135deg, #08a6b8, #f4b000);
      font-weight: 800;
      color: #101827;
      box-shadow: 0 12px 28px rgba(0,0,0,.24);
    }}
    .brand h1 {{ font-size: 22px; line-height: 1.05; margin: 0; }}
    .brand span {{ color: #aab4c5; font-size: 12px; }}
    .nav-label {{ color: #7f8ba3; font-size: 11px; text-transform: uppercase; margin: 24px 0 8px; }}
    .nav-item {{
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 10px 8px;
      border-radius: 8px;
      color: #dbe3f2;
      font-size: 14px;
      width: 100%;
      border: 0;
      background: transparent;
      text-align: left;
      transition: background .18s ease, color .18s ease, transform .18s ease;
    }}
    .nav-item:hover {{ background: rgba(255,255,255,.06); transform: translateX(2px); }}
    .nav-item.active {{
      background: rgba(255, 255, 255, .11);
      color: white;
      box-shadow: inset 3px 0 0 #f4b000;
    }}
    .icon {{
      width: 19px;
      height: 19px;
      display: inline-block;
    }}
    main {{ padding: 26px; min-width: 0; }}
    .topbar {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      margin-bottom: 18px;
      background: rgba(255,255,255,.92);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px 18px;
      box-shadow: var(--shadow);
    }}
    .eyebrow {{
      color: var(--blue);
      font-size: 12px;
      font-weight: 800;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .title h2 {{ font-size: 30px; line-height: 1.1; margin: 0 0 6px; }}
    .title p {{ color: var(--muted); margin: 0; max-width: 760px; }}
    .status-pill {{
      border: 1px solid #b9d8ff;
      background: #edf6ff;
      color: #074f95;
      border-radius: 8px;
      padding: 10px 12px;
      font-size: 13px;
      min-width: 210px;
      text-align: right;
      box-shadow: inset 0 0 0 1px rgba(255,255,255,.55);
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }}
    .metric {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
      box-shadow: var(--shadow);
      min-height: 104px;
      position: relative;
      overflow: hidden;
      display: grid;
      grid-template-columns: 42px 1fr;
      gap: 12px;
      align-items: start;
    }}
    .metric::before {{
      content: "";
      position: absolute;
      inset: 0 0 auto 0;
      height: 4px;
      background: var(--blue);
    }}
    .metric:nth-child(2)::before {{ background: var(--cyan); }}
    .metric:nth-child(3)::before {{ background: var(--orange); }}
    .metric:nth-child(4)::before {{ background: var(--green); }}
    .metric-icon {{
      width: 42px;
      height: 42px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: #eef5ff;
      color: var(--blue);
      font-weight: 800;
      font-size: 17px;
    }}
    .metric:nth-child(2) .metric-icon {{ background: #e8fbfd; color: #087888; }}
    .metric:nth-child(3) .metric-icon {{ background: #fff4e3; color: #a65000; }}
    .metric:nth-child(4) .metric-icon {{ background: #eaf7ef; color: var(--green); }}
    .metric .label {{ color: var(--muted); font-size: 11px; text-transform: uppercase; font-weight: 800; }}
    .metric .value {{ font-size: 29px; font-weight: 850; margin-top: 7px; }}
    .metric .note {{ color: var(--muted); font-size: 12px; margin-top: 8px; }}
    .view {{ display: none; }}
    .view.active {{ display: block; }}
    .view-title {{
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 12px;
      margin: 0 0 14px;
    }}
    .view-title h3 {{ margin: 0; font-size: 22px; }}
    .view-title p {{ margin: 4px 0 0; color: var(--muted); }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(340px, .8fr);
      gap: 14px;
      align-items: start;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }}
    .panel-head {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
      background: #fbfcff;
    }}
    .panel-head h3 {{ margin: 0; font-size: 17px; }}
    .panel-head span {{ color: var(--muted); font-size: 12px; }}
    .map-wrap {{ padding: 14px; }}
    #map {{
      height: 600px;
      border: 1px solid var(--line);
      border-radius: 8px;
      position: relative;
      overflow: hidden;
      background:
        linear-gradient(90deg, rgba(11,92,173,.075) 1px, transparent 1px),
        linear-gradient(rgba(11,92,173,.075) 1px, transparent 1px),
        linear-gradient(135deg, #ecf5f8 0%, #f6f8ee 100%);
      background-size: 54px 54px;
    }}
    .road {{ position: absolute; background: rgba(23,32,51,.16); border-radius: 99px; box-shadow: inset 0 0 0 2px rgba(255,255,255,.25); }}
    .road.a {{ left: -5%; top: 44%; width: 112%; height: 20px; transform: rotate(-12deg); }}
    .road.b {{ left: 14%; top: -4%; width: 20px; height: 112%; transform: rotate(18deg); }}
    .road.c {{ right: 8%; top: 5%; width: 18px; height: 96%; transform: rotate(-30deg); }}
    .map-zone {{
      position: absolute;
      background: rgba(255,255,255,.78);
      color: #455266;
      border: 1px solid rgba(216,222,232,.9);
      border-radius: 8px;
      padding: 7px 9px;
      font-size: 12px;
      font-weight: 700;
      pointer-events: none;
    }}
    .map-zone.market {{ left: 54%; top: 24%; }}
    .map-zone.metro {{ left: 21%; top: 67%; }}
    .map-zone.core {{ left: 34%; top: 38%; }}
    .dot {{
      position: absolute;
      border-radius: 50%;
      transform: translate(-50%, -50%);
      border: 2px solid white;
      box-shadow: 0 4px 13px rgba(23,32,51,.22);
      cursor: pointer;
      transition: transform .15s ease, box-shadow .15s ease, outline .15s ease;
    }}
    .dot:hover, .dot.selected {{
      transform: translate(-50%, -50%) scale(1.18);
      box-shadow: 0 8px 22px rgba(23,32,51,.32);
      outline: 3px solid rgba(11,92,173,.20);
    }}
    .map-label {{
      position: absolute;
      left: 12px;
      bottom: 12px;
      background: rgba(255,255,255,.92);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 12px;
      color: var(--muted);
    }}
    .filters {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      padding: 0 14px 14px;
    }}
    button {{
      border: 1px solid var(--line);
      background: white;
      color: var(--ink);
      border-radius: 8px;
      padding: 8px 10px;
      font: inherit;
      font-size: 13px;
      cursor: pointer;
    }}
    button:hover {{ border-color: #aeb8c8; }}
    button.active {{ background: #101827; color: white; border-color: #101827; }}
    .details {{ padding: 14px 16px; display: grid; gap: 10px; }}
    .selected-title {{ font-size: 18px; font-weight: 800; line-height: 1.2; }}
    .score-row {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
    .mini {{
      border: 1px solid var(--line);
      background: #fafbfd;
      border-radius: 8px;
      padding: 10px;
    }}
    .mini .k {{ color: var(--muted); font-size: 12px; }}
    .mini .v {{ font-size: 19px; font-weight: 800; margin-top: 4px; }}
    .action {{
      border-left: 4px solid var(--blue);
      background: #f1f7ff;
      padding: 12px;
      border-radius: 8px;
      line-height: 1.45;
    }}
    .command-strip {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
      margin: 0 0 14px;
    }}
    .command-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      box-shadow: var(--shadow);
      padding: 13px 14px;
      display: grid;
      gap: 5px;
    }}
    .command-card strong {{ font-size: 14px; }}
    .command-card span {{ color: var(--muted); font-size: 12px; line-height: 1.35; }}
    .section-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 14px;
      margin-top: 14px;
    }}
    .table-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.25fr) minmax(360px, .8fr);
      gap: 14px;
      align-items: start;
    }}
    .single-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 14px;
    }}
    .record-grid {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 14px;
    }}
    .file-card {{
      border: 1px solid var(--line);
      background: white;
      border-radius: 8px;
      padding: 14px;
      box-shadow: var(--shadow);
    }}
    .file-card strong {{ display: block; margin-bottom: 6px; }}
    .file-card span {{ color: var(--muted); font-size: 13px; line-height: 1.4; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th, td {{ padding: 10px 11px; border-bottom: 1px solid #edf0f5; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ color: var(--muted); font-size: 11px; text-transform: uppercase; background: #f5f8fc; position: sticky; top: 0; }}
    tr:hover td {{ background: #f8fbff; }}
    .band {{
      display: inline-block;
      border-radius: 999px;
      padding: 4px 8px;
      font-size: 12px;
      font-weight: 700;
    }}
    .Critical {{ color: white; background: var(--red); }}
    .High {{ color: #3f2500; background: #ffc76a; }}
    .Medium {{ color: #243b10; background: #cce98e; }}
    .Watch {{ color: white; background: var(--green); }}
    .bars {{ padding: 14px 16px; display: grid; gap: 8px; }}
    .bar-row {{ display: grid; grid-template-columns: 38px 1fr 70px; gap: 10px; align-items: center; font-size: 12px; }}
    .bar-track {{ height: 10px; background: #eef1f6; border-radius: 99px; overflow: hidden; }}
    .bar-fill {{ height: 100%; background: linear-gradient(90deg, var(--cyan), var(--amber), var(--orange)); }}
    .impact-grid {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      padding: 14px 16px;
    }}
    .impact-card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: linear-gradient(180deg, #ffffff, #f8fbff);
    }}
    .impact-card strong {{ display: block; font-size: 20px; margin-bottom: 4px; }}
    @media (max-width: 1080px) {{
      .app-shell {{ grid-template-columns: 1fr; }}
      aside {{ position: relative; height: auto; }}
      .layout, .section-grid, .table-grid, .record-grid, .metrics, .impact-grid {{ grid-template-columns: 1fr; }}
      .topbar {{ flex-direction: column; }}
      .status-pill {{ text-align: left; }}
    }}
  </style>
</head>
<body>
  <div class="app-shell">
    <aside>
      <div class="brand">
        <div class="mark">P</div>
        <div><h1>ParkSense AI</h1><span>Parking Congestion Intelligence</span></div>
      </div>
      <div class="nav-label">Views</div>
      <button class="nav-item active" data-view="dashboard-view">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 12h18"/><path d="M12 3v18"/><circle cx="12" cy="12" r="4"/></svg>
        Dashboard Map
      </button>
      <button class="nav-item" data-view="hotspot-view">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19V5"/><path d="M4 19h16"/><path d="M8 17V9"/><path d="M13 17V7"/><path d="M18 17v-5"/></svg>
        Hotspot Analysis
      </button>
      <button class="nav-item" data-view="station-view">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 19V5"/><path d="M4 19h16"/><path d="M8 17V9"/><path d="M13 17V7"/><path d="M18 17v-5"/></svg>
        Station Profile
      </button>
      <button class="nav-item" data-view="peak-view">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M8 7V3"/><path d="M16 7V3"/><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 11h18"/></svg>
        Peak Hour Impact
      </button>
      <button class="nav-item" data-view="records-view">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6"/><path d="M8 13h8"/><path d="M8 17h6"/></svg>
        Record Management
      </button>
      <button class="nav-item" data-view="model-view">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20V10"/><path d="M18 20V4"/><path d="M6 20v-6"/><path d="M3 20h18"/></svg>
        Model Performance
      </button>
      <div class="nav-label">Decision Signal</div>
      <div class="nav-item">
        <svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m12 2 8 4v6c0 5-3.5 8.5-8 10-4.5-1.5-8-5-8-10V6l8-4Z"/><path d="m9 12 2 2 4-5"/></svg>
        Priority Score: Frequency + Repeat + Junction
      </div>
    </aside>
    <main>
      <div class="topbar">
        <div class="title">
          <div class="eyebrow">Round 2 Prototype | Enforcement Intelligence</div>
          <h2>Targeted Enforcement for Parking-Induced Congestion</h2>
          <p>Ranks illegal parking hotspots using geo-tagged violation records, then converts that ranking into patrol timing and field actions.</p>
        </div>
        <div class="status-pill">Dataset window<br><strong>Nov 2023 to Apr 2024</strong></div>
      </div>

      <div class="metrics">
        <div class="metric"><div class="metric-icon">R</div><div><div class="label">Parking Records Used</div><div class="value">{total_violations:,}</div><div class="note">filtered from violation data</div></div></div>
        <div class="metric"><div class="metric-icon">H</div><div><div class="label">Spatial Hotspots</div><div class="value">{total_hotspots:,}</div><div class="note">approx. patrol-level grid cells</div></div></div>
        <div class="metric"><div class="metric-icon">!</div><div><div class="label">High/Critical Zones</div><div class="value">{high + critical:,}</div><div class="note">{critical} critical, {high} high</div></div></div>
        <div class="metric"><div class="metric-icon">S</div><div><div class="label">Top Impact Score</div><div class="value">{top_score:.1f}</div><div class="note">{top_station} jurisdiction</div></div></div>
      </div>

      <section class="view active" id="dashboard-view">
        <div class="view-title">
          <div><h3>Dashboard Map</h3><p>Map-first command view for choosing immediate enforcement zones from violation frequency, repeat hotspot density, junction proximity, peak timing, and vehicle blockage weight.</p></div>
        </div>
        <div class="command-strip">
          <div class="command-card"><strong>Detect</strong><span>Convert violation records into spatial illegal-parking hotspots.</span></div>
          <div class="command-card"><strong>Prioritize</strong><span>Score hotspots by frequency, repeats, junction proximity, peak timing, and vehicle blockage.</span></div>
          <div class="command-card"><strong>Deploy</strong><span>Recommend station-level patrol windows and field actions.</span></div>
        </div>
        <div class="layout">
          <section class="panel">
            <div class="panel-head">
              <div><h3>Illegal Parking Hotspot Map</h3><span>Dot size = congestion impact score. Colors show action urgency.</span></div>
              <div>
                <button class="filter active" data-band="All">All</button>
                <button class="filter" data-band="Critical">Critical</button>
                <button class="filter" data-band="High">High</button>
                <button class="filter" data-band="Medium">Medium</button>
              </div>
            </div>
            <div class="map-wrap">
            <div id="map">
              <div class="road a"></div><div class="road b"></div><div class="road c"></div>
              <div class="map-zone market">Market corridor</div>
              <div class="map-zone metro">Metro spillover</div>
              <div class="map-zone core">Commercial core</div>
              <div class="map-label">Bengaluru hotspot preview from anonymized latitude/longitude bins</div>
            </div>
            </div>
            <div class="filters">
              <button class="station active" data-station="All">All stations</button>
            </div>
          </section>

          <section class="panel">
            <div class="panel-head"><h3>Selected Hotspot</h3><span id="selected-rank">Priority #1</span></div>
            <div class="details">
              <div class="selected-title" id="selected-title">{top_junction}</div>
              <div id="selected-station" style="color:var(--muted)">{top_station}</div>
              <div class="score-row">
                <div class="mini"><div class="k">Impact Score</div><div class="v" id="selected-score">{top_score:.2f}</div></div>
                <div class="mini"><div class="k">Violations</div><div class="v" id="selected-count">{int(top_hotspots.iloc[0]['violation_count']):,}</div></div>
                <div class="mini"><div class="k">Peak Share</div><div class="v" id="selected-peak">{float(top_hotspots.iloc[0]['peak_hour_share']):.1f}%</div></div>
                <div class="mini"><div class="k">Risk Band</div><div class="v" id="selected-band">{top_hotspots.iloc[0]['risk_band']}</div></div>
              </div>
              <div class="score-row">
                <div class="mini"><div class="k">Violation Frequency</div><div class="v" id="selected-frequency">{float(top_hotspots.iloc[0]['parking_violation_frequency_score']):.1f}</div></div>
                <div class="mini"><div class="k">Repeat Density</div><div class="v" id="selected-repeat">{float(top_hotspots.iloc[0]['repeat_hotspot_density_score']):.1f}</div></div>
                <div class="mini"><div class="k">Junction Proximity</div><div class="v" id="selected-junction">{float(top_hotspots.iloc[0]['junction_proximity_score']):.1f}</div></div>
                <div class="mini"><div class="k">Vehicle Blockage</div><div class="v" id="selected-vehicle">{float(top_hotspots.iloc[0]['vehicle_blockage_score']):.1f}</div></div>
              </div>
              <div class="action">Formula: 0.40 x violation frequency + 0.25 x repeat hotspot density + 0.15 x junction proximity + 0.10 x peak-hour factor + 0.10 x vehicle blockage weight.</div>
              <div class="action" id="selected-action">{top_hotspots.iloc[0]['recommended_action']}</div>
            </div>
            <div class="panel-head"><h3>Expected Operational Impact</h3><span>prototype estimate</span></div>
            <div class="impact-grid">
              <div class="impact-card"><strong>18-25%</strong><span>less curbside blockage at treated hotspots</span></div>
              <div class="impact-card"><strong>2 hrs</strong><span>focused patrol window per hotspot</span></div>
              <div class="impact-card"><strong>Top 40</strong><span>zones cover the first enforcement wave</span></div>
            </div>
          </section>
        </div>
      </section>

      <section class="view" id="hotspot-view">
        <div class="view-title">
          <div><h3>Hotspot Analysis</h3><p>Ranked zones, risk labels, and recommended field actions.</p></div>
        </div>
        <div class="table-grid">
          <section class="panel">
            <div class="panel-head"><h3>Top Hotspots</h3><span>ranked by congestion impact</span></div>
            <table id="hotspot-table"></table>
          </section>
          <section class="panel">
            <div class="panel-head"><h3>Enforcement Shift Plan</h3><span>where and when to act</span></div>
            <table id="shift-table"></table>
          </section>
        </div>
      </section>

      <section class="view" id="station-view">
        <div class="view-title">
          <div><h3>Station Profile</h3><p>Police-station level prioritization for patrol allocation.</p></div>
        </div>
        <div class="table-grid">
          <section class="panel">
            <div class="panel-head"><h3>Station Priority</h3><span>deployment planning</span></div>
            <table id="station-table"></table>
          </section>
          <section class="panel">
            <div class="panel-head"><h3>Station Action Summary</h3><span>operational interpretation</span></div>
            <div class="details">
              <div class="action"><strong>Upparpet</strong><br>Highest total parking impact and the largest count of high-priority hotspots. Start with market and theatre-adjacent junctions.</div>
              <div class="action"><strong>Shivajinagar</strong><br>Critical hotspot around Safina Plaza and strong peak-hour concentration. Needs tow-away readiness during demand windows.</div>
              <div class="action"><strong>City Market</strong><br>High junction sensitivity near KR Market. Focus patrol timing around loading, shopping, and peak movement periods.</div>
            </div>
          </section>
        </div>
      </section>

      <section class="view" id="peak-view">
        <div class="view-title">
          <div><h3>Peak Hour Impact</h3><p>Hourly parking pressure profile for patrol scheduling.</p></div>
        </div>
        <div class="table-grid">
          <section class="panel">
            <div class="panel-head"><h3>Peak-Hour Impact Profile</h3><span>all parking records</span></div>
            <div class="bars" id="hour-bars"></div>
          </section>
          <section class="panel">
            <div class="panel-head"><h3>Shift Planning Rule</h3><span>how the model acts on the profile</span></div>
            <div class="details">
              <div class="mini"><div class="k">Morning Window</div><div class="v">08:00-10:00</div></div>
              <div class="mini"><div class="k">Evening Window</div><div class="v">17:00-20:00</div></div>
              <div class="action">Hotspots that combine peak-hour concentration with junction sensitivity are escalated above raw-count-only locations.</div>
            </div>
          </section>
        </div>
      </section>

      <section class="view" id="records-view">
        <div class="view-title">
          <div><h3>Record Management</h3><p>Clean exports and evidence-ready operational records for review.</p></div>
        </div>
        <div class="record-grid">
          <div class="file-card"><strong>hotspot_priority.csv</strong><span>Ranked hotspot records with score, band, station, junction, and recommended action.</span></div>
          <div class="file-card"><strong>station_priority_summary.csv</strong><span>Station-level deployment priorities and high-priority hotspot counts.</span></div>
          <div class="file-card"><strong>enforcement_shift_plan.csv</strong><span>Top enforcement records with recommended patrol hours.</span></div>
        </div>
        <div class="section-grid">
          <section class="panel">
            <div class="panel-head"><h3>Managed Enforcement Records</h3><span>top operational rows</span></div>
            <table id="record-table"></table>
          </section>
          <section class="panel">
            <div class="panel-head"><h3>Data Governance</h3><span>prototype-ready controls</span></div>
            <div class="details">
              <div class="action">Uses anonymized vehicle identifiers and aggregated spatial bins for dashboard visibility.</div>
              <div class="action">Keeps raw evidence separate from decision outputs so enforcement officers can review before action.</div>
              <div class="action">Exports CSV records that can be searched, audited, and attached to patrol logs.</div>
            </div>
          </section>
        </div>
      </section>

      <section class="view" id="model-view">
        <div class="view-title">
          <div><h3>Model Performance</h3><p>Walk-forward validation for the hybrid ranking model used to prioritize enforcement zones.</p></div>
        </div>
        <div class="record-grid">
          <div class="file-card"><strong>{model_precision50:.1f}% Precision@50</strong><span>Predicted top-50 hotspots that also appeared in the actual next-month top-50.</span></div>
          <div class="file-card"><strong>{model_ndcg50:.2f}% NDCG@50</strong><span>Ranking quality: rewards placing the most important future hotspots near the top.</span></div>
          <div class="file-card"><strong>{model_spearman:.3f} Spearman</strong><span>Stability between historical hotspot priority and future hotspot priority.</span></div>
        </div>
        <div class="section-grid">
          <section class="panel">
            <div class="panel-head"><h3>Hybrid Model Ranked Hotspots</h3><span>history + last-month persistence model</span></div>
            <table id="model-table"></table>
          </section>
          <section class="panel">
            <div class="panel-head"><h3>Month-wise Backtest</h3><span>future month validation</span></div>
            <table id="backtest-table"></table>
          </section>
        </div>
        <div class="panel" style="margin-top:14px">
          <div class="panel-head"><h3>Learned Ranking Blend</h3><span>selected by validation</span></div>
          <div class="details">
            <div class="action">Best blend: {100 * float(model_weights.get("history", 0)):.0f}% long-term history + {100 * float(model_weights.get("last_month", 0)):.0f}% last-month signal. This shows illegal parking hotspots are persistent, but recent activity improves next-month prioritization.</div>
          </div>
        </div>
      </section>
    </main>
  </div>

  <script>
    const hotspots = {records(top_hotspots)};
    const stations = {records(station_view)};
    const shifts = {records(shift_view)};
    const hourly = {records(hourly)};
    const modelRows = {records(hybrid_view)};
    const backtestRows = {records(backtest_view)};
    const colors = {{ Critical: '#c62828', High: '#e76f00', Medium: '#e69f00', Watch: '#138a50' }};
    let bandFilter = 'All';
    let stationFilter = 'All';
    const map = document.getElementById('map');
    const minLat = Math.min(...hotspots.map(p => p.latitude));
    const maxLat = Math.max(...hotspots.map(p => p.latitude));
    const minLon = Math.min(...hotspots.map(p => p.longitude));
    const maxLon = Math.max(...hotspots.map(p => p.longitude));

    function fmt(n) {{ return Number(n).toLocaleString('en-IN'); }}
    function escapeHtml(s) {{ return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
    function rowBand(band) {{ return `<span class="band ${{band}}">${{band}}</span>`; }}

    function setSelected(p) {{
      document.getElementById('selected-rank').textContent = `Priority #${{p.priority_rank}}`;
      document.getElementById('selected-title').textContent = p.junction_name || p.grid_id;
      document.getElementById('selected-station').textContent = `${{p.police_station}} | Grid ${{p.grid_id}}`;
      document.getElementById('selected-score').textContent = Number(p.congestion_impact_score).toFixed(2);
      document.getElementById('selected-count').textContent = fmt(p.violation_count);
      document.getElementById('selected-peak').textContent = `${{p.peak_hour_share}}%`;
      document.getElementById('selected-band').textContent = p.risk_band;
      document.getElementById('selected-frequency').textContent = Number(p.parking_violation_frequency_score).toFixed(1);
      document.getElementById('selected-repeat').textContent = Number(p.repeat_hotspot_density_score).toFixed(1);
      document.getElementById('selected-junction').textContent = Number(p.junction_proximity_score).toFixed(1);
      document.getElementById('selected-vehicle').textContent = Number(p.vehicle_blockage_score).toFixed(1);
      document.getElementById('selected-action').textContent = p.recommended_action;
    }}

    function visibleHotspots() {{
      return hotspots.filter(p =>
        (bandFilter === 'All' || p.risk_band === bandFilter) &&
        (stationFilter === 'All' || p.police_station === stationFilter)
      );
    }}

    function renderMap() {{
      map.querySelectorAll('.dot').forEach(d => d.remove());
      for (const p of visibleHotspots()) {{
        const x = 5 + 90 * ((p.longitude - minLon) / Math.max(0.000001, maxLon - minLon));
        const y = 95 - 90 * ((p.latitude - minLat) / Math.max(0.000001, maxLat - minLat));
        const d = document.createElement('button');
        const size = Math.max(10, Math.min(34, 6 + p.congestion_impact_score / 3.1));
        d.className = 'dot';
        d.style.left = `${{x}}%`;
        d.style.top = `${{y}}%`;
        d.style.width = `${{size}}px`;
        d.style.height = `${{size}}px`;
        d.style.background = colors[p.risk_band] || '#138a50';
        d.title = `#${{p.priority_rank}} ${{p.police_station}} | ${{p.junction_name}}`;
        d.addEventListener('click', () => {{
          map.querySelectorAll('.dot').forEach(x => x.classList.remove('selected'));
          d.classList.add('selected');
          setSelected(p);
        }});
        map.appendChild(d);
      }}
    }}

    function renderStationsFilter() {{
      const container = document.querySelector('.filters');
      const names = [...new Set(hotspots.slice(0, 80).map(p => p.police_station))].slice(0, 8);
      for (const name of names) {{
        const b = document.createElement('button');
        b.className = 'station';
        b.dataset.station = name;
        b.textContent = name;
        b.addEventListener('click', () => {{
          stationFilter = name;
          document.querySelectorAll('.station').forEach(x => x.classList.toggle('active', x.dataset.station === name));
          renderMap();
        }});
        container.appendChild(b);
      }}
      container.querySelector('[data-station="All"]').addEventListener('click', () => {{
        stationFilter = 'All';
        document.querySelectorAll('.station').forEach(x => x.classList.toggle('active', x.dataset.station === 'All'));
        renderMap();
      }});
    }}

    function renderTable(id, cols, data, limit) {{
      const table = document.getElementById(id);
      table.innerHTML = `<thead><tr>${{cols.map(c => `<th>${{c.label}}</th>`).join('')}}</tr></thead><tbody>` +
        data.slice(0, limit).map(r => `<tr>${{cols.map(c => `<td>${{c.html ? c.html(r) : escapeHtml(r[c.key])}}</td>`).join('')}}</tr>`).join('') +
        '</tbody>';
    }}

    function renderBars() {{
      const maxImpact = Math.max(...hourly.map(h => h.impact));
      const wrap = document.getElementById('hour-bars');
      wrap.innerHTML = hourly.map(h => {{
        const width = 100 * h.impact / maxImpact;
        return `<div class="bar-row"><div>${{String(h.hour).padStart(2, '0')}}:00</div><div class="bar-track"><div class="bar-fill" style="width:${{width}}%"></div></div><div>${{fmt(h.violations)}}</div></div>`;
      }}).join('');
    }}

    document.querySelectorAll('.filter').forEach(b => b.addEventListener('click', () => {{
      bandFilter = b.dataset.band;
      document.querySelectorAll('.filter').forEach(x => x.classList.toggle('active', x === b));
      renderMap();
    }}));

    document.querySelectorAll('button.nav-item[data-view]').forEach(b => b.addEventListener('click', () => {{
      const view = b.dataset.view;
      document.querySelectorAll('button.nav-item[data-view]').forEach(x => x.classList.toggle('active', x === b));
      document.querySelectorAll('.view').forEach(x => x.classList.toggle('active', x.id === view));
      if (view === 'dashboard-view') renderMap();
    }}));

    renderStationsFilter();
    renderMap();
    setSelected(hotspots[0]);
    renderTable('hotspot-table', [
      {{ label: 'Rank', key: 'priority_rank' }},
      {{ label: 'Zone', html: r => `<strong>${{escapeHtml(r.junction_name)}}</strong><br><span style="color:#647083">${{escapeHtml(r.police_station)}}</span>` }},
      {{ label: 'Count', html: r => fmt(r.violation_count) }},
      {{ label: 'Score', html: r => Number(r.congestion_impact_score).toFixed(2) }},
      {{ label: 'Risk', html: r => rowBand(r.risk_band) }}
    ], hotspots, 12);
    renderTable('station-table', [
      {{ label: 'Station', key: 'police_station' }},
      {{ label: 'Violations', html: r => fmt(r.violation_count) }},
      {{ label: 'High Zones', key: 'high_priority_hotspots' }},
      {{ label: 'Score', html: r => Number(r.station_priority_score).toFixed(2) }}
    ], stations, 12);
    renderTable('shift-table', [
      {{ label: 'Rank', key: 'priority_rank' }},
      {{ label: 'Zone', html: r => `<strong>${{escapeHtml(r.junction_name)}}</strong><br><span style="color:#647083">${{escapeHtml(r.police_station)}}</span>` }},
      {{ label: 'Hours', key: 'best_enforcement_hours' }},
      {{ label: 'Action', key: 'recommended_action' }}
    ], shifts, 10);
    renderTable('record-table', [
      {{ label: 'Record', html: r => `HP-${{String(r.priority_rank).padStart(4, '0')}}` }},
      {{ label: 'Zone', html: r => `<strong>${{escapeHtml(r.junction_name)}}</strong><br><span style="color:#647083">${{escapeHtml(r.police_station)}}</span>` }},
      {{ label: 'Risk', html: r => rowBand(r.risk_band) }},
      {{ label: 'Score', html: r => Number(r.congestion_impact_score).toFixed(2) }},
      {{ label: 'Action', key: 'recommended_action' }}
    ], hotspots, 14);
    renderTable('model-table', [
      {{ label: 'Rank', key: 'model_rank' }},
      {{ label: 'Zone', html: r => `<strong>${{escapeHtml(r.junction_name)}}</strong><br><span style="color:#647083">${{escapeHtml(r.police_station)}}</span>` }},
      {{ label: 'Model Score', html: r => Number(r.model_priority_score).toFixed(2) }},
      {{ label: 'History', html: r => Number(r.history).toFixed(2) }},
      {{ label: 'Last Month', html: r => Number(r.last_month).toFixed(2) }}
    ], modelRows, 12);
    renderTable('backtest-table', [
      {{ label: 'Test Month', key: 'test_month' }},
      {{ label: 'P@20', html: r => `${{(100 * Number(r.precision_at_20)).toFixed(1)}}%` }},
      {{ label: 'P@50', html: r => `${{(100 * Number(r.precision_at_50)).toFixed(1)}}%` }},
      {{ label: 'P@100', html: r => `${{(100 * Number(r.precision_at_100)).toFixed(1)}}%` }},
      {{ label: 'NDCG@50', html: r => `${{(100 * Number(r.ndcg_at_50)).toFixed(1)}}%` }},
      {{ label: 'Spearman', html: r => Number(r.spearman_score_correlation).toFixed(3) }}
    ], backtestRows, 8);
    renderBars();
  </script>
</body>
</html>"""

    out_path = OUT / "parksense_command_center.html"
    out_path.write_text(html, encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
