#!/usr/bin/env python3
"""
compute_effort_metrics.py

Utilities to compute effort-based sprint metrics (throughput, volatility, plannedness,
carryover, rework, and a simple burnout/workload indicator). Designed to be imported
by forecast_release.py or run standalone.

Outputs:
 - Returns a DataFrame with added feature columns.
 - Optionally writes a features CSV.

Usage (standalone):
  python3 compute_effort_metrics.py --in sprints.csv --out features.csv --window 6

Functions:
 - compute_effort_metrics(df, window=6, min_periods=1, team_size_col=None)
 - save_features(df, out_path)
"""
from pathlib import Path
from datetime import timedelta
import json
import argparse

import pandas as pd
import numpy as np


def weekend_count(s, e):
    return sum(1 for d in pd.date_range(s, e) if d.weekday() >= 5)


def compute_effort_metrics(df,
                           window: int = 6,
                           min_periods: int = 1,
                           team_size_col: str = "team_size"):
    """
    Accepts a dataframe with at least: sprint_id, start_date, end_date, velocity, scope_added.
    velocity and scope_added are expected in the same effort unit (e.g. person_days).
    Returns a new dataframe with derived effort-based metrics.

    New columns (examples):
      - calendar_days, weekend_days, effective_days
      - net_done_pd
      - throughput_pd_daily
      - throughput_mean_w, throughput_std_w, throughput_cv_w
      - unplanned_fraction
      - prev_throughput_daily
      - carryover_ratio (if carryover_pd & committed_pd present)
      - rework_fraction (if rework_pd present)
      - workload_ratio = velocity / (team_size * effective_days) if team_size present
      - burnout_index = rolling mean of workload_ratio (window)
    """

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.get("start_date")):
        df["start_date"] = pd.to_datetime(df["start_date"])
    if not pd.api.types.is_datetime64_any_dtype(df.get("end_date")):
        df["end_date"] = pd.to_datetime(df["end_date"])

    # calendar/effective days
    df["calendar_days"] = (df["end_date"] - df["start_date"]).dt.days + 1
    df["weekend_days"] = [
        weekend_count(s, e) for s, e in zip(df["start_date"], df["end_date"])
    ]
    df["effective_days"] = (df["calendar_days"] - df["weekend_days"]).clip(lower=1)

    # net done and throughput (effort-based)
    df["net_done_pd"] = (df["velocity"] - df["scope_added"]).clip(lower=0.0)
    df["throughput_pd_daily"] = df["net_done_pd"] / df["effective_days"]

    # unplanned fraction: scope_added / velocity (guard against 0)
    df["unplanned_fraction"] = df["scope_added"] / df["velocity"].replace({0: np.nan})
    df["unplanned_fraction"] = df["unplanned_fraction"].fillna(0.0)

    # rolling stats of throughput (window)
    w = int(window)
    df["throughput_mean_w"] = df["throughput_pd_daily"].rolling(window=w, min_periods=min_periods).mean()
    df["throughput_std_w"] = df["throughput_pd_daily"].rolling(window=w, min_periods=min_periods).std().fillna(0.0)
    df["throughput_cv_w"] = (df["throughput_std_w"] / df["throughput_mean_w"]).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    # prev lag features
    df["prev_throughput_daily"] = df["throughput_pd_daily"].shift(1).fillna(df["throughput_pd_daily"].mean())

    # carryover and rework if present
    if "carryover_pd" in df.columns and "committed_pd" in df.columns:
        df["carryover_ratio"] = df["carryover_pd"] / df["committed_pd"].replace({0: np.nan})
        df["carryover_ratio"] = df["carryover_ratio"].fillna(0.0)
    if "rework_pd" in df.columns:
        df["rework_fraction"] = df["rework_pd"] / df["velocity"].replace({0: np.nan})
        df["rework_fraction"] = df["rework_fraction"].fillna(0.0)

    # workload / burnout indicators (if team_size present)
    if team_size_col in df.columns:
        # team effective capacity (person_days) in sprint
        df["_team_capacity_pd"] = df[team_size_col] * df["effective_days"]
        # workload ratio: velocity / capacity (how much of capacity was utilized)
        df["workload_ratio"] = df["velocity"].replace({0:0.0}) / df["_team_capacity_pd"].replace({0: np.nan})
        df["workload_ratio"] = df["workload_ratio"].fillna(0.0)
        # burnout index: rolling mean of workload_ratio
        df["burnout_index"] = df["workload_ratio"].rolling(window=w, min_periods=min_periods).mean().fillna(0.0)
    else:
        df["workload_ratio"] = np.nan
        df["burnout_index"] = np.nan

    # story-level metrics (if story_sizes present)
    if "story_sizes" in df.columns:
        avg_sizes = []
        std_sizes = []
        for _, row in df.iterrows():
            try:
                sizes = json.loads(row["story_sizes"]) if isinstance(row["story_sizes"], str) else []
                if sizes and len(sizes) > 0:
                    avg_sizes.append(float(np.mean(sizes)))
                    std_sizes.append(float(np.std(sizes, ddof=1)) if len(sizes) > 1 else 0.0)
                else:
                    avg_sizes.append(np.nan)
                    std_sizes.append(np.nan)
            except Exception:
                avg_sizes.append(np.nan)
                std_sizes.append(np.nan)
        df["avg_story_size"] = avg_sizes
        df["story_size_std"] = std_sizes
        df["story_size_cv"] = df["story_size_std"] / df["avg_story_size"].replace({0: np.nan})
        df["story_size_cv"] = df["story_size_cv"].fillna(0.0)
        # flag: high variance in story sizes
        df["story_variance_flag"] = df["story_size_cv"] > 0.5
    else:
        df["avg_story_size"] = np.nan
        df["story_size_std"] = np.nan
        df["story_size_cv"] = np.nan
        df["story_variance_flag"] = False

    # simple flags (useful for dashboards)
    df["volatility_flag"] = df["throughput_cv_w"] > 0.3
    df["unplanned_flag"] = df["unplanned_fraction"] > 0.2
    df["burnout_flag"] = df["burnout_index"] > 0.85  # heuristic threshold

    return df


def save_features(df, out_path: str):
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)


def _cli():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", dest="out", default="sprint_features.csv")
    p.add_argument("--window", type=int, default=6)
    p.add_argument("--team-size-col", default="team_size")
    args = p.parse_args()

    df = pd.read_csv(args.inp, parse_dates=["start_date", "end_date"])
    out = compute_effort_metrics(df, window=args.window, team_size_col=args.team_size_col)
    save_features(out, args.out)
    print(f"Wrote features to {args.out}")


if __name__ == "__main__":
    _cli()