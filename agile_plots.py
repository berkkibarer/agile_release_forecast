#!/usr/bin/env python3
"""
agile_plots.py

Plot helper functions to create common Agile visuals from a sprint-level dataframe
that includes derived effort metrics (usually produced by compute_effort_metrics.py).

Produces and saves PNGs:
 - burndown (cumulative remaining effort over time)
 - burnup (cumulative done vs target)
 - throughput trend (throughput_pd_daily over sprints)
 - burnout / workload plot (workload_ratio, burnout_index)

This file is defensive about matplotlib styles: it prefers seaborn-darkgrid if available,
falls back to other built-in styles, and never raises an import-time error if a style is missing.
It avoids the pandas FutureWarning by using the pandas-to-numpy conversion API (.to_numpy())
instead of calling .dt.to_pydatetime() directly.
"""
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

try:
    available = plt.style.available
    preferred_styles = [
        "seaborn-darkgrid",
        "seaborn",
        "seaborn-whitegrid",
        "ggplot",
        "tableau-colorblind10",
        "default",
    ]
    chosen = next((s for s in preferred_styles if s in available), None)
    if chosen:
        plt.style.use(chosen)
except Exception:
    pass

plt.rcParams.update({"figure.autolayout": True})

def _dates_from_series(dt_series):
    """
    Return a numpy array of datetime64 objects derived from a pandas DatetimeIndex/Series.
    Using dt_series.to_numpy() avoids the pandas FutureWarning while providing a NumPy
    array that matplotlib understands.
    """
    return dt_series.to_numpy()

def plot_burndown(df, total_release_effort, out_path):
    df = df.sort_values("start_date").reset_index(drop=True)
    cum_done = df["net_done_pd"].cumsum()
    remaining = total_release_effort - cum_done
    dates = _dates_from_series(df["end_date"])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, remaining, marker="o", label="Remaining effort (person-days)")
    ax.fill_between(dates, remaining, alpha=0.1)
    ax.set_ylabel("Remaining effort (person-days)")
    ax.set_title("Burndown: Remaining Effort Over Time")
    ax.grid(True)
    ax.legend()
    fig.autofmt_xdate()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_burnup(df, total_release_effort, out_path):
    df = df.sort_values("start_date").reset_index(drop=True)
    cum_done = df["net_done_pd"].cumsum()
    dates = _dates_from_series(df["end_date"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, cum_done, marker="o", label="Cumulative done")
    ax.axhline(total_release_effort, color="red", linestyle="--", label="Release target")
    ax.set_ylabel("Cumulative done (person-days)")
    ax.set_title("Burnup Chart")
    ax.grid(True)
    ax.legend()
    fig.autofmt_xdate()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_throughput_trend(df, out_path):
    df = df.sort_values("start_date").reset_index(drop=True)
    dates = _dates_from_series(df["end_date"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, df.get("throughput_pd_daily", pd.Series([np.nan]*len(df))), marker="o", label="throughput (person-days/day)")
    ax.plot(dates, df.get("throughput_mean_w", pd.Series([np.nan]*len(df))), linestyle="--", label="rolling mean")
    lower = df.get("throughput_mean_w", 0) - df.get("throughput_std_w", 0)
    upper = df.get("throughput_mean_w", 0) + df.get("throughput_std_w", 0)
    ax.fill_between(dates, lower, upper, alpha=0.12, label=" ±1σ")
    ax.set_ylabel("Throughput (person-days/day)")
    ax.set_title("Throughput Trend and Volatility")
    ax.grid(True)
    ax.legend()
    fig.autofmt_xdate()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_burnout(df, out_path):
    df = df.sort_values("start_date").reset_index(drop=True)
    if "workload_ratio" not in df.columns:
        return
    dates = _dates_from_series(df["end_date"])
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(dates, df["workload_ratio"], marker="o", label="workload ratio (velocity / team capacity)")
    ax.plot(dates, df["burnout_index"], linestyle="--", label="burnout index (rolling avg)")
    ax.axhline(1.0, color="red", linestyle=":", label="100% capacity")
    ax.set_ylabel("Ratio")
    ax.set_title("Workload / Burnout Indicator")
    ax.grid(True)
    ax.legend()
    fig.autofmt_xdate()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _cli():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out-dir", dest="out_dir", default="plots")
    p.add_argument("--total-release", type=float, required=True)
    args = p.parse_args()

    df = pd.read_csv(args.inp, parse_dates=["start_date", "end_date"])
    outdir = Path(args.out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    plot_burndown(df, args.total_release, outdir / "burndown.png")
    plot_burnup(df, args.total_release, outdir / "burnup.png")
    plot_throughput_trend(df, outdir / "throughput_trend.png")
    if "team_size" in df.columns:
        plot_burnout(df, outdir / "burnout.png")

    print(f"Wrote plots to {outdir}")


if __name__ == "__main__":
    _cli()