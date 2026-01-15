#!/usr/bin/env python3
"""
forecast_release.py — integrated version (updated)

Changes in this patch:
- Default use_features includes throughput_cv_w and unplanned_fraction.
- If CSV contains committed_pd, the script derives carryover_pd and carryover_ratio.
- Suppresses the small-sample scipy kurtosis warning only when printing model.summary().
- Other functionality remains: compute_effort_metrics -> optional plots -> OLS/fallback ->
  MVN parameter sampling + residual sampling -> sequential MC sims -> artifacts/history.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
import math
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm

from compute_effort_metrics import compute_effort_metrics, save_features
from agile_plots import plot_burndown, plot_burnup, plot_throughput_trend, plot_burnout

SAFE_EPS = 1e-6
DEFAULT_N_SIMS = 5000
MIN_RECENT_WINDOW = 3
MAX_RECENT_WINDOW = 6

def parse_config(path):
    j = json.loads(Path(path).read_text())
    required = ["csv_path", "total_release_effort", "effort_unit", "save_model_path", "history_path"]
    for k in required:
        if k not in j:
            raise KeyError(f"Config missing required key: {k}")
    # expand defaults: include volatility and plannedness features by default
    j.setdefault("use_features", ["scope_added", "percent_bug", "prev_daily_rate", "throughput_cv_w", "unplanned_fraction"])
    j.setdefault("future_holiday_dates", [])
    j.setdefault("n_sims", None)
    j.setdefault("recent_sprint_window", None)
    j.setdefault("avg_point_effort", None)
    j.setdefault("compute_effort_metrics", {"enabled": True, "window": 6, "team_size_col": "team_size"})
    j.setdefault("generate_plots", True)
    j.setdefault("plots_output_dir", "plots")
    return j

def weekend_count_in_range(start, end):
    days = (end - start).days + 1
    return sum(1 for i in range(days) if (start + timedelta(days=i)).weekday() >= 5)

def add_workdays(start_date, workdays, holiday_dates_set=None):
    if holiday_dates_set is None:
        holiday_dates_set = set()
    cur = start_date
    counted = 0
    while True:
        is_weekend = cur.weekday() >= 5
        is_holiday = cur.strftime("%Y-%m-%d") in holiday_dates_set
        if (not is_weekend) and (not is_holiday):
            counted += 1
            if counted >= workdays:
                return cur
        cur = cur + timedelta(days=1)

def build_features_matrix(df, feature_names):
    X = []
    for _, row in df.iterrows():
        x = []
        for fn in feature_names:
            if fn.lower() in ("const","intercept"):
                x.append(1.0)
            else:
                x.append(float(row.get(fn, 0.0)))
        X.append(x)
    X = np.vstack(X)
    return X

def safe_mvnormal_sample(mean, cov, rng):
    mean = np.asarray(mean, float)
    if cov is None:
        return mean.copy()
    cov = np.asarray(cov, float)
    n = mean.size
    if cov.shape != (n, n):
        return mean.copy()
    cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
    eps = 1e-8
    for _ in range(6):
        try:
            L = np.linalg.cholesky(cov + eps * np.eye(n))
            return mean + L.dot(rng.normal(size=n))
        except np.linalg.LinAlgError:
            eps *= 10.0
    vals, vecs = np.linalg.eigh(cov)
    vals_clipped = np.clip(vals, 0.0, None)
    cov2 = (vecs * vals_clipped) @ vecs.T
    try:
        L = np.linalg.cholesky(cov2 + 1e-12 * np.eye(n))
        return mean + L.dot(rng.normal(size=n))
    except Exception:
        return mean.copy()

def compute_covariance(model):
    try:
        cov = model.cov_params().values.astype(float)
        return cov
    except Exception:
        pass
    try:
        robust = model.get_robustcov_results(cov_type="HC3")
        cov = robust.cov_params().values.astype(float)
        return cov
    except Exception:
        pass
    try:
        bse = np.asarray(model.bse.astype(float))
        cov = np.diag((bse ** 2))
        return cov
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--n-sims", type=int, default=None)
    p.add_argument("--recent-sprint-window", type=int, default=None)
    args = p.parse_args()

    cfg = parse_config(args.config)
    if args.n_sims is not None:
        cfg["n_sims"] = int(args.n_sims)
    if args.recent_sprint_window is not None:
        cfg["recent_sprint_window"] = int(args.recent_sprint_window)

    csv_path = cfg["csv_path"]
    df = pd.read_csv(csv_path, parse_dates=["start_date","end_date"]).sort_values("start_date").reset_index(drop=True)

    # Compute effort metrics if enabled
    cem_cfg = cfg.get("compute_effort_metrics", {})
    if cem_cfg.get("enabled", True):
        window = int(cem_cfg.get("window", 6))
        team_size_col = cem_cfg.get("team_size_col", "team_size")
        df = compute_effort_metrics(df, window=window, team_size_col=team_size_col)
        features_out = Path(cfg.get("history_path", "history/model_history.jsonl")).parent / "sprint_features.csv"
        df.to_csv(features_out, index=False)
        print(f"Wrote computed effort metrics to {features_out}")

    # Net done and daily_rate (only create if not already present)
    if "net_done" not in df.columns:
        df["net_done"] = (df["velocity"] - df["scope_added"]).clip(lower=0.0)
    # daily_rate (recompute/overwrite is fine)
    df["daily_rate"] = df["net_done"] / df["effective_days"]

    # If committed_pd present, derive carryover_pd and carryover_ratio
    if "committed_pd" in df.columns:
        df["carryover_pd"] = (df["committed_pd"] - df["net_done"]).clip(lower=0.0)
        df["carryover_ratio"] = df["carryover_pd"] / df["committed_pd"].replace({0: np.nan})
        df["carryover_ratio"] = df["carryover_ratio"].fillna(0.0)
        print("Derived carryover_pd and carryover_ratio from committed_pd and net_done.")

    required_cols = ["sprint_id","start_date","end_date","velocity","scope_added"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"CSV missing required columns: {missing}")

    # calendar/effective days (compute if missing)
    if "calendar_days" not in df.columns:
        calendar_days_list = []
        weekend_days_list = []
        effective_days_list = []
        for _, row in df.iterrows():
            start = pd.to_datetime(row["start_date"]).date()
            end = pd.to_datetime(row["end_date"]).date()
            calendar_days = (end - start).days + 1
            weekend_days = weekend_count_in_range(start, end)
            holidays = 0
            effective_days = max(1, calendar_days - weekend_days - holidays)
            calendar_days_list.append(calendar_days)
            weekend_days_list.append(weekend_days)
            effective_days_list.append(effective_days)
        df["calendar_days"] = calendar_days_list
        df["weekend_days"] = weekend_days_list
        df["effective_days"] = effective_days_list

    # Net done and daily_rate (only create if not already present)
    if "net_done" not in df.columns:
        df["net_done"] = (df["velocity"] - df["scope_added"]).clip(lower=0.0)
    # daily_rate (recompute/overwrite is fine)
    df["daily_rate"] = df["net_done"] / df["effective_days"]

    # prev_daily_rate mapping/backfill
    use_features = cfg.get("use_features", ["scope_added","percent_bug","prev_daily_rate","throughput_cv_w","unplanned_fraction"])
    use_features = ["prev_daily_rate" if f=="prev_velocity" else f for f in use_features]
    if "prev_daily_rate" not in df.columns:
        df["prev_daily_rate"] = df["daily_rate"].shift(1).fillna(df["daily_rate"].mean())

    cumulative_done = float(df["net_done"].sum())
    cumulative_effective_days = float(df["effective_days"].sum())
    print("Historical summary:")
    print("  rows:", len(df), " cumulative_net_done:", cumulative_done, " cumulative_effective_days:", cumulative_effective_days)

    effort_unit = cfg["effort_unit"]
    total_release_raw = float(cfg["total_release_effort"])
    print(f"Config declares total_release_effort = {total_release_raw} (unit: {effort_unit})")

    # Optional plots
    if cfg.get("generate_plots", True):
        plots_dir = Path(cfg.get("plots_output_dir", "plots"))
        plots_dir.mkdir(parents=True, exist_ok=True)
        try:
            plot_burndown(df, total_release_raw, plots_dir / "burndown.png")
            plot_burnup(df, total_release_raw, plots_dir / "burnup.png")
            plot_throughput_trend(df, plots_dir / "throughput_trend.png")
            if "team_size" in df.columns:
                plot_burnout(df, plots_dir / "burnout.png")
            print(f"Wrote plots to {plots_dir}")
        except Exception as e:
            print("Warning: failed to generate plots:", e)

    # compute remaining in configured unit
    if effort_unit == "points":
        remaining_amount = float(max(0.0, total_release_raw - cumulative_done))
    else:
        remaining_amount = float(max(0.0, total_release_raw - cumulative_done))

    n_sims = int(cfg.get("n_sims") or DEFAULT_N_SIMS)
    print(f"Running {n_sims} MC simulations.")

    if cfg.get("recent_sprint_window") is None:
        computed = min(MAX_RECENT_WINDOW, max(MIN_RECENT_WINDOW, max(1, len(df)//2)))
        recent_k = computed
        print(f"Computed recent_sprint_window = {recent_k} (len(history)={len(df)})")
    else:
        recent_k = int(cfg.get("recent_sprint_window"))
        print(f"Using recent_sprint_window from config: {recent_k}")

    # prepare regression target and design
    y = df["daily_rate"].astype(float).values
    X = build_features_matrix(df, use_features)
    X = sm.add_constant(X)
    feature_names_with_const = ["const"] + use_features

    # LOW-DATA FALLBACK
    low_data_mode = len(df) < 3
    if low_data_mode:
        print("WARNING: Low-data fallback engaged (len(history) < 3). Skipping OLS fit.")
        observed_rate = float(df["daily_rate"].iloc[-1]) if len(df) >= 1 else 0.0
        if observed_rate <= SAFE_EPS:
            observed_rate = max(observed_rate, 0.1)
            print("WARNING: observed daily_rate very small; using fallback observed_rate =", observed_rate)
        beta_hat = np.zeros(len(feature_names_with_const), dtype=float)
        beta_hat[0] = observed_rate
        cov = None
        sigma = max(0.5 * observed_rate, 0.5)
    else:
        model = sm.OLS(y, X).fit()
        resid = model.resid
        sigma = float(np.std(resid, ddof=1))
        beta_hat = model.params.astype(float)
        cov = compute_covariance(model)
        if cov is None:
            print("WARNING: Could not compute parameter covariance; MC parameter uncertainty limited.")
        else:
            print("Parameter covariance matrix computed for MC sampling.")
        # suppress the small-sample kurtosistest warning only around summary print
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="kurtosistest only valid for n>=20.*")
            print(model.summary())

    # Save artifact
    artifact = {
        "param_names": feature_names_with_const,
        "beta": list(beta_hat),
        "cov": (cov.tolist() if cov is not None else None),
        "sigma": sigma,
        "trained_on_rows": int(len(df)),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "effort_unit": effort_unit,
        "low_data_mode": bool(low_data_mode),
        "remaining_amount_in_unit": float(remaining_amount),
        "total_release_effort": float(total_release_raw)
    }
    Path(cfg["save_model_path"]).write_text(json.dumps(artifact, indent=2))
    print(f"Saved model artifact to {cfg['save_model_path']}")

    # prepare future sprint pattern
    K = recent_k
    recent_calendar_days = list(df["calendar_days"].tail(K).astype(int).values)
    if len(recent_calendar_days) == 0:
        recent_calendar_days = [14]
    median_effective_days = int(max(1, np.median(df["effective_days"].tail(K).values)))

    last_end = pd.to_datetime(df["end_date"].max()).date()
    next_sprint_start = last_end + timedelta(days=1)
    holiday_dates_set = set(cfg.get("future_holiday_dates", []))

    rng = np.random.default_rng(12345)
    sims = []
    hist_rows = df[use_features].to_dict("records")

    for i in range(n_sims):
        beta_star = safe_mvnormal_sample(beta_hat, cov, rng) if cov is not None else beta_hat.copy()
        remaining = float(remaining_amount)
        sprint_count = 0
        cur_start = next_sprint_start
        total_effective_days_consumed = 0.0
        finish_calendar_date = None

        while remaining > SAFE_EPS and sprint_count < 200:
            pattern_idx = sprint_count % len(recent_calendar_days)
            calendar_days_j = int(recent_calendar_days[pattern_idx])
            weekend_days_j = weekend_count_in_range(cur_start, cur_start + timedelta(days=calendar_days_j - 1))
            holidays_j = sum(1 for d in range(calendar_days_j) if (cur_start + timedelta(days=d)).strftime("%Y-%m-%d") in holiday_dates_set)
            effective_days_j = max(1, calendar_days_j - weekend_days_j - holidays_j)

            if len(hist_rows) > 0:
                rec = hist_rows[rng.integers(len(hist_rows))]
            else:
                rec = {f: 0.0 for f in use_features}

            x_vec = np.array([1.0] + [float(rec.get(f,0.0)) for f in use_features], float)
            eps = float(rng.normal(0, sigma)) if sigma and sigma > 0 else 0.0
            sampled_rate_j = float(np.dot(x_vec, beta_star) + eps)
            sampled_rate_j = max(SAFE_EPS, sampled_rate_j)
            capacity_j = sampled_rate_j * effective_days_j

            if capacity_j >= remaining:
                days_needed_in_sprint = remaining / sampled_rate_j
                total_effective_days_consumed += days_needed_in_sprint
                finish_calendar_date = add_workdays(cur_start, int(math.ceil(total_effective_days_consumed)), holiday_dates_set)
                sprint_count += 1
                remaining = 0.0
                break
            else:
                remaining -= capacity_j
                total_effective_days_consumed += effective_days_j
                cur_start = cur_start + timedelta(days=calendar_days_j)
                sprint_count += 1

        if finish_calendar_date is None:
            finish_calendar_date = cur_start

        sims.append({
            "sim_id": i,
            "days_needed": total_effective_days_consumed,
            "sprints_needed": sprint_count,
            "finish_date": finish_calendar_date.strftime("%Y-%m-%d")
        })

    sims_df = pd.DataFrame(sims)
    sims_out = cfg.get("output_paths", {}).get("sims_csv", "sims_output_regression.csv")
    sims_df.to_csv(sims_out, index=False)
    print(f"Saved {len(sims_df)} sims to {sims_out}")

    # Summaries
    med_days = float(np.median(sims_df["days_needed"]))
    p90_days = float(np.percentile(sims_df['days_needed'], 90))
    med_sprints = int(np.median(sims_df["sprints_needed"]))
    p90_sprints = int(np.percentile(sims_df['sprints_needed'], 90))
    med_finish = sims_df["finish_date"].mode().iloc[0] if len(sims_df) > 0 else ""
    p90_finish = sims_df.loc[sims_df['days_needed'] >= np.percentile(sims_df['days_needed'], 90), 'finish_date']
    p90_finish = p90_finish.iloc[0] if len(p90_finish) > 0 else med_finish

    print("\n=== FORECAST SUMMARY (sequential MC) ===")
    print("Data:", csv_path)
    print("Features used:", use_features)
    if effort_unit == "points":
        print("Remaining (points):", remaining_amount)
        if cfg.get("avg_point_effort") is not None:
            print("NOTE: avg_point_effort provided but NOT used for scheduling.")
    else:
        print("Remaining (person-days):", remaining_amount)
    print()
    print(f"Median estimate: {math.ceil(med_days/median_effective_days)} sprints (~{int(round(med_days))} days) → target date ≈ {med_finish}")
    print(f"90%-ile (conservative): {p90_sprints} sprints (~{int(round(p90_days))} days) → by ≈ {p90_finish}")
    print()
    print("Percentile summary (days):")
    for p in [10,25,50,75,90,95]:
        print(f" p{p}: {np.percentile(sims_df['days_needed'], p):.1f}", end=",")
    print("\n")
    print("Actionable recommendation:")
    print(" - Use conservative p90 for planning.")
    print(" - Retrain model each sprint; keep total_release_effort up-to-date (in the declared unit).")
    print(" - If you require person-day scheduling, provide total_release_effort in person-days (effort_unit='person_days').")
    print("===========================================")

    # Append to history
    history_path = Path(cfg["history_path"])
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "type": "forecast-sequential",
        "features": use_features,
        "trained_on_rows": len(df),
        "rsquared": (model.rsquared if not low_data_mode else None),
        "resid_std": sigma,
        "median_days": med_days,
        "p90_days": p90_days,
        "median_sprints": med_sprints,
        "p90_sprints": p90_sprints,
        "remaining_amount_in_unit": remaining_amount,
        "remaining_unit": "points" if effort_unit == "points" else "person_days",
        "config_snapshot": {k: v for k, v in cfg.items() if k in ("csv_path","save_model_path","effort_unit","use_features")}
    }
    with open(history_path, "a") as fh:
        fh.write(json.dumps(history_entry) + "\n")

if __name__ == "__main__":
    main()