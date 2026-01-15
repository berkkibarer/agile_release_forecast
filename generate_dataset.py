#!/usr/bin/env python3
"""
generate_dataset_minimal.py

Produce a synthetic sprint CSV that follows the minimal schema agreed:

Required output columns (always):
 - sprint_id
 - start_date (YYYY-MM-DD)
 - end_date   (YYYY-MM-DD)
 - velocity   (top-level unit; unit = --unit)
 - scope_added (top-level unit; same unit as velocity)
 - team_size
 - percent_bug

Optional additions (flags):
 - --include-committed : include committed_pd column (in the same top-level unit)
 - --include-story-sizes : include story_sizes JSON list column (optional)

Behavior / rules:
 - If --unit points: velocity/scope_added/committed_pd are in story-points.
 - If --unit person_days: velocity/scope_added/committed_pd are in person-days.
 - This generator is for synthetic data only (useful to test the forecasting pipeline).
 - Supports fixed sprint lengths in weeks via --fixed-sprint-weeks.
"""
from datetime import datetime, timedelta
import argparse
from pathlib import Path
import numpy as np
import json
import csv

SAFE_EPS = 1e-6

def weekend_count_in_range(start, end):
    days = (end - start).days + 1
    return sum(1 for i in range(days) if (start + timedelta(days=i)).weekday() >= 5)

def make_story_sizes(avg_stories=6, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    k = max(1, int(abs(int(rng.normal(avg_stories, 1.5)))))
    sizes = []
    for _ in range(k):
        s = abs(rng.normal(2.0, 1.6))
        s = max(0.25, min(8.0, s))
        sizes.append(round(float(s), 2))
    return json.dumps(sizes)

def make_dataset(n=120, start_date="2022-01-03", seed=42, out="example_sprints_minimal.csv",
                 unit="points", fixed_sprint_weeks=None, include_committed=True, include_story_sizes=False):
    rng = np.random.default_rng(seed)
    rows = []
    sd = datetime.strptime(start_date, "%Y-%m-%d").date()

    # Team size schedule: piecewise constant
    team_sizes = []
    while len(team_sizes) < n:
        block_len = int(rng.integers(6, 13))
        ts = int(rng.choice([5,6,7,8,9], p=[0.04,0.12,0.6,0.18,0.06]))
        team_sizes.extend([ts] * block_len)
    team_sizes = team_sizes[:n]

    base_daily = 11.5   # baseline team points per workday (used when unit == points)
    prev_daily = base_daily

    cur_start = sd

    for i in range(1, n+1):
        # sprint length (calendar days)
        if fixed_sprint_weeks is not None:
            sprint_len = int(7 * int(fixed_sprint_weeks))
        else:
            sprint_len = int(rng.choice([14,14,14,7,21], p=[0.65,0.15,0.15,0.03,0.02]))

        start = cur_start
        end = start + timedelta(days=sprint_len - 1)

        team_size = team_sizes[i-1]

        # weekend days + effective days (used internally)
        weekend_days = weekend_count_in_range(start, end)
        effective_days = max(1, sprint_len - weekend_days)  # no holidays in minimal output

        # percent_bug feature (0..1)
        bug_base = 0.085 + 0.02 * np.cos(2 * np.pi * (i / 26.0))
        percent_bug = float(np.clip(bug_base + rng.normal(0, 0.03), 0.01, 0.30))

        # scope_added in points (simulated)
        if rng.random() < 0.06:
            scope_added_points = int(rng.integers(4, 10))
        else:
            scope_added_points = int(rng.poisson(1.1))

        # simulate team daily productivity (points per team per workday)
        season = 1.6 * np.sin(2 * np.pi * (i / 26.0))
        trend = 0.005 * i
        daily_target = base_daily + (team_size - 7) * 1.1 + season + trend
        ar_rho = 0.45
        daily_prod = ar_rho * prev_daily + (1 - ar_rho) * daily_target + rng.normal(0, 1.8)
        if daily_prod <= 0:
            daily_prod = max(0.1, daily_prod + 2.0)

        # penalties and raw done in points
        bug_penalty_per_day = 6.0 * percent_bug
        raw_done_points = daily_prod * effective_days - bug_penalty_per_day * effective_days - 1.9 * scope_added_points
        raw_done_points += rng.normal(0, max(1.0, 0.08 * abs(raw_done_points)))
        net_done_points = float(max(0.0, round(raw_done_points, 1)))
        velocity_points = float(max(scope_added_points, round(net_done_points + scope_added_points, 1)))

        # committed (in the top-level unit)
        if unit == "points":
            # committed in points = expected team points production * utilization
            committed_top = round(daily_prod * effective_days * float(rng.uniform(0.8, 1.0)), 2)
        else:
            # committed in person-days = team capacity * utilization
            committed_top = round(team_size * effective_days * float(rng.uniform(0.8, 1.0)), 2)

        # If output unit is points -> write points as top-level and do NOT write person-days
        if unit == "points":
            velocity_out = velocity_points
            scope_out = int(scope_added_points)
            committed_out = committed_top
        else:
            # derive person-days from simulated team productivity (points -> person-days conversion)
            points_per_person_per_day = daily_prod / max(1, team_size)
            if points_per_person_per_day <= SAFE_EPS:
                velocity_out = 0.0
                scope_out = 0.0
                committed_out = 0.0
            else:
                velocity_out = float(round(velocity_points / points_per_person_per_day, 3))
                scope_out = float(round(scope_added_points / points_per_person_per_day, 3))
                # committed_top in person-days computed above already when unit == person_days
                committed_out = committed_top if unit == "person_days" else float(round(committed_top / points_per_person_per_day, 3))

        row = {
            "sprint_id": f"S{i:03d}",
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "velocity": velocity_out,
            "scope_added": scope_out,
            "team_size": int(team_size),
            "percent_bug": round(percent_bug, 3)
        }

        if include_committed:
            # committed_out is in the same unit as velocity/scope (top-level unit)
            row["committed_pd"] = committed_out

        if include_story_sizes:
            row["story_sizes"] = make_story_sizes(rng=rng)

        rows.append(row)

        # advance
        cur_start = end + timedelta(days=1)
        prev_daily = daily_prod

    # write CSV
    # preserve column order: minimal required + optional fields in a deterministic order
    keys = ["sprint_id","start_date","end_date","velocity","scope_added","team_size","percent_bug"]
    if include_committed:
        keys.append("committed_pd")
    if include_story_sizes:
        keys.append("story_sizes")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"Wrote {len(rows)} rows to {out} (unit={unit}, fixed_sprint_weeks={fixed_sprint_weeks}, committed_included={include_committed}, story_sizes={include_story_sizes})")
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="example_sprints_minimal.csv")
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--start", default="2022-01-03")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--unit", choices=["points", "person_days"], default="points",
                   help="Top-level unit for velocity/scope_added")
    p.add_argument("--fixed-sprint-weeks", type=int, default=None,
                   help="If provided, each sprint will be exactly 7 * N calendar days")
    p.add_argument("--no-committed", dest="include_committed", action="store_false",
                   help="Do not include committed_pd column (default: include)")
    p.add_argument("--include-story-sizes", dest="include_story_sizes", action="store_true",
                   help="Include story_sizes JSON list column (optional)")
    args = p.parse_args()

    make_dataset(n=args.n, start_date=args.start, seed=args.seed, out=args.out,
                 unit=args.unit, fixed_sprint_weeks=args.fixed_sprint_weeks,
                 include_committed=args.include_committed, include_story_sizes=args.include_story_sizes)

if __name__ == "__main__":
    main()