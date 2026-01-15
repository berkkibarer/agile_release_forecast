#!/usr/bin/env python3
"""
generate_report.py

Generate a readable HTML report for the forecasting pipeline (cleaned / simplified).

Changes vs earlier:
 - Show all percentiles 0..100 (per-user request) for both days and sprints.
 - For each percentile also show a representative finish_date (from sims).
 - Provide sprint-based counts (sprints_needed percentiles) not just days.
 - Deduplicate plot list and embed directly.
 - Provide direct links to raw CSV/JSON artifacts and a toggleable full history table.
 - Cleaner automated commentary and flags area; keep all numeric summaries visible.

Run:
  python3 generate_report.py \
    --history history/sprint_features.csv \
    --sims sims_output_regression.csv \
    --artifact model_artifact.json \
    --plots-dir plots \
    --out report.html
"""
from pathlib import Path
import argparse
import pandas as pd
import json
import numpy as np
from datetime import datetime
import html

HTML_HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Forecast Report</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; margin: 20px; color:#222 }
h1,h2,h3 { color: #1f4e79 }
.section { margin-bottom: 22px; }
.kv { display:flex; gap:12px; flex-wrap:wrap; }
.kv div { background:#f3f7fb; padding:8px 12px; border-radius:6px; box-shadow:0 1px 0 rgba(0,0,0,0.03) }
table { border-collapse: collapse; width:100%; margin-top:8px; font-size:0.9em }
th, td { border: 1px solid #ddd; padding: 6px 8px; text-align: left; vertical-align: top }
th { background:#f6f8fa; font-weight:700 }
.small { color:#666; font-size:0.9em }
.img-row { display:flex; flex-wrap:wrap; gap:16px }
.img-card { width: 420px; border:1px solid #e6eef6; padding:8px; border-radius:6px; background:#fff }
.img-card img { max-width:100%; height:auto; display:block; margin-bottom:6px }
.flag { display:inline-block; padding:6px 10px; border-radius:6px; color:white; font-weight:600 }
.flag.warn { background:#d9534f } .flag.ok { background:#5cb85c }
.note { background:#fff8dc; padding:8px; border-radius:6px; border:1px solid #f0e6b6; margin-bottom:8px }
.footer { margin-top:28px; color:#666; font-size:0.9em }
.code { background:#fafafa; padding:8px; border-radius:6px; border:1px solid #eee; font-family:monospace; white-space:pre-wrap }
.toggle { cursor:pointer; color:#1a73e8; text-decoration:underline; font-size:0.95em }
.percentile-table { max-height: 360px; overflow:auto; display:block }
</style>
<script>
function toggleFullHistory() {
  var el = document.getElementById('full-history');
  var link = document.getElementById('toggle-link');
  if (!el) return;
  if (el.style.display === 'none') {
    el.style.display = 'block';
    link.textContent = 'Hide full history table';
  } else {
    el.style.display = 'none';
    link.textContent = 'Show full history table';
  }
}
</script>
</head><body>
"""

HTML_TAIL = """
<div class="footer">
Generated on {ts} by generate_report.py — artifacts: <a href="{history}">history</a>, <a href="{sims}">sims</a>, <a href="{artifact}">artifact</a>.
</div>
</body></html>
"""

def safe_read_csv(p):
    p = Path(p)
    if not p.exists():
        return None
    try:
        return pd.read_csv(p, parse_dates=["start_date","end_date"])
    except Exception:
        return pd.read_csv(p)

def fmt_num(x, nd=2):
    try:
        if pd.isna(x):
            return ""
        if isinstance(x, (int, np.integer)):
            return str(int(x))
        x = float(x)
        if abs(x) >= 1000 or (abs(x) < 0.01 and x != 0):
            return f"{x:.3g}"
        return f"{x:.{nd}f}"
    except Exception:
        return str(x)

def percentile_rows_from_sims(sims_df):
    """
    Produce percentile table rows for percentiles 0..100:
      percentile, days_needed, sprints_needed, finish_date (representative)
    """
    if sims_df is None or sims_df.shape[0] == 0:
        return []
    sims = sims_df.copy().reset_index(drop=True)
    # sort by days_needed
    sims_sorted = sims.sort_values("days_needed").reset_index(drop=True)
    n = len(sims_sorted)
    rows = []
    days_arr = sims_sorted["days_needed"].values
    sprints_arr = sims_sorted["sprints_needed"].values
    for p in range(0, 101):
        # percentile days
        day_val = float(np.percentile(days_arr, p))
        sprint_val = float(np.percentile(sprints_arr, p))
        # representative finish_date: choose the sim at floor(p/100*(n-1))
        idx = int(np.floor(p/100 * (n-1)))
        idx = max(0, min(n-1, idx))
        rep_date = sims_sorted.iloc[idx].get("finish_date", "")
        rows.append({
            "percentile": p,
            "days": day_val,
            "sprints": int(np.ceil(sprint_val)),
            "finish_date": rep_date
        })
    return rows

def build_commentary(df, artifact, sims_df, remaining_amount, effort_unit):
    comments = []
    flags = []
    if df is not None and not df.empty:
        if "percent_bug" in df.columns:
            avg = float(df["percent_bug"].mean())
            if avg >= 0.10:
                comments.append(f"Average percent_bug is high ({avg:.2f}). Expect rework to reduce throughput.")
                flags.append(("percent_bug","warn"))
            else:
                flags.append(("percent_bug","ok"))
        if "throughput_cv_w" in df.columns:
            cv = float(df["throughput_cv_w"].tail(6).mean())
            if cv >= 0.4:
                comments.append(f"Throughput volatility (rolling CV) is elevated (~{cv:.2f}). Forecast uncertainty increases.")
                flags.append(("throughput_cv_w","warn"))
            else:
                flags.append(("throughput_cv_w","ok"))
        if "carryover_ratio" in df.columns:
            recent = float(df["carryover_ratio"].tail(6).mean())
            if recent >= 0.20:
                comments.append(f"Recent carryover_ratio ≈ {recent:.2f} — execution issues may reduce throughput.")
                flags.append(("carryover_ratio","warn"))
            else:
                flags.append(("carryover_ratio","ok"))
        if "unplanned_fraction" in df.columns:
            recent = float(df["unplanned_fraction"].tail(6).mean())
            if recent >= 0.15:
                comments.append(f"Recent unplanned_fraction ≈ {recent:.2f} — frequent scope changes observed.")
                flags.append(("unplanned_fraction","warn"))
            else:
                flags.append(("unplanned_fraction","ok"))
    if artifact:
        beta = artifact.get("beta", [])
        pnames = artifact.get("param_names", [])
        if beta and pnames and len(beta) == len(pnames):
            pairs = list(zip(pnames, beta))
            pairs_sorted = sorted(pairs, key=lambda p: p[1])
            neg = pairs_sorted[:3]
            pos = pairs_sorted[-3:]
            comments.append("Model point estimates - top negative contributors: " +
                            ", ".join([f"{k}({v:.2f})" for k,v in neg]))
            comments.append("Model point estimates - top positive contributors: " +
                            ", ".join([f"{k}({v:.2f})" for k,v in pos]))
    if remaining_amount is not None:
        if remaining_amount <= 0:
            comments.insert(0, "No remaining work detected — release target met or unit mismatch.")
            flags.append(("remaining","warn"))
        else:
            comments.append(f"Remaining = {remaining_amount:.2f} {effort_unit}. Use conservative percentiles (e.g. p90) for planning.")
            flags.append(("remaining","ok"))
    if not comments:
        comments.append("No automated flags. Review plots and distributions for anomalies.")
    return comments, flags

def render_html(out_path, context):
    html_parts = [HTML_HEAD]
    html_parts.append(f"<h1>Release Forecast Report</h1><div class='small'>Generated: {context['generated_at']}</div>")

    html_parts.append("<div class='section' style='background:#f0f8ff;padding:16px;border-radius:8px;border:2px solid #1f4e79'>")
    html_parts.append("<h2 style='margin-top:0'>🎯 Forecast Summary (Critical Planning Information)</h2>")
    
    remaining = context.get('remaining_amount')
    effort_unit = context.get('effort_unit', 'person_days')
    percentile_rows = context.get('percentile_rows', [])
    
    if remaining is not None and remaining > 0:
        html_parts.append(f"<div style='font-size:1.1em;margin-bottom:12px'><strong>📊 Remaining Work:</strong> {remaining:.1f} {effort_unit}</div>")
    
    if percentile_rows:
        p50_row = next((r for r in percentile_rows if r['percentile'] == 50), None)
        p90_row = next((r for r in percentile_rows if r['percentile'] == 90), None)
        
        if p50_row:
            html_parts.append(f"<div style='margin:12px 0;padding:12px;background:white;border-radius:6px'>")
            html_parts.append(f"<div style='font-size:1.1em;color:#2e7d32'><strong>✓ Median Estimate (50% confidence):</strong></div>")
            html_parts.append(f"<div style='margin-left:20px;margin-top:8px'>")
            html_parts.append(f"<div>• <strong>{p50_row['sprints']} sprints</strong> (~{p50_row['days']:.1f} effective days)</div>")
            html_parts.append(f"<div>• Target date: <strong>{p50_row['finish_date']}</strong></div>")
            html_parts.append("</div></div>")
        
        if p90_row:
            html_parts.append(f"<div style='margin:12px 0;padding:12px;background:white;border-radius:6px'>")
            html_parts.append(f"<div style='font-size:1.1em;color:#d9534f'><strong>⚠️ Conservative Estimate (90% confidence - RECOMMENDED):</strong></div>")
            html_parts.append(f"<div style='margin-left:20px;margin-top:8px'>")
            html_parts.append(f"<div>• <strong>{p90_row['sprints']} sprints</strong> (~{p90_row['days']:.1f} effective days)</div>")
            html_parts.append(f"<div>• Target date: <strong>{p90_row['finish_date']}</strong></div>")
            html_parts.append("</div></div>")
        
        html_parts.append("<div style='margin-top:12px;padding:10px;background:#fff8dc;border-radius:6px'>")
        html_parts.append("<strong>📈 Percentile Range (effective days):</strong><br/>")
        p_vals = [10, 25, 50, 75, 90, 95]
        p_strs = []
        for pv in p_vals:
            pr = next((r for r in percentile_rows if r['percentile'] == pv), None)
            if pr:
                p_strs.append(f"p{pv}: {pr['days']:.1f}")
        html_parts.append(" | ".join(p_strs))
        html_parts.append("</div>")
    
    html_parts.append("<div style='margin-top:16px;padding:10px;background:#e8f5e9;border-radius:6px'>")
    html_parts.append("<strong>💡 Recommendation:</strong> Use the <strong>p90 (conservative)</strong> estimate for sprint planning and stakeholder commitments.")
    html_parts.append("</div>")
    html_parts.append("</div>")

    # Top KPIs
    html_parts.append("<div class='section'><h2>Historical Data Summary</h2>")
    html_parts.append("<div class='kv'>")
    for k,v in context["kpis"].items():
        html_parts.append(f"<div><strong>{html.escape(k)}</strong><div class='small'>{html.escape(str(v))}</div></div>")
    html_parts.append("</div></div>")

    html_parts.append("<div class='section'><h2>Automated checks</h2>")
    for name, status in context["flags"]:
        badge = "warn" if status == "warn" else "ok"
        html_parts.append(f"<div style='margin-bottom:6px'><span class='flag {badge}'>{html.escape(name)}</span></div>")
    html_parts.append("</div>")

    html_parts.append("<div class='section'><h2>Automated commentary</h2>")
    for c in context["comments"]:
        html_parts.append(f"<div class='note'>{html.escape(c)}</div>")
    html_parts.append("</div>")

    html_parts.append("<div class='section'><h2>Data snapshot</h2>")
    html_parts.append(f"<div class='small'>History file: <a href='{context['history_path']}'>{context['history_path']}</a> &nbsp; | &nbsp; Sims file: <a href='{context['sims_path']}'>{context['sims_path']}</a> &nbsp; | &nbsp; Artifact: <a href='{context['artifact_path']}'>{context['artifact_path']}</a></div>")
    html_parts.append("<h3>Columns</h3><div class='code'>")
    html_parts.append(html.escape(", ".join(context["columns"])))
    html_parts.append("</div>")
    html_parts.append("<h3>Sample rows</h3>")
    html_parts.append(context["sample_table_html"])
    html_parts.append(f"<div style='margin-top:8px'><a id='toggle-link' class='toggle' onclick='toggleFullHistory()'>Show full history table</a></div>")
    html_parts.append("<div id='full-history' style='display:none; margin-top:12px'>")
    html_parts.append(context["full_history_table_html"])
    html_parts.append("</div>")
    html_parts.append("</div>")

    html_parts.append("<div class='section'><h2>Numeric summaries</h2>")
    html_parts.append("<table><thead><tr><th>column</th><th>count</th><th>mean</th><th>std</th><th>min</th><th>p25</th><th>median</th><th>p75</th><th>max</th></tr></thead><tbody>")
    for r in context["numeric_summary"]:
        html_parts.append("<tr>")
        html_parts.append(f"<td>{html.escape(r['col'])}</td><td>{r['count']}</td><td>{fmt_num(r['mean'])}</td><td>{fmt_num(r['std'])}</td>")
        html_parts.append(f"<td>{fmt_num(r['min'])}</td><td>{fmt_num(r['p25'])}</td><td>{fmt_num(r['median'])}</td><td>{fmt_num(r['p75'])}</td><td>{fmt_num(r['max'])}</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody></table></div>")

    html_parts.append("<div class='section'><h2>Model artifact</h2>")
    if context["artifact"]:
        art = context["artifact"]
        html_parts.append(f"<div class='small'>Artifact file: <a href='{context['artifact_path']}'>{context['artifact_path']}</a></div>")
        if art.get("param_names") and art.get("beta"):
            html_parts.append("<table><thead><tr><th>param</th><th>beta</th></tr></thead><tbody>")
            for n,b in zip(art.get("param_names"), art.get("beta")):
                html_parts.append(f"<tr><td>{html.escape(n)}</td><td>{fmt_num(b)}</td></tr>")
            html_parts.append("</tbody></table>")
        else:
            html_parts.append("<div class='code'>No parameter table in artifact.</div>")
        html_parts.append(f"<h3>Residual sigma</h3><div class='code'>{fmt_num(art.get('sigma', np.nan))}</div>")
    else:
        html_parts.append("<div class='small'>No model artifact found.</div>")
    html_parts.append("</div>")

    html_parts.append("<div class='section'><h2>Simulation percentiles (days & sprints)</h2>")
    if context["sims"]:
        html_parts.append("<div class='percentile-table'><table><thead><tr><th>%ile</th><th>days_needed</th><th>sprints_needed</th><th>representative_finish_date</th></tr></thead><tbody>")
        for r in context["percentile_rows"]:
            html_parts.append(f"<tr><td>{r['percentile']}%</td><td>{fmt_num(r['days'])}</td><td>{r['sprints']}</td><td>{html.escape(str(r['finish_date']))}</td></tr>")
        html_parts.append("</tbody></table></div>")
        html_parts.append("<h3>Sample of sims</h3>")
        html_parts.append(context["sims_sample_html"])
    else:
        html_parts.append("<div class='small'>No sims CSV found.</div>")
    html_parts.append("</div>")

    html_parts.append("<div class='section'><h2>Plots</h2>")
    if context["plots"]:
        html_parts.append("<div class='img-row'>")
        for p in context["plots"]:
            fname = Path(p).name
            html_parts.append("<div class='img-card'>")
            html_parts.append(f"<img src='{p}' alt='{fname}'/>")
            html_parts.append(f"<div class='small'>{html.escape(fname)}</div>")
            html_parts.append("</div>")
        html_parts.append("</div>")
    else:
        html_parts.append("<div class='small'>No plots found.</div>")
    html_parts.append("</div>")

    html_parts.append(HTML_TAIL.format(ts=context["generated_at"], history=context["history_path"], sims=context["sims_path"], artifact=context["artifact_path"]))

    Path(out_path).write_text("\n".join(html_parts))
    print(f"Wrote HTML report to {out_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", default="history/sprint_features.csv")
    p.add_argument("--sims", default="sims_output_regression.csv")
    p.add_argument("--artifact", default="model_artifact.json")
    p.add_argument("--plots-dir", default="plots")
    p.add_argument("--out", default="report.html")
    args = p.parse_args()

    history = safe_read_csv(args.history)
    sims = safe_read_csv(args.sims)
    artifact = None
    if Path(args.artifact).exists():
        try:
            artifact = json.loads(Path(args.artifact).read_text())
        except Exception:
            artifact = None

    plots_dir = Path(args.plots_dir)
    plots = []
    if plots_dir.exists():
        imgs = sorted([str(p) for p in plots_dir.glob("*.png")])
        seen = set()
        for im in imgs:
            if im not in seen:
                seen.add(im)
                plots.append(im)

    kpis = {}
    if history is not None:
        rows = len(history)
        cumulative_done = float(history.get("net_done", history.get("net_done_pd", pd.Series([0.0]))).sum())
        eff_days = float(history.get("effective_days", pd.Series([np.nan])).sum())
        kpis["rows"] = rows
        kpis["cumulative_net_done"] = f"{cumulative_done:.3f}"
        kpis["cumulative_effective_days"] = f"{eff_days:.1f}" if not np.isnan(eff_days) else ""
    else:
        kpis["rows"] = 0

    if history is not None:
        sample_html = history.head(8).to_html(index=False, classes="small")
        full_history_html = history.to_html(index=False, classes="small")
        cols = history.columns.tolist()
        numeric_summary = []
        num_cols = history.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            s = history[c].dropna().astype(float)
            if s.size == 0:
                continue
            numeric_summary.append({
                "col": c,
                "count": int(s.count()),
                "mean": float(s.mean()),
                "std": float(s.std(ddof=1)) if s.size>1 else float(np.nan),
                "min": float(s.min()),
                "p25": float(np.percentile(s,25)),
                "median": float(np.median(s)),
                "p75": float(np.percentile(s,75)),
                "max": float(s.max())
            })
    else:
        sample_html = "<div class='small'>No history</div>"
        full_history_html = "<div class='small'>No history</div>"
        cols = []
        numeric_summary = []

    percentile_rows = percentile_rows_from_sims(sims)

    remaining_amount = None
    effort_unit = "person_days"
    if artifact:
        remaining_amount = artifact.get("remaining_amount_in_unit", None) or artifact.get("remaining", None)
        effort_unit = artifact.get("effort_unit", effort_unit)

    comments, flags = build_commentary(history, artifact, sims, remaining_amount if remaining_amount is not None else 0.0, effort_unit)

    context = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "kpis": kpis,
        "columns": cols,
        "sample_table_html": sample_html,
        "full_history_table_html": full_history_html,
        "numeric_summary": numeric_summary,
        "artifact": artifact,
        "artifact_path": args.artifact,
        "sims": sims is not None,
        "sims_path": args.sims,
        "remaining_amount": remaining_amount,
        "effort_unit": effort_unit,
        "percentile_rows": percentile_rows,
        "sims_sample_html": (sims.head(8).to_html(index=False, classes="small") if sims is not None else "<div class='small'>No sims</div>"),
        "plots": plots,
        "comments": comments,
        "flags": flags,
        "history_path": args.history,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

    render_html(args.out, context)

if __name__ == "__main__":
    main()