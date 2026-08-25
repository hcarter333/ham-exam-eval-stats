#!/usr/bin/env python3
"""
hamexam_stats.py — statistics and a static HTML dashboard from hamexam_evals.db.

    python3 hamexam_stats.py                       # reads hamexam_evals.db, writes hamexam_report.html
    python3 hamexam_stats.py --db other.db --out report.html --pool extra_pool.json

Stdlib only. Everything is computed from the `runs` and `answers` tables that
hamexam_fleet.py (and hamexam_backfill.py) write. A "configuration" is the
tuple (model, thinking, effort, temperature, prompt) — every run with the same
tuple pools together regardless of which manifest produced it.

Statistics, following Miller, "Adding Error Bars to Evals" (2024):

  * naive SE       — Eq. 1, treats every (run, question) score as independent
  * clustered SE   — Eq. 4, cluster-robust, with clusters = question id
                     (replicas of the same question are not independent) and
                     separately clusters = E-group (E7B etc., same topic)
  * paired diff    — Sec. 3.3, two configurations compared on the questions
                     they both answered; per-question mean difference, SE over
                     questions, and the same clustered by E-group
  * pass rate, score mean/sd per configuration, per-subelement accuracy,
    per-question stability, and a run log flagging parse failures.

The pool map (one SVG per configuration) draws every pool question as a cell
in its E-group row, colored by observed accuracy — the clustering structure
made visible. Question text comes from --pool (extra_pool.json) when present.
"""

import argparse
import collections
import html
import itertools
import json
import math
import sqlite3
import sys
import time
from pathlib import Path

PASS_MARK = 37

# ---------------------------------------------------------------- statistics

def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def sd(xs):
    xs = list(xs)
    if len(xs) < 2:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def se_naive(scores):
    """Eq. 1: sqrt(var/n) with population variance, matching Eq. 4's first term."""
    n = len(scores)
    if n < 2:
        return float("nan")
    m = mean(scores)
    return math.sqrt(sum((s - m) ** 2 for s in scores) / n / n)


def se_clustered(scores, clusters):
    """Eq. 4, in its cluster-robust form: (1/n^2) sum_c (sum_{i in c}(s_i - sbar))^2.
    Expanding the square gives the CLT term plus the i!=j cross terms."""
    n = len(scores)
    if n < 2:
        return float("nan")
    m = mean(scores)
    resid = collections.defaultdict(float)
    for s, c in zip(scores, clusters):
        resid[c] += s - m
    return math.sqrt(sum(r * r for r in resid.values())) / n


# -------------------------------------------------------------------- loading

def config_key(r):
    return (r["model"], bool(r["thinking"]), r["effort"], r["temperature"], r["prompt"],
            r.get("tag"))


def config_label(k):
    model, thinking, effort, temp, prompt, tag = k
    bits = [model]
    if tag:
        bits.append(tag)
    if thinking:
        bits.append(f"thinking{':' + effort if effort else ''}")
    if temp is not None and temp != 1.0:
        bits.append(f"T={temp}")
    if prompt:
        bits.append(Path(prompt).stem)
    return " · ".join(bits)


def load(db_path: Path):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    runs = [dict(r) for r in db.execute("SELECT * FROM runs ORDER BY finished_at, trajectory_id")]
    answers = [dict(r) for r in db.execute(
        "SELECT trajectory_id, question_id, cluster, subelement, given, correct, right "
        "FROM answers WHERE right IS NOT NULL")]
    db.close()
    return runs, answers


def load_pool(pool_path: Path | None):
    """{question_id: {'group': 'E7B', 'text': ..., 'answer': 'B'}} for the whole pool."""
    if pool_path is None or not pool_path.exists():
        return {}
    pool = json.loads(pool_path.read_text(encoding="utf-8-sig"))
    return {str(q["id"]): {"group": f"E{q['subelement']}{q['group_index']}",
                           "num": q["group_number"],
                           "text": " ".join(str(q["question"]).split()),
                           "answer": str(q["answer"]).strip().upper()[:1]}
            for q in pool}


# ------------------------------------------------------------------ analysis

def analyze(runs, answers):
    by_tid = {r["trajectory_id"]: r for r in runs}
    ok_runs = [r for r in runs if r["result_type"] == "succeeded" and r["right"] is not None]

    configs = collections.OrderedDict()
    for r in ok_runs:
        configs.setdefault(config_key(r), []).append(r)

    ans_by_cfg = collections.defaultdict(list)
    for a in answers:
        r = by_tid.get(a["trajectory_id"])
        if r is None or r["result_type"] != "succeeded":
            continue
        ans_by_cfg[config_key(r)].append(a)

    cfg_stats = []
    for k, rs in configs.items():
        A = ans_by_cfg.get(k, [])
        scores = [a["right"] for a in A]
        qids = [a["question_id"] for a in A]
        groups = [a["cluster"] or "?" for a in A]
        totals = [r["right"] for r in rs]
        per_q = collections.defaultdict(list)
        for a in A:
            per_q[a["question_id"]].append(a["right"])
        p_hat = mean(scores)
        s_naive = se_naive(scores)
        s_q = se_clustered(scores, qids)
        s_g = se_clustered(scores, groups)
        sub = collections.defaultdict(lambda: [0, 0])
        for a in A:
            sub[a["subelement"] or "?"][1] += 1
            sub[a["subelement"] or "?"][0] += a["right"]
        cfg_stats.append({
            "key": k, "label": config_label(k), "runs": len(rs),
            "exams": len({r["exam"] for r in rs}),
            "questions": len(per_q), "n": len(scores),
            "score_mean": mean(totals), "score_sd": sd(totals),
            "score_min": min(totals), "score_max": max(totals),
            "pass": sum(t >= PASS_MARK for t in totals),
            "p": p_hat, "se_naive": s_naive, "se_q": s_q, "se_g": s_g,
            "per_q": {q: mean(v) for q, v in per_q.items()},
            "per_q_n": {q: len(v) for q, v in per_q.items()},
            "sub": dict(sorted(sub.items())),
            "served": sorted({r.get("model_served") for r in rs if r.get("model_served")}),
            "flagged": [r for r in rs if r["parse"] != "json"
                        or (r["answered"] or 0) < (r["questions"] or 0)
                        or r["stop_reason"] not in (None, "end_turn")],
        })

    # paired comparisons on shared questions, per-question means, clustered by group
    q_group = {a["question_id"]: a["cluster"] or "?" for a in answers}
    pairs = []
    for A, B in itertools.combinations(cfg_stats, 2):
        shared = sorted(set(A["per_q"]) & set(B["per_q"]))
        if len(shared) < 2:
            continue
        d = [A["per_q"][q] - B["per_q"][q] for q in shared]
        g = [q_group[q] for q in shared]
        pairs.append({"a": A["label"], "b": B["label"], "q": len(shared),
                      "diff": mean(d), "se": sd(d) / math.sqrt(len(d)),
                      "se_g": se_clustered(d, g),
                      "a_p": mean(A["per_q"][q] for q in shared),
                      "b_p": mean(B["per_q"][q] for q in shared)})

    # per-question overall: accuracy, n, answer distribution
    per_q_all = collections.defaultdict(list)
    per_q_given = collections.defaultdict(collections.Counter)
    correct_of = {}
    for a in answers:
        per_q_all[a["question_id"]].append(a["right"])
        per_q_given[a["question_id"]][a["given"] or "-"] += 1
        correct_of[a["question_id"]] = a["correct"]
    qtable = sorted(
        ({"id": q, "group": q_group[q], "p": mean(v), "n": len(v),
          "correct": correct_of[q], "given": per_q_given[q]}
         for q, v in per_q_all.items()),
        key=lambda x: (x["p"], -x["n"]))

    return {"runs": runs, "ok_runs": ok_runs, "configs": cfg_stats, "pairs": pairs,
            "qtable": qtable, "q_group": q_group}


# ------------------------------------------------------------------- report

CSS = """
:root{--ink:#1b2a49;--ink2:#4a5675;--rule:#c9cfdd;--paper:#f7f8fb;--card:#fff;
--ok:#1f8f5f;--mid:#e0a800;--bad:#c8102e;--none:#e6e9f0;--ocean:#0d6efd}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);
font:15px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,"Liberation Mono",monospace}
header{padding:28px 36px 18px;border-bottom:3px double var(--rule);background:var(--card)}
header h1{margin:0 0 4px;font:600 26px/1.2 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif;letter-spacing:-.01em}
header .meta{color:var(--ink2);font-size:13px}
main{padding:0 36px 48px;max-width:1400px}
section{margin-top:40px}
h2{font:600 18px/1.3 ui-sans-serif,system-ui,sans-serif;margin:0 0 6px;border-left:6px solid var(--ocean);padding-left:10px}
.lede{color:var(--ink2);font-size:13px;margin:0 0 14px;max-width:80ch;font-family:ui-sans-serif,system-ui,sans-serif}
table{border-collapse:collapse;width:100%;background:var(--card);font-size:13px}
th,td{padding:6px 10px;border-bottom:1px solid var(--rule);text-align:right;vertical-align:top;white-space:nowrap}
th{background:#eef1f7;text-align:right;font-weight:600;color:var(--ink2);position:sticky;top:0}
th:first-child,td:first-child{text-align:left}
td.l,th.l{text-align:left;white-space:normal}
tr.flag td{background:#fff4f4}
.pill{display:inline-block;padding:0 7px;border-radius:9px;font-size:12px;color:#fff;background:var(--ink2)}
.pill.ok{background:var(--ok)}.pill.bad{background:var(--bad)}.pill.mid{background:var(--mid);color:var(--ink)}
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:18px}
.card{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:14px 16px}
.card h3{margin:0 0 6px;font:600 14px/1.3 ui-sans-serif,system-ui,sans-serif}
.big{font-size:30px;font-weight:600;line-height:1.1}
.sub{color:var(--ink2);font-size:12px}
.bar{height:10px;background:var(--none);border-radius:5px;overflow:hidden;display:inline-block;width:120px;vertical-align:middle;margin-right:6px}
.bar i{display:block;height:100%;background:var(--ok)}
svg.pool{max-width:100%;height:auto;background:var(--card);border:1px solid var(--rule);border-radius:8px}
svg.pool text{font:10px ui-monospace,monospace;fill:var(--ink2)}
.legend span{display:inline-block;width:14px;height:14px;vertical-align:middle;margin:0 4px 0 12px;border-radius:2px}
details summary{cursor:pointer;color:var(--ocean);font-family:ui-sans-serif,system-ui,sans-serif;font-size:13px}
.qtext{font-family:ui-sans-serif,system-ui,sans-serif;color:var(--ink);white-space:normal;max-width:60ch}
footer{color:var(--ink2);font-size:12px;padding:20px 36px;border-top:1px solid var(--rule)}
"""


def f(x, d=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def pct(x):
    return "–" if x is None or math.isnan(x) else f"{100 * x:.1f}%"


def color_for(p):
    if p is None or math.isnan(p):
        return "var(--none)"
    if p >= 0.95:
        return "#1f8f5f"
    if p >= 0.75:
        return "#7dc67d"
    if p >= 0.5:
        return "#e0a800"
    if p > 0:
        return "#f08a5d"
    return "#c8102e"


def pool_map_svg(cfg, pool, q_group):
    """50 rows (E-groups), one cell per pool question in group_number order
    (or observed questions only when the pool file is absent)."""
    if pool:
        rows = collections.defaultdict(list)
        for qid, q in pool.items():
            rows[q["group"]].append((q["num"], qid))
    else:
        rows = collections.defaultdict(list)
        for qid, g in q_group.items():
            rows[g].append((qid.zfill(4), qid))
    groups = sorted(rows, key=lambda g: (int(g[1]), g[2:]))
    cell, gap, left, top = 18, 3, 46, 8
    width = left + max(len(v) for v in rows.values()) * (cell + gap) + 8
    height = top + len(groups) * (cell + gap) + 4
    out = [f'<svg class="pool" viewBox="0 0 {width} {height}" width="{int(width * 1.5)}" '
           f'xmlns="http://www.w3.org/2000/svg">']
    for r, g in enumerate(groups):
        y = top + r * (cell + gap)
        out.append(f'<text x="4" y="{y + cell - 4}">{g}</text>')
        for c, (_, qid) in enumerate(sorted(rows[g])):
            x = left + c * (cell + gap)
            p = cfg["per_q"].get(qid)
            n = cfg["per_q_n"].get(qid, 0)
            title = f"{g} id {qid}: " + (f"{pct(p)} of {n}" if n else "not sampled")
            if pool and qid in pool:
                title += " — " + pool[qid]["text"][:120]
            out.append(f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" rx="2" '
                       f'fill="{color_for(p)}"><title>{html.escape(title)}</title></rect>')
    out.append("</svg>")
    return "\n".join(out)


def served_note(c):
    if c["served"] and c["served"] != [c["key"][0]]:
        return f"<div class='sub'>served as {html.escape(', '.join(c['served']))}</div>"
    return ""


def render(an, pool, db_path):
    cfgs, pairs = an["configs"], an["pairs"]
    H = []
    H.append(f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
             f"<title>Ham exam fleet — {html.escape(db_path.name)}</title>"
             f"<style>{CSS}</style></head><body>")
    n_exams = len({r['exam'] for r in an['ok_runs']})
    n_q = len(an["qtable"])
    H.append(f"<header><h1>Ham exam fleet report</h1><div class='meta'>"
             f"{html.escape(str(db_path))} · {len(an['runs'])} runs "
             f"({len(an['ok_runs'])} succeeded) · {n_exams} exams · {n_q} distinct "
             f"questions{f' of {len(pool)} in pool' if pool else ''} · "
             f"{len(cfgs)} configurations · generated {time.strftime('%Y-%m-%d %H:%M')}"
             f"</div></header><main>")

    # --- configurations
    H.append("<section><h2>Configurations</h2><p class='lede'>Exam score is out of "
             f"50; pass is {PASS_MARK}. p̂ is mean per-question accuracy over every "
             "(run, question) score. The three SEs are for p̂: naive (Eq. 1, all scores "
             "independent), clustered by question id (Eq. 4 — replicas of the same "
             "question share a cluster), and clustered by E-group (same topic).</p>")
    H.append("<div class='grid'>")
    for c in cfgs:
        ratio = c["se_q"] / c["se_naive"] if c["se_naive"] else float("nan")
        H.append(f"<div class='card'><h3>{html.escape(c['label'])}</h3>"
                 f"<div class='big'>{f(c['score_mean'])} <span class='sub'>± {f(c['score_sd'])} sd</span></div>"
                 f"<div class='sub'>{c['runs']} runs · {c['exams']} exams · {c['questions']} questions · "
                 f"pass {c['pass']}/{c['runs']} · range {c['score_min']}–{c['score_max']}</div>"
                 f"{served_note(c)}"
                 f"<table style='margin-top:10px'><tr><th class='l'>p̂</th><td>{pct(c['p'])}</td></tr>"
                 f"<tr><th class='l'>SE naive</th><td>{pct(c['se_naive'])}</td></tr>"
                 f"<tr><th class='l'>SE clustered by question</th><td>{pct(c['se_q'])} "
                 f"<span class='sub'>({f(ratio, 2)}× naive)</span></td></tr>"
                 f"<tr><th class='l'>SE clustered by E-group</th><td>{pct(c['se_g'])}</td></tr>"
                 f"<tr><th class='l'>95% CI (by question)</th><td>{pct(c['p'] - 1.96 * c['se_q'])} – "
                 f"{pct(min(1.0, c['p'] + 1.96 * c['se_q']))}</td></tr></table></div>")
    H.append("</div></section>")

    # --- paired comparisons
    if pairs:
        H.append("<section><h2>Paired comparisons</h2><p class='lede'>Sec. 3.3: for "
                 "questions both configurations answered, the per-question accuracy "
                 "difference (A − B, means over replicas), its SE across questions, and "
                 "the same clustered by E-group. A CI that excludes 0 separates the two.</p>"
                 "<table><tr><th class='l'>A</th><th class='l'>B</th><th>shared q</th>"
                 "<th>A p̂</th><th>B p̂</th><th>diff</th><th>SE</th><th>95% CI</th>"
                 "<th>SE (E-group)</th><th>95% CI (E-group)</th></tr>")
        for p in pairs:
            lo, hi = p["diff"] - 1.96 * p["se"], p["diff"] + 1.96 * p["se"]
            lo2, hi2 = p["diff"] - 1.96 * p["se_g"], p["diff"] + 1.96 * p["se_g"]
            sig = "pill ok" if lo > 0 or hi < 0 else "pill"
            H.append(f"<tr><td class='l'>{html.escape(p['a'])}</td><td class='l'>{html.escape(p['b'])}</td>"
                     f"<td>{p['q']}</td><td>{pct(p['a_p'])}</td><td>{pct(p['b_p'])}</td>"
                     f"<td><span class='{sig}'>{pct(p['diff'])}</span></td><td>{pct(p['se'])}</td>"
                     f"<td>{pct(lo)} – {pct(hi)}</td><td>{pct(p['se_g'])}</td>"
                     f"<td>{pct(lo2)} – {pct(hi2)}</td></tr>")
        H.append("</table></section>")

    # --- pool maps
    H.append("<section><h2>Pool map</h2><p class='lede'>Every question in the pool, one "
             "row per E-group, colored by observed accuracy for the configuration. Hover a "
             "cell for id, accuracy, sample count, and question text. Grey cells were never "
             "sampled. Rows are the paper's clusters made visible.</p>"
             "<div class='legend sub'>accuracy <span style='background:#c8102e'></span>0 "
             "<span style='background:#f08a5d'></span>&lt;50% <span style='background:#e0a800'></span>50–75% "
             "<span style='background:#7dc67d'></span>75–95% <span style='background:#1f8f5f'></span>≥95% "
             "<span style='background:#e6e9f0'></span>unsampled</div>")
    for c in cfgs:
        H.append(f"<h3 style='margin:16px 0 6px;font:600 14px ui-sans-serif,system-ui'>"
                 f"{html.escape(c['label'])}</h3>{pool_map_svg(c, pool, an['q_group'])}")
    H.append("</section>")

    # --- subelements
    subs = sorted({s for c in cfgs for s in c["sub"]}, key=lambda s: (len(s), s))
    H.append("<section><h2>Accuracy by subelement</h2><table><tr><th class='l'>subelement</th>"
             + "".join(f"<th>{html.escape(c['label'])}</th>" for c in cfgs) + "</tr>")
    for s in subs:
        H.append(f"<tr><td>{s}</td>")
        for c in cfgs:
            r, n = c["sub"].get(s, (0, 0))
            p = r / n if n else float("nan")
            H.append(f"<td><span class='bar'><i style='width:{0 if n == 0 else 100 * p:.0f}%'></i></span>"
                     f"{pct(p)} <span class='sub'>({r}/{n})</span></td>")
        H.append("</tr>")
    H.append("</table></section>")

    # --- hardest questions
    hard = [q for q in an["qtable"] if q["p"] < 1.0][:40]
    H.append("<section><h2>Questions missed</h2><p class='lede'>All configurations pooled, "
             "lowest accuracy first. Given-answer counts show whether misses agree on a "
             "wrong choice (a knowledge gap) or scatter (guessing).</p>")
    if hard:
        H.append("<table><tr><th>id</th><th>group</th><th>acc</th><th>n</th><th>key</th>"
                 "<th class='l'>given</th><th class='l'>question</th></tr>")
        for q in hard:
            given = ", ".join(f"{k}×{v}" for k, v in q["given"].most_common())
            text = pool.get(q["id"], {}).get("text", "")
            H.append(f"<tr><td>{q['id']}</td><td>{q['group']}</td><td>{pct(q['p'])}</td>"
                     f"<td>{q['n']}</td><td>{q['correct'] or '–'}</td><td class='l'>{given}</td>"
                     f"<td class='l'><div class='qtext'>{html.escape(text)}</div></td></tr>")
        H.append("</table>")
    else:
        H.append("<p class='lede'>Nothing was missed by any configuration.</p>")
    H.append("</section>")

    # --- run log
    H.append("<section><h2>Run log</h2><p class='lede'>Rows flagged red had a non-JSON parse, "
             "fewer answers than questions, a stop reason other than end_turn, or failed "
             "outright — check these before trusting their scores.</p>"
             "<details><summary>Show all runs</summary><table><tr><th class='l'>trajectory</th>"
             "<th class='l'>exam</th><th class='l'>configuration</th><th class='l'>served</th><th>score</th><th>answered</th>"
             "<th>parse</th><th>stop</th><th>in tok</th><th>out tok</th><th class='l'>finished</th></tr>")
    for r in an["runs"]:
        ok = r["result_type"] == "succeeded"
        flag = (not ok or r["parse"] != "json" or (r["answered"] or 0) < (r["questions"] or 0)
                or r["stop_reason"] not in (None, "end_turn"))
        H.append(f"<tr class='{'flag' if flag else ''}'><td>{html.escape(r['trajectory_id'])}</td>"
                 f"<td class='l'>{html.escape(Path(r['exam']).stem if r['exam'] else '')}</td>"
                 f"<td class='l'>{html.escape(config_label(config_key(r)))}</td>"
                 f"<td class='l'>{html.escape(r.get('model_served') or '')}</td>"
                 f"<td>{r['right'] if ok else html.escape(r['result_type'] or '')}</td>"
                 f"<td>{r['answered'] or ''}/{r['questions'] or ''}</td><td>{r['parse'] or ''}</td>"
                 f"<td>{r['stop_reason'] or ''}</td><td>{r['input_tokens'] or ''}</td>"
                 f"<td>{r['output_tokens'] or ''}</td><td class='l'>{r['finished_at'] or ''}</td></tr>")
    H.append("</table></details></section>")

    H.append("</main><footer>Standard errors per Miller, <i>Adding Error Bars to Evals</i> "
             "(2024), Eq. 1, Eq. 4, and Sec. 3.3. Clustered SE uses the cluster-robust "
             "form (1/n²)Σ_c(Σ_i∈c(s_i − s̄))², which expands to Eq. 4.</footer></body></html>")
    return "\n".join(H)


def print_summary(an):
    for c in an["configs"]:
        print(f"{c['label']}: {c['runs']} runs, {c['exams']} exams, {c['questions']} q; "
              f"score {f(c['score_mean'])}±{f(c['score_sd'])} pass {c['pass']}/{c['runs']}; "
              f"p={pct(c['p'])} SE naive {pct(c['se_naive'])} by-q {pct(c['se_q'])} "
              f"by-group {pct(c['se_g'])}")
        for r in c["flagged"]:
            print(f"   flagged: {r['trajectory_id']} parse={r['parse']} "
                  f"answered={r['answered']}/{r['questions']} stop={r['stop_reason']}")
    for p in an["pairs"]:
        print(f"{p['a']} − {p['b']}: {pct(p['diff'])} ± {pct(p['se'])} "
              f"(by-group ± {pct(p['se_g'])}) over {p['q']} shared q")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("hamexam_evals.db"))
    ap.add_argument("--out", type=Path, default=Path("hamexam_report.html"))
    ap.add_argument("--pool", type=Path, default=Path("extra_pool.json"),
                    help="Pool json for question text and full pool map (optional)")
    args = ap.parse_args()
    if not args.db.exists():
        print(f"error: {args.db} not found", file=sys.stderr)
        return 1
    runs, answers = load(args.db)
    if not runs:
        print("no runs in db", file=sys.stderr)
        return 1
    pool = load_pool(args.pool)
    an = analyze(runs, answers)
    print_summary(an)
    args.out.write_text(render(an, pool, args.db), encoding="utf-8")
    print(f"wrote {args.out} ({args.out.stat().st_size // 1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
