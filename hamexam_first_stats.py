#!/usr/bin/env python3
"""
hamexam_first_stats.py — does a forced first question project the model into
a worse (or better) test-taker for the rest of the exam?

    python3 hamexam_first_stats.py                       # every tag in hamexam_evals.db
    python3 hamexam_first_stats.py --tag first34 --model claude-haiku-4-5-20251001
    python3 hamexam_first_stats.py --out first_report.html

Treatments are runs with a `tag` (written by hamexam_fleet.py from the
configuration json file that hamexam_make.py --first produced). Each is paired
with the untagged baseline of the same model, thinking, effort, temperature
and prompt, on the same exams: TestNN_<tag> is matched to TestNN, and the
comparison is restricted to the 49 questions both presented (the forced first
question and the baseline's own draw from its group are excluded).

For every (tag, model) pair the script reports:

  * the paired per-question accuracy difference, treatment − baseline, with
    its SE over questions and the same clustered by E-group (Miller 2024,
    Sec. 3.3 and Eq. 4), and a 95% CI
  * whether the treatment landed: how the forced question was actually
    answered at position 1
  * the difference by position in the exam — does any effect fade or persist
  * the difference by subelement
  * within-treatment split, for unstable first questions: runs where the
    model got question 1 right vs wrong, compared on the rest of the exam
  * the questions that moved most

Writes hamexam_first_report.html (stdlib only) and prints a summary.
"""

import argparse
import collections
import html
import math
import sqlite3
import sys
import time
from pathlib import Path

from hamexam_stats import CSS, mean, sd, se_clustered, f, pct

POSITION_BINS = [(2, 10), (11, 20), (21, 30), (31, 40), (41, 50)]


def load(db_path: Path):
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    runs = {r["trajectory_id"]: dict(r) for r in db.execute(
        "SELECT trajectory_id, exam, model, thinking, effort, temperature, prompt, tag, "
        "parse, answered, questions, stop_reason, right FROM runs "
        "WHERE result_type='succeeded' AND right IS NOT NULL")}
    answers = collections.defaultdict(list)
    for a in db.execute("SELECT trajectory_id, question_id, number, cluster, subelement, "
                        "given, correct, right FROM answers WHERE right IS NOT NULL"):
        answers[a["trajectory_id"]].append(dict(a))
    db.close()
    return runs, answers


def cfg_key(r):
    return (r["model"], bool(r["thinking"]), r["effort"], r["temperature"], r["prompt"])


def exam_stem(path):
    return Path(path).stem


def base_stem(stem, tag):
    return stem[:-len(tag) - 1] if stem.endswith("_" + tag) else None


def clean(runs):
    """Drop runs the log would flag: non-JSON parse, short answers, odd stop."""
    return {t: r for t, r in runs.items()
            if r["parse"] == "json" and (r["answered"] or 0) >= (r["questions"] or 0)
            and r["stop_reason"] in (None, "end_turn")}


def analyze_pair(tag, key, runs, answers):
    treat = {t: r for t, r in runs.items() if r["tag"] == tag and cfg_key(r) == key}
    base = {t: r for t, r in runs.items() if not r["tag"] and cfg_key(r) == key}
    if not treat or not base:
        return None

    # per-exam-stem, per-question accuracy lists, both sides
    t_by_exam = collections.defaultdict(lambda: collections.defaultdict(list))
    b_by_exam = collections.defaultdict(lambda: collections.defaultdict(list))
    t_pos = {}                       # (stem, qid) -> position in treatment exam
    first_q = collections.Counter()  # how the forced question was answered
    first_right = []                 # per treatment run: (right at pos 1, rest accuracy dict)
    q_group, q_sub = {}, {}
    forced_id = None
    for t, r in treat.items():
        stem = base_stem(exam_stem(r["exam"]), tag)
        if stem is None:
            continue
        rest = {}
        for a in answers.get(t, []):
            q_group[a["question_id"]] = a["cluster"] or "?"
            q_sub[a["question_id"]] = a["subelement"] or "?"
            if a["number"] == 1:
                forced_id = a["question_id"]
                first_q[a["given"] or "-"] += 1
                first_ok = a["right"]
                continue
            t_by_exam[stem][a["question_id"]].append(a["right"])
            t_pos[(stem, a["question_id"])] = a["number"]
            rest[a["question_id"]] = a["right"]
        first_right.append((first_ok, rest))
    for t, r in base.items():
        stem = exam_stem(r["exam"])
        if stem not in t_by_exam:
            continue
        for a in answers.get(t, []):
            b_by_exam[stem][a["question_id"]].append(a["right"])

    # paired differences over shared questions
    rows = []   # (stem, qid, t_mean, b_mean, diff, position, group, sub)
    for stem, tq in t_by_exam.items():
        bq = b_by_exam.get(stem, {})
        for qid in tq:
            if qid in bq and qid != forced_id:
                tm, bm = mean(tq[qid]), mean(bq[qid])
                rows.append((stem, qid, tm, bm, tm - bm, t_pos[(stem, qid)],
                             q_group[qid], q_sub[qid]))
    if len(rows) < 2:
        return None
    d = [x[4] for x in rows]
    out = {
        "tag": tag, "key": key, "forced_id": forced_id, "first_given": first_q,
        "n_treat": len(treat), "n_base": len(base), "exams": len(t_by_exam),
        "shared_q": len(rows),
        "t_p": mean(x[2] for x in rows), "b_p": mean(x[3] for x in rows),
        "diff": mean(d), "se": sd(d) / math.sqrt(len(d)),
        "se_g": se_clustered(d, [x[6] for x in rows]),
        "rows": rows,
    }
    # by position
    out["by_pos"] = []
    for lo, hi in POSITION_BINS:
        dd = [x[4] for x in rows if lo <= x[5] <= hi]
        if len(dd) >= 2:
            out["by_pos"].append((f"{lo}–{hi}", mean(dd), sd(dd) / math.sqrt(len(dd)), len(dd)))
    # by subelement
    out["by_sub"] = []
    for sub in sorted({x[7] for x in rows}, key=lambda s: (len(s), s)):
        dd = [x[4] for x in rows if x[7] == sub]
        if len(dd) >= 2:
            out["by_sub"].append((sub, mean(dd), sd(dd) / math.sqrt(len(dd)), len(dd)))
    # within-treatment split on whether question 1 was right
    ok_runs = [rest for ok, rest in first_right if ok]
    bad_runs = [rest for ok, rest in first_right if not ok]
    out["split"] = None
    if len(ok_runs) >= 2 and len(bad_runs) >= 2:
        qs = set().union(*(r.keys() for r in ok_runs)) & set().union(*(r.keys() for r in bad_runs))
        dd, gg = [], []
        for q in qs:
            a = [r[q] for r in ok_runs if q in r]
            b = [r[q] for r in bad_runs if q in r]
            if a and b:
                dd.append(mean(a) - mean(b)); gg.append(q_group[q])
        if len(dd) >= 2:
            out["split"] = {"n_ok": len(ok_runs), "n_bad": len(bad_runs), "q": len(dd),
                            "diff": mean(dd), "se": sd(dd) / math.sqrt(len(dd)),
                            "se_g": se_clustered(dd, gg)}
    return out


def ci(m, s):
    return f"{pct(m - 1.96 * s)} – {pct(m + 1.96 * s)}"


def verdict(m, s):
    lo, hi = m - 1.96 * s, m + 1.96 * s
    if lo > 0:
        return "pill ok", "treatment higher"
    if hi < 0:
        return "pill bad", "treatment lower"
    return "pill", "no separation"


def bar_svg(items, width=520):
    """Horizontal bars of mean diff ± SE, zero line in the middle."""
    if not items:
        return ""
    lim = max(0.02, max(abs(m) + s for _, m, s, _ in items)) * 1.1
    row_h, left, right = 26, 90, 150
    mid = left + (width - left - right) / 2
    scale = (width - left - right) / 2 / lim
    h = row_h * len(items) + 10
    o = [f'<svg viewBox="0 0 {width} {h}" width="{width}" style="max-width:100%" '
         f'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace" font-size="11">']
    o.append(f'<line x1="{mid}" y1="0" x2="{mid}" y2="{h}" stroke="#4a5675" stroke-dasharray="3,3"/>')
    for i, (label, m, s, n) in enumerate(items):
        y = 5 + i * row_h
        x0, x1 = mid + min(0, m) * scale, mid + max(0, m) * scale
        color = "#1f8f5f" if m - 1.96 * s > 0 else "#c8102e" if m + 1.96 * s < 0 else "#9aa3b8"
        o.append(f'<text x="4" y="{y + 15}" fill="#1b2a49">{html.escape(label)}</text>')
        o.append(f'<rect x="{x0}" y="{y + 5}" width="{max(1, x1 - x0)}" height="14" fill="{color}"/>')
        o.append(f'<line x1="{mid + (m - 1.96 * s) * scale}" y1="{y + 12}" '
                 f'x2="{mid + (m + 1.96 * s) * scale}" y2="{y + 12}" stroke="#1b2a49" stroke-width="1.5"/>')
        o.append(f'<text x="{width - 6}" y="{y + 15}" text-anchor="end" fill="#4a5675">'
                 f'{m * 100:+.1f} ± {s * 100:.1f} (n={n})</text>')
    o.append("</svg>")
    return "\n".join(o)


def render(results, db_path):
    H = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
         f"<title>Forced first question — {html.escape(db_path.name)}</title>"
         f"<style>{CSS}</style></head><body><header><h1>Forced first question</h1>"
         f"<div class='meta'>{html.escape(str(db_path))} · {len(results)} treatment/model "
         f"pair(s) · generated {time.strftime('%Y-%m-%d %H:%M')}</div></header><main>"]
    H.append("<section><h2>Summary</h2><p class='lede'>Each row: runs whose exam had a "
             "forced question at position 1, against the same model's baseline on the same "
             "exams, over the 49 questions both presented. diff is treatment − baseline "
             "per-question accuracy; SE is over questions (Sec. 3.3), and again clustered by "
             "E-group (Eq. 4). Flagged runs are excluded.</p><table><tr><th class='l'>treatment</th>"
             "<th class='l'>model</th><th>runs (t/b)</th><th>exams</th><th>shared q</th>"
             "<th>baseline p̂</th><th>treatment p̂</th><th>diff</th><th>SE</th><th>95% CI</th>"
             "<th>95% CI (E-group)</th><th class='l'>verdict</th></tr>")
    for r in results:
        cls, word = verdict(r["diff"], r["se"])
        H.append(f"<tr><td class='l'>{html.escape(r['tag'])}</td><td class='l'>{html.escape(r['key'][0])}</td>"
                 f"<td>{r['n_treat']}/{r['n_base']}</td><td>{r['exams']}</td><td>{r['shared_q']}</td>"
                 f"<td>{pct(r['b_p'])}</td><td>{pct(r['t_p'])}</td>"
                 f"<td><span class='{cls}'>{r['diff'] * 100:+.1f}</span></td><td>{pct(r['se'])}</td>"
                 f"<td>{ci(r['diff'], r['se'])}</td><td>{ci(r['diff'], r['se_g'])}</td>"
                 f"<td class='l'>{word}</td></tr>")
    H.append("</table></section>")

    for r in results:
        label = f"{r['tag']} · {r['key'][0]}"
        H.append(f"<section><h2>{html.escape(label)}</h2>")
        given = ", ".join(f"{k}×{v}" for k, v in r["first_given"].most_common())
        H.append(f"<p class='lede'>Forced question id {r['forced_id']} at position 1 was "
                 f"answered {given} across {r['n_treat']} treatment runs — this is whether "
                 f"the treatment landed as designed.</p>")
        H.append("<div class='grid'><div class='card'><h3>Difference by position</h3>"
                 "<div class='sub'>treatment − baseline, questions binned by their position "
                 "in the treatment exam. A projection effect that decays shows as a gradient "
                 "toward zero.</div>" + bar_svg(r["by_pos"]) + "</div>")
        H.append("<div class='card'><h3>Difference by subelement</h3><div class='sub'>same, "
                 "grouped by subelement.</div>" + bar_svg(r["by_sub"]) + "</div></div>")
        if r["split"]:
            s = r["split"]
            cls, word = verdict(s["diff"], s["se"])
            H.append(f"<p class='lede' style='margin-top:14px'><b>Within-treatment split.</b> "
                     f"Runs that got question 1 right ({s['n_ok']}) vs wrong ({s['n_bad']}), "
                     f"compared on the rest of the exam: right-minus-wrong "
                     f"<span class='{cls}'>{s['diff'] * 100:+.1f}</span> ± {pct(s['se'])} "
                     f"(E-group ± {pct(s['se_g'])}) over {s['q']} questions — {word}. This is "
                     f"the sharpest test when the forced question is one the model wobbles on.</p>")
        moved = sorted(r["rows"], key=lambda x: abs(x[4]), reverse=True)[:15]
        H.append("<details><summary>Questions that moved most</summary><table><tr><th class='l'>exam</th>"
                 "<th>id</th><th>group</th><th>pos</th><th>baseline</th><th>treatment</th><th>diff</th></tr>")
        for stem, qid, tm, bm, dd, pos, grp, _ in moved:
            H.append(f"<tr><td>{html.escape(stem)}</td><td>{qid}</td><td>{grp}</td><td>{pos}</td>"
                     f"<td>{pct(bm)}</td><td>{pct(tm)}</td><td>{dd * 100:+.0f}</td></tr>")
        H.append("</table></details></section>")
    H.append("</main><footer>Paired differences per Miller, <i>Adding Error Bars to Evals</i> "
             "(2024), Sec. 3.3; E-group clustering per Eq. 4.</footer></body></html>")
    return "\n".join(H)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", type=Path, default=Path("hamexam_evals.db"))
    ap.add_argument("--out", type=Path, default=Path("hamexam_first_report.html"))
    ap.add_argument("--tag", action="append", default=None, help="Restrict to these tags")
    ap.add_argument("--model", action="append", default=None, help="Restrict to these models")
    ap.add_argument("--keep-flagged", action="store_true",
                    help="Include runs with parse/answer-count/stop-reason problems")
    args = ap.parse_args()
    if not args.db.exists():
        print(f"error: {args.db} not found", file=sys.stderr)
        return 1
    runs, answers = load(args.db)
    if not args.keep_flagged:
        n = len(runs)
        runs = clean(runs)
        if n - len(runs):
            print(f"excluded {n - len(runs)} flagged run(s)")
    tags = sorted({r["tag"] for r in runs.values() if r["tag"]})
    if args.tag:
        tags = [t for t in tags if t in args.tag]
    if not tags:
        print("no tagged runs in db", file=sys.stderr)
        return 1
    results = []
    for tag in tags:
        keys = sorted({cfg_key(r) for r in runs.values() if r["tag"] == tag}, key=str)
        for key in keys:
            if args.model and key[0] not in args.model:
                continue
            res = analyze_pair(tag, key, runs, answers)
            if res is None:
                print(f"{tag} · {key[0]}: no matching baseline runs — skipped")
                continue
            results.append(res)
            _, word = verdict(res["diff"], res["se"])
            print(f"{tag} · {key[0]}: {res['n_treat']} treatment vs {res['n_base']} baseline runs, "
                  f"{res['shared_q']} shared q; diff {res['diff'] * 100:+.1f} ± {res['se'] * 100:.1f} "
                  f"(E-group ± {res['se_g'] * 100:.1f}) — {word}; q1 answered "
                  + ", ".join(f"{k}×{v}" for k, v in res["first_given"].most_common()))
            if res["split"]:
                s = res["split"]
                print(f"   q1-right vs q1-wrong runs ({s['n_ok']}/{s['n_bad']}): "
                      f"{s['diff'] * 100:+.1f} ± {s['se'] * 100:.1f}")
    if not results:
        return 1
    args.out.write_text(render(results, args.db), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
