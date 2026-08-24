#!/usr/bin/env python3
"""
Score ham-exam answer files against a key.

    python3 hamexam_score.py --key keys/FirstTest.key.json answers/*.answers.json

Key formats accepted (all map question id -> letter):
    {"599": "A", "9": "D", ...}
    [{"id": "599", "answer": "A"}, ...]
    [{"id": "599", "correct": "A"}, ...]          (scraper-with-key shape)
    a list of question dicts carrying a "correct"/"answer" field

Passing score for the Extra element is 37/50 (74%). Sub-element breakdown
(E1..E9, E0) comes from the question's cluster field in the exam json when
the answers file was written by hamexam_batch/hamexam_fleet, or from a
--exam json if you point one at us.
"""

import argparse
import collections
import json
import sys
from pathlib import Path

PASS_MARK = 37


def load_key(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    key: dict[str, str] = {}
    if isinstance(data, dict):
        for k, v in data.items():
            key[str(k)] = str(v).strip().upper()[:1]
    elif isinstance(data, list):
        for q in data:
            if not isinstance(q, dict) or "id" not in q:
                continue
            v = q.get("correct", q.get("answer"))
            if v is not None:
                key[str(q["id"])] = str(v).strip().upper()[:1]
    if not key:
        raise ValueError(f"{path}: no id->letter pairs found")
    return key


def load_answers(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, dict) and "answers" in data:
        return data["answers"]
    return data


def score(answers: list[dict], key: dict[str, str],
          exam: list[dict] | None = None) -> dict:
    cluster_of = {str(q["id"]): q.get("cluster") or "?" for q in (exam or [])}
    right = wrong = unanswered = unkeyed = 0
    by_sub: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0])
    misses = []
    for a in answers:
        qid = str(a["id"])
        given = (a.get("answer") or "").upper()
        sub = cluster_of.get(qid, "?")[:2]
        if qid not in key:
            unkeyed += 1
            continue
        by_sub[sub][1] += 1
        if not given:
            unanswered += 1
            misses.append((qid, sub, given, key[qid]))
        elif given == key[qid]:
            right += 1
            by_sub[sub][0] += 1
        else:
            wrong += 1
            misses.append((qid, sub, given, key[qid]))
    total = right + wrong + unanswered
    return {"right": right, "wrong": wrong, "unanswered": unanswered,
            "unkeyed": unkeyed, "total": total,
            "pct": (100.0 * right / total) if total else 0.0,
            "passed": right >= PASS_MARK,
            "by_sub": {k: tuple(v) for k, v in sorted(by_sub.items())},
            "misses": misses}


def format_score(s: dict, label: str = "") -> str:
    head = (f"{label + ': ' if label else ''}{s['right']}/{s['total']} "
            f"({s['pct']:.0f}%) {'PASS' if s['passed'] else 'FAIL'}"
            f" — wrong {s['wrong']}, blank {s['unanswered']}"
            + (f", unkeyed {s['unkeyed']}" if s['unkeyed'] else ""))
    subs = "  ".join(f"{k}:{r}/{n}" for k, (r, n) in s["by_sub"].items())
    lines = [head, "  " + subs]
    if s["misses"]:
        lines.append("  misses: " + ", ".join(
            f"{qid}({sub}) {g or '-'}≠{c}" for qid, sub, g, c in s["misses"]))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--key", type=Path, required=True)
    ap.add_argument("--exam", type=Path, default=None,
                    help="Exam json, for sub-element breakdown")
    ap.add_argument("answers", type=Path, nargs="+")
    args = ap.parse_args()
    key = load_key(args.key)
    exam = None
    if args.exam:
        from . import hamexam_batch as hb
        exam = hb.load_exam(args.exam)
    scores = []
    for p in args.answers:
        s = score(load_answers(p), key, exam)
        scores.append(s["right"])
        print(format_score(s, p.stem))
    if len(scores) > 1:
        mean = sum(scores) / len(scores)
        sd = (sum((x - mean) ** 2 for x in scores) / (len(scores) - 1)) ** 0.5
        print(f"\n{len(scores)} runs: mean {mean:.1f}  sd {sd:.1f}  "
              f"min {min(scores)}  max {max(scores)}  "
              f"pass rate {sum(x >= PASS_MARK for x in scores)}/{len(scores)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
