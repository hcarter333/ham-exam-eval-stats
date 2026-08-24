#!/usr/bin/env python3
"""
Backfill a pre-sqlite fleet run (stats .csv + answers/*.answers.json) into
hamexam_evals.db, using the same runs/answers tables hamexam_fleet.py writes.

    python3 hamexam_backfill.py --csv exam_jobs.stats.csv \\
        --key keys/FirstTest.key.json [--answers-dir answers] [--db hamexam_evals.db]

Each csv row names a trajectory_id like exam_jobs_FirstTest_think_r3; the
answers file for it is <answers-dir>/FirstTest_think_r3.answers.json (the
trajectory id minus the jobs-file stem prefix). The answers file supplies the
per-question data, usage, and score; the csv supplies finished_at and the
rows for failed jobs that have no answers file. The .raw.txt beside each
answers file, if present, becomes response_text. Batch id is not recorded in
either artifact, so it's left null. Idempotent: re-running upserts.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import hamexam_batch as hb
import hamexam_fleet as hf
import hamexam_score as hs


def none_if_blank(v):
    return None if v in ("", None) else v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--key", type=Path, default=None,
                    help="Answer key; if omitted, uses the csv's score columns and "
                         "leaves answers.correct null")
    ap.add_argument("--answers-dir", type=Path, default=Path("answers"))
    ap.add_argument("--db", type=Path, default=Path("hamexam_evals.db"))
    args = ap.parse_args()

    jobs_stem = args.csv.name.split(".")[0]          # exam_jobs.stats.csv -> exam_jobs
    key = hs.load_key(args.key) if args.key else {}
    db = hf.open_db(args.db)
    exam_cache: dict[str, list[dict]] = {}
    n_runs = n_ans = 0

    with args.csv.open(newline="") as f:
        for row in csv.DictReader(f):
            tid = row["trajectory_id"]
            out_stem = tid[len(jobs_stem) + 1:] if tid.startswith(jobs_stem + "_") else tid
            ans_path = args.answers_dir / f"{out_stem}.answers.json"
            raw_path = ans_path.with_suffix(".raw.txt")

            ar = json.loads(ans_path.read_text()) if ans_path.exists() else None
            job = (ar or {}).get("job") or {}
            run = {
                "trajectory_id": tid, "jobs_file": f"{jobs_stem}.json",
                "exam": row["exam"], "key": str(args.key) if args.key else None,
                "prompt": none_if_blank(row.get("prompt")) or job.get("prompt"),
                "model": row["model"],
                "temperature": (None if row["temperature"] in ("", "None")
                                else float(row["temperature"])),
                "thinking": int(row["thinking"] == "True"),
                "effort": (None if row["effort"] in ("", "None") else row["effort"]),
                "figures_json": json.dumps(job.get("figures")) if job else None,
                "batch_id": None, "batch_created_at": None,
                "finished_at": row["timestamp"],
                "result_type": "succeeded" if row["status"] == "done" else "errored",
                "stop_reason": none_if_blank(row.get("stop_reason")),
                "parse": none_if_blank(row.get("parse")),
                "questions": none_if_blank(row.get("questions")),
                "answered": none_if_blank(row.get("answered")),
                "right": none_if_blank(row.get("right")),
                "wrong": none_if_blank(row.get("wrong")),
                "blank": none_if_blank(row.get("blank")),
                "pct": none_if_blank(row.get("pct")),
                "passed": (None if row.get("passed") in ("", None)
                           else int(row["passed"] == "True")),
                "input_tokens": none_if_blank(row.get("input_tokens")),
                "output_tokens": none_if_blank(row.get("output_tokens")),
                "cache_read_tokens": none_if_blank(row.get("cache_read_tokens")),
                "cache_create_tokens": none_if_blank(row.get("cache_create_tokens")),
                "response_text": raw_path.read_text() if raw_path.exists() else None,
                "message_json": None,
            }

            if ar is not None:
                exam_path = row["exam"]
                if exam_path not in exam_cache:
                    exam_cache[exam_path] = hb.load_exam(Path(exam_path))
                exam = exam_cache[exam_path]
                if key:      # rescore from the answers so the db is self-consistent
                    s = hs.score(ar["answers"], key, exam)
                    run.update({"right": s["right"], "wrong": s["wrong"],
                                "blank": s["unanswered"], "pct": s["pct"],
                                "passed": int(s["passed"]), "questions": len(exam),
                                "answered": len(exam) - len(ar.get("missing_ids", []))})
                given = {str(a["id"]): a["answer"] for a in ar["answers"]}
                db.execute("DELETE FROM answers WHERE trajectory_id=?", (tid,))
                db.executemany(
                    "INSERT INTO answers (trajectory_id, question_id, number, cluster, "
                    "subelement, given, correct, right) VALUES (?,?,?,?,?,?,?,?)",
                    [(tid, str(q["id"]), q.get("number"), q.get("cluster"),
                      (q.get("cluster") or "")[:2], given.get(str(q["id"])),
                      key.get(str(q["id"])),
                      None if str(q["id"]) not in key
                      else int(given.get(str(q["id"])) == key[str(q["id"])]))
                     for q in exam])
                n_ans += len(exam)
            else:
                print(f"note: no answers file for {tid} ({ans_path}); "
                      f"runs row only", file=sys.stderr)

            cols = ", ".join(run)
            marks = ", ".join("?" for _ in run)
            updates = ", ".join(f"{c}=excluded.{c}" for c in run if c != "trajectory_id")
            db.execute(f"INSERT INTO runs ({cols}) VALUES ({marks}) "
                       f"ON CONFLICT(trajectory_id) DO UPDATE SET {updates}",
                       list(run.values()))
            n_runs += 1

    db.commit()
    db.close()
    print(f"backfilled {n_runs} run(s), {n_ans} answer row(s) into {args.db}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
