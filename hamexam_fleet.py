#!/usr/bin/env python3
"""
Ham exam fleet — many exams / many parameter arms / N replicas, one batch.

Sibling of polecat_fleet.py. Because the exam prompt forbids tools, every
request is a single turn: there is no pause_turn, no continuation round, no
container, no context editing. The fleet is therefore one submit + one poll,
and its job is parameter sweeps and replicas (variance floors per model,
thinking on/off, effort levels, temperature settings), plus scoring.

    export ANTHROPIC_API_KEY=sk-ant-...
    python3 hamexam_fleet.py --jobs exam_jobs.json

Jobs manifest:

{
  "defaults": {
    "model": "claude-sonnet-4-6",
    "max_tokens": 16000,
    "temperature": null,             // null = API default
    "thinking": false,
    "effort": null,                  // low|medium|high|xhigh|max (implies thinking)
    "figures": ["figures/E5_E6.png", "figures/E7_E9-1.png", "figures/E9-2_E9-3.png"],
    "key": null,                     // answer key json -> scores go in the db
    "replicas": 1
  },
  "jobs": [
    { "exam": "exams/FirstTest.json", "key": "keys/FirstTest.key.json",
      "replicas": 10 },                                        // variance floor
    { "exam": "exams/FirstTest.json", "key": "keys/FirstTest.key.json",
      "output": "answers/FirstTest_think.answers.json",
      "thinking": true, "effort": "high", "replicas": 10 },    // thinking variant
    { "exam": "exams/FirstTest.json", "key": "keys/FirstTest.key.json",
      "output": "answers/FirstTest_opus.answers.json",
      "model": "claude-opus-4-8", "replicas": 5 }
  ]
}

Trajectory ids follow the polecat scheme: <jobs stem>[_<prompt stem>]_<output
stem>_rN, with rN continued past whatever the transcript db already holds,
so re-running an identical manifest appends replicas rather than colliding.
Every result goes into a sqlite db (--db, default hamexam_evals.db — keep it
separate from polecat_evals.db). Tables:
    runs      one row per trajectory: configuration, usage, parse status,
              score, full response text and the raw message json
    answers   one row per (trajectory, question): cluster, given, correct,
              right — the long table the clustered-SE stats read from
and, when polecat_exhume is importable, the same transcripts table the
polecat fleet writes, so the two dbs can be browsed with the same datasette
habits. Replica numbering continues from the runs table.

State: <jobs file>.state.json — re-run the same command to resume/re-attach.
"""

import argparse
import json
import re
import sqlite3
import sys
import time
from pathlib import Path

import anthropic

import hamexam_batch as hb
import hamexam_score as hs

try:
    import polecat_exhume as px      # shared transcript schema, if present
except ImportError:                  # pragma: no cover
    px = None

JOB_DEFAULTS = {
    "model": hb.DEFAULT_MODEL,
    "max_tokens": 16000,
    "temperature": None,
    "thinking": False,
    "effort": None,
    "figures": [str(p) for p in hb.DEFAULT_FIGURES],
    "key": None,
    "replicas": 1,
    "output": None,
    "tag": None,                        # free label stored with the run; the stats
                                        # treat it as part of the configuration
}

_ID_BAD = re.compile(r"[^a-zA-Z0-9_-]")
TRAJ_BASE_MAX = 54
_R_SUFFIX = re.compile(r"_r(\d+)$")


def sanitize_id(s: str) -> str:
    return _ID_BAD.sub("-", s) or "job"


def trajectory_base(jobs_stem: str, prompt, out_stem: str) -> str:
    parts = [sanitize_id(jobs_stem)]
    if prompt:
        parts.append(sanitize_id(Path(prompt).stem))
    parts.append(sanitize_id(out_stem))
    base = "_".join(parts)
    if len(base) > TRAJ_BASE_MAX:
        short = base[:TRAJ_BASE_MAX]
        print(f"warning: trajectory base {base!r} truncated to {short!r}",
              file=sys.stderr)
        base = short
    return base


SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    trajectory_id     TEXT PRIMARY KEY,
    jobs_file         TEXT,
    exam              TEXT,
    key               TEXT,
    prompt            TEXT,
    model             TEXT,
    model_served      TEXT,
    temperature       REAL,
    thinking          INTEGER,
    effort            TEXT,
    figures_json      TEXT,
    tag               TEXT,
    batch_id          TEXT,
    batch_created_at  TEXT,
    finished_at       TEXT,
    result_type       TEXT,
    error_json        TEXT,
    stop_reason       TEXT,
    parse             TEXT,
    questions         INTEGER,
    answered          INTEGER,
    right             INTEGER,
    wrong             INTEGER,
    blank             INTEGER,
    pct               REAL,
    passed            INTEGER,
    input_tokens      INTEGER,
    output_tokens     INTEGER,
    cache_read_tokens INTEGER,
    cache_create_tokens INTEGER,
    response_text     TEXT,
    message_json      TEXT
);
CREATE TABLE IF NOT EXISTS answers (
    trajectory_id TEXT NOT NULL,
    question_id   TEXT NOT NULL,
    number        INTEGER,
    cluster       TEXT,
    subelement    TEXT,
    given         TEXT,
    correct       TEXT,
    right         INTEGER,
    PRIMARY KEY (trajectory_id, question_id)
);
CREATE INDEX IF NOT EXISTS answers_q ON answers(question_id);
CREATE INDEX IF NOT EXISTS answers_cluster ON answers(cluster);
"""


def open_db(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.executescript(SCHEMA)
    cols = {r[1] for r in db.execute("PRAGMA table_info(runs)")}
    if "tag" not in cols:                    # migrate dbs created before tags
        db.execute("ALTER TABLE runs ADD COLUMN tag TEXT")
    if "model_served" not in cols:
        db.execute("ALTER TABLE runs ADD COLUMN model_served TEXT")
        db.execute("UPDATE runs SET model_served = json_extract(message_json, '$.model') "
                   "WHERE message_json IS NOT NULL")
    if px is not None:
        db.executescript(px.SCHEMA)
    return db


def next_replica(db: sqlite3.Connection, base: str, floor: int) -> int:
    """One past the largest _rN the runs table already holds for this
    identity, and past anything allocated earlier in this expansion."""
    best = floor
    for (tid,) in db.execute(
            "SELECT trajectory_id FROM runs "
            "WHERE trajectory_id LIKE ? || '\\_r%' ESCAPE '\\'", (base,)):
        m = _R_SUFFIX.fullmatch(tid[len(base):])
        if m:
            best = max(best, int(m.group(1)))
    return best + 1


def build_job_params(job: dict, prefix_cache: dict, prompt_text: str | None) -> dict:
    """One job -> Messages API params. The static block (prompt + figures) is
    built once per distinct figure set and shared across jobs so the cache
    breakpoint lands on identical bytes."""
    figs = tuple(str(f) for f in job["figures"])
    if figs not in prefix_cache:
        prefix_cache[figs] = hb.static_blocks(prompt_text or hb.EXAM_PROMPT,
                                              [Path(f) for f in figs])
    content = list(prefix_cache[figs]) + [hb.exam_block(Path(job["exam"]))]
    params = {"model": job["model"], "max_tokens": job["max_tokens"],
              "messages": [{"role": "user", "content": content}]}
    thinking = job["thinking"] or bool(job["effort"])
    if thinking:
        if job["temperature"] not in (None, 1.0):
            raise ValueError(f"job {job['exam']}: thinking requires temperature "
                             f"unset/1.0, got {job['temperature']}")
        params["thinking"] = {"type": "adaptive", "display": "summarized"}
        if job["effort"]:
            params["output_config"] = {"effort": job["effort"]}
    elif job["temperature"] is not None:
        params["temperature"] = job["temperature"]
    return params


def expand_jobs(manifest: dict, jobs_stem: str, db: sqlite3.Connection,
                prompt_path) -> dict[str, dict]:
    defaults = {**JOB_DEFAULTS, **manifest.get("defaults", {})}
    prompt_text = hb.load_prompt(Path(prompt_path)) if prompt_path else None
    prefix_cache: dict = {}
    alloc: dict[str, int] = {}
    records: dict[str, dict] = {}
    for raw in manifest["jobs"]:
        job = {**defaults, **raw}
        if "exam" not in raw:
            raise ValueError(f"job missing 'exam': {raw}")
        exam_path = Path(job["exam"])
        if not exam_path.exists():
            raise ValueError(f"exam json not found: {exam_path}")
        if job["key"] and not Path(job["key"]).exists():
            raise ValueError(f"key not found: {job['key']}")
        base_out = Path(job["output"]) if job["output"] else (
            Path("answers") / f"{exam_path.stem}.answers.json")
        # ".answers.json" is a two-part suffix; keep the stem before it.
        out_stem = base_out.name[:-len(".answers.json")] \
            if base_out.name.endswith(".answers.json") else base_out.stem
        base = trajectory_base(jobs_stem, prompt_path, out_stem)
        n = int(job["replicas"])
        start = next_replica(db, base, alloc.get(base, 0))
        alloc[base] = start + n - 1
        params = build_job_params(job, prefix_cache, prompt_text)
        for r in range(start, start + n):
            out = base_out.with_name(f"{out_stem}_r{r}.answers.json")
            tid = f"{base}_r{r}"
            records[tid] = {
                "cid": sanitize_id(f"{out_stem}_r{r}"),
                "exam": str(exam_path),
                "key": job["key"],
                "output": str(out),
                "job": {**{k: job[k] for k in
                           ("model", "temperature", "thinking", "effort")},
                        "figures": list(job["figures"]),
                        "tag": job["tag"],
                        "prompt": str(prompt_path) if prompt_path else None},
                "params": params,
                "status": "pending",        # pending|done|failed
                "score": None,
            }
        if start > 1:
            print(f"{base}: continuing replicas at r{start}")
    return records


def record_run(db: sqlite3.Connection, tid: str, rec: dict, jobs_file: str,
               result, batch_id: str, created_iso: str,
               ar: dict | None = None, exam: list[dict] | None = None,
               text: str = ""):
    """Upsert the runs row (and answers rows when the run succeeded and was
    parsed). Keyed on trajectory_id, so re-processing a batch is idempotent.
    Also mirrors into polecat_exhume's transcripts table when available."""
    j = rec["job"]
    row = {
        "trajectory_id": tid, "jobs_file": jobs_file, "exam": rec["exam"],
        "key": rec["key"], "prompt": j["prompt"], "model": j["model"],
        "temperature": j["temperature"], "thinking": int(bool(j["thinking"])),
        "effort": j["effort"], "figures_json": json.dumps(j["figures"]),
        "tag": j.get("tag"),
        "batch_id": batch_id, "batch_created_at": created_iso,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "result_type": result.result.type,
    }
    if result.result.type == "succeeded":
        msg = result.result.message
        row["message_json"] = json.dumps(msg.model_dump(exclude_none=True))
        row["response_text"] = text
        row["stop_reason"] = msg.stop_reason
        row["model_served"] = getattr(msg, "model", None)   # what the API actually ran
        if ar is not None:
            u = ar["usage"]
            s = ar.get("score") or {}
            row.update({
                "parse": ar["parse"], "questions": len(exam),
                "answered": len(exam) - len(ar["missing_ids"]),
                "right": s.get("right"), "wrong": s.get("wrong"),
                "blank": s.get("unanswered"), "pct": s.get("pct"),
                "passed": None if not s else int(s["passed"]),
                "input_tokens": u["input"], "output_tokens": u["output"],
                "cache_read_tokens": u["cache_read"],
                "cache_create_tokens": u["cache_create"],
            })
    else:
        err = getattr(result.result, "error", None)
        row["error_json"] = json.dumps(
            err.model_dump() if hasattr(err, "model_dump") else str(err))

    cols = ", ".join(row)
    marks = ", ".join("?" for _ in row)
    updates = ", ".join(f"{c}=excluded.{c}" for c in row if c != "trajectory_id")
    db.execute(f"INSERT INTO runs ({cols}) VALUES ({marks}) "
               f"ON CONFLICT(trajectory_id) DO UPDATE SET {updates}",
               list(row.values()))

    if ar is not None and exam is not None:
        key = ar.get("_key") or {}
        given = {a["id"]: a["answer"] for a in ar["answers"]}
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

    if px is not None:
        trow = {"custom_id": result.custom_id, "cid": rec["cid"], "turn": 0,
                "trajectory_id": tid, "batch_id": batch_id,
                "batch_created_at": created_iso,
                "result_type": result.result.type}
        if result.result.type == "succeeded":
            trow.update(px.message_row(result.result.message.model_dump()))
        else:
            trow["error_json"] = row["error_json"]
        px.upsert(db, trow)


def finish_job(tid: str, rec: dict, message, key_cache: dict):
    """Parse, score, write the answers file. Returns (answers_record, exam, text)."""
    exam = hb.load_exam(Path(rec["exam"]))
    text = hb.message_text(message)
    answers, how = hb.extract_answers(text)
    ar = hb.answers_record(answers, how, exam, message)
    ar["trajectory_id"] = tid
    ar["job"] = rec["job"]
    line = f"  {tid}: {len(answers)}/{len(exam)} answered ({how})"
    if rec["key"]:
        if rec["key"] not in key_cache:
            key_cache[rec["key"]] = hs.load_key(Path(rec["key"]))
        key = key_cache[rec["key"]]
        s = hs.score(ar["answers"], key, exam)
        rec["score"] = {k: s[k] for k in
                        ("right", "wrong", "unanswered", "total", "pct", "passed")}
        ar["score"] = rec["score"]
        ar["misses"] = s["misses"]
        ar["_key"] = key                        # for the answers table; not written
        line += (f"  score {s['right']}/{s['total']} "
                 f"{'PASS' if s['passed'] else 'FAIL'}")
    out = Path(rec["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({k: v for k, v in ar.items() if k != "_key"}, indent=1))
    out.with_suffix(".raw.txt").write_text(text)
    rec["status"] = "done"
    print(line)
    return ar, exam, text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--prompt", type=Path, default=None,
                    help="Exam prompt file for the whole fleet (its stem joins "
                         "the trajectory id, like --bead-prompt)")
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--db", type=Path, default=Path("hamexam_evals.db"),
                    help="Sqlite db receiving runs + per-question answers "
                         "(and polecat_exhume transcripts when importable)")
    args = ap.parse_args()

    state_path = args.jobs.with_suffix(args.jobs.suffix + ".state.json")
    db = open_db(args.db)

    state = None if args.fresh else (
        json.loads(state_path.read_text()) if state_path.exists() else None)
    if state is None:
        manifest = json.loads(args.jobs.read_text(encoding="utf-8-sig"))
        records = expand_jobs(manifest, args.jobs.stem, db, args.prompt)
        state = {"batch_id": None, "records": records}
        print(f"built {len(records)} request(s) from {args.jobs}")
    else:
        records = state["records"]
        print(f"resumed {state_path}: batch {state['batch_id']}, "
              f"{sum(1 for r in records.values() if r['status'] == 'pending')} "
              f"job(s) unfinished")

    client = anthropic.Anthropic()
    live = {tid: rec for tid, rec in records.items() if rec["status"] == "pending"}
    if not live:
        print("nothing to do")
        return 0

    if state["batch_id"] is None:
        requests = [{"custom_id": f"{tid}-t0", "params": rec["params"]}
                    for tid, rec in live.items()]
        batch = client.messages.batches.create(requests=requests)
        state["batch_id"] = batch.id
        state_path.write_text(json.dumps(state))
        print(f"submitted batch {batch.id} ({len(requests)} request(s))")

    batch = hb.wait_for_batch(client, state["batch_id"], args.poll_seconds)
    created = batch.created_at
    created_iso = created.isoformat() if hasattr(created, "isoformat") else str(created)

    key_cache: dict = {}
    jobs_file = str(args.jobs)
    for result in client.messages.batches.results(state["batch_id"]):
        tid = result.custom_id.rsplit("-t", 1)[0]
        rec = records.get(tid)
        if rec is None or rec["status"] != "pending":
            continue
        if result.result.type != "succeeded":
            print(f"  {tid}: {result.result.type} "
                  f"{getattr(result.result, 'error', '')}", file=sys.stderr)
            rec["status"] = "failed"
            record_run(db, tid, rec, jobs_file, result, state["batch_id"], created_iso)
            continue
        ar, exam, text = finish_job(tid, rec, result.result.message, key_cache)
        record_run(db, tid, rec, jobs_file, result, state["batch_id"], created_iso,
                   ar, exam, text)
        db.commit()
    db.commit()
    db.close()

    # Params are large (base64 figures); don't keep them in the finished state.
    for rec in records.values():
        rec.pop("params", None)
    done = [r for r in records.values() if r["status"] == "done"]
    failed = [t for t, r in records.items() if r["status"] == "failed"]
    scored = [r["score"]["right"] for r in done if r["score"]]
    print(f"\nfleet done: {len(done)} ok, {len(failed)} failed")
    if len(scored) > 1:
        mean = sum(scored) / len(scored)
        sd = (sum((x - mean) ** 2 for x in scored) / (len(scored) - 1)) ** 0.5
        print(f"scores: mean {mean:.1f}  sd {sd:.1f}  min {min(scored)}  "
              f"max {max(scored)}  pass {sum(x >= hs.PASS_MARK for x in scored)}"
              f"/{len(scored)}")
    for t in failed:
        print(f"  failed: {t}", file=sys.stderr)
    # The batch is fully processed either way; failed jobs are in the runs table,
    # and re-running the manifest will allocate fresh replica numbers.
    if state_path.exists():
        state_path.unlink()
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
