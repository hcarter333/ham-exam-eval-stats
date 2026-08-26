#!/usr/bin/env python3
"""
Generate Extra-class practice exams and answer keys straight from the pool,
replicating projecttoucans.com's buildExam(): one random question from each
of the 50 E-groups, sorted by subelement then group letter. No browser.

    python3 hamexam_make.py --count 10                     # 10 random exams
    python3 hamexam_make.py --cover                        # enough exams to hit every pool question
    python3 hamexam_make.py --count 10 --seed 42 --jobs exam_jobs_10.json --replicas 3

Writes exams/<prefix>NN.json in the console scraper's shape (number, cluster,
id, header, question, choices) and keys/<prefix>NN.key.json as a list of
{"id","answer"} — both exactly what hamexam_fleet.py already reads. With
--jobs it also writes a fleet manifest listing every generated exam (one job
line per exam per configuration in --config, a JSON list of overrides).

The pool is fetched from the site's GitHub raw URL by default and cached to
extra_pool.json; --pool points at a local copy instead.

--cover: instead of independent draws, each group's questions are shuffled
once and dealt round-robin across exams, so with N = the largest group size
(15 as of this pool) every question appears at least once and none more than
twice. Numbering is deterministic under --seed.
"""

import argparse
import json
import random
import re
import sys
import urllib.request
from pathlib import Path

POOL_URL = ("https://raw.githubusercontent.com/hcarter333/"
            "project-toucans-ham-exam-prep/refs/heads/main/extra_pool.json")
_WS = re.compile(r"\s+")


def sanitize(t) -> str:
    return _WS.sub(" ", str(t or "")).strip()


def load_pool(pool_path: Path | None) -> list[dict]:
    cache = Path("extra_pool.json")
    if pool_path is None:
        if cache.exists():
            pool_path = cache
        else:
            print(f"fetching {POOL_URL}")
            with urllib.request.urlopen(POOL_URL, timeout=60) as r:
                data = r.read()
            cache.write_bytes(data)
            pool_path = cache
    return json.loads(pool_path.read_text(encoding="utf-8-sig"))


def group_key(q: dict) -> str:
    return f"E{q['subelement']}{q['group_index']}"


def sort_key(q: dict):
    return (int(q["subelement"]), q["group_index"])


def bucket(pool: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for q in pool:
        groups.setdefault(group_key(q), []).append(q)
    return groups


def to_scraper_shape(chosen: list[dict]) -> tuple[list[dict], list[dict]]:
    """(exam questions in the console dump's shape, key list)."""
    chosen = sorted(chosen, key=sort_key)
    exam, key = [], []
    for n, q in enumerate(chosen, 1):
        cluster = group_key(q)
        qid = str(q["id"])
        choices = [f"{L}. {sanitize(q[f'answer_{L.lower()}'])}" for L in "ABCD"
                   if sanitize(q.get(f"answer_{L.lower()}"))]
        exam.append({"number": n, "cluster": cluster, "id": qid,
                     "header": f"{cluster}{q['group_number']} (id {qid})",
                     "question": sanitize(q["question"]), "choices": choices})
        key.append({"id": qid, "answer": sanitize(q["answer"]).upper()[:1]})
    return exam, key


def random_exams(groups: dict, count: int, rng: random.Random) -> list[list[dict]]:
    return [[rng.choice(qs) for qs in groups.values()] for _ in range(count)]


def cover_exams(groups: dict, rng: random.Random, count: int | None) -> list[list[dict]]:
    n = count or max(len(qs) for qs in groups.values())
    decks = {g: rng.sample(qs, len(qs)) for g, qs in groups.items()}
    return [[decks[g][i % len(decks[g])] for g in groups] for i in range(n)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--count", type=int, default=None,
                    help="Number of exams (default 1; with --cover, default = "
                         "largest group size)")
    ap.add_argument("--cover", action="store_true",
                    help="Deal each group's questions round-robin so the set "
                         "covers the whole pool")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--pool", type=Path, default=None, help="Local extra_pool.json")
    ap.add_argument("--prefix", default="Test",
                    help="File prefix; exams are <prefix>NN (default Test)")
    ap.add_argument("--start", type=int, default=None,
                    help="First NN (default: one past the highest existing "
                         "<prefix>NN in exams/)")
    ap.add_argument("--exams-dir", type=Path, default=Path("exams"))
    ap.add_argument("--keys-dir", type=Path, default=Path("keys"))
    ap.add_argument("--jobs", type=Path, default=None,
                    help="Also write a fleet manifest covering the generated exams")
    ap.add_argument("--jobs-only", action="store_true",
                    help="Generate nothing; write --jobs over the existing "
                         "<prefix>NN exams in --exams-dir")
    ap.add_argument("--first", default=None,
                    help="Derive exams from the existing <prefix>NN set with pool "
                         "question ID placed first, replacing that exam's own draw "
                         "from the same group. Writes <prefix>NN_first<ID>.json and "
                         "keys; with --jobs, the manifest tags them first<ID>")
    ap.add_argument("--replicas", type=int, default=3, help="Replicas per job line")
    ap.add_argument("--config", default='[{}]',
                    help="JSON list of per-job overrides applied to every exam, "
                         "e.g. '[{}, {\"thinking\": true, \"effort\": \"high\", "
                         "\"suffix\": \"think\"}]' (suffix names the output)")
    ap.add_argument("--defaults", default=None,
                    help="JSON object for the manifest's defaults block")
    args = ap.parse_args()

    if args.first is not None:
        return derive_first(args)

    if args.jobs_only:
        if not args.jobs:
            print("error: --jobs-only needs --jobs", file=sys.stderr)
            return 1
        pat = re.compile(rf"^{re.escape(args.prefix)}(\d+)\.json$")
        exams = sorted((p for p in args.exams_dir.glob(f"{args.prefix}*.json")
                        if pat.match(p.name)), key=lambda p: int(pat.match(p.name).group(1)))
        written = [(ep, args.keys_dir / f"{ep.stem}.key.json") for ep in exams]
        missing = [kp for _, kp in written if not kp.exists()]
        if missing:
            print(f"error: missing keys: {missing[:3]}...", file=sys.stderr)
            return 1
        print(f"{len(written)} existing exam(s) under {args.exams_dir}")
        write_manifest(args, written)
        return 0

    rng = random.Random(args.seed)
    pool = load_pool(args.pool)
    groups = dict(sorted(bucket(pool).items(),
                         key=lambda kv: sort_key(kv[1][0])))
    print(f"pool: {len(pool)} questions, {len(groups)} groups "
          f"(sizes {min(map(len, groups.values()))}-{max(map(len, groups.values()))})")

    draws = (cover_exams(groups, rng, args.count) if args.cover
             else random_exams(groups, args.count or 1, rng))

    args.exams_dir.mkdir(parents=True, exist_ok=True)
    args.keys_dir.mkdir(parents=True, exist_ok=True)
    if args.start is None:
        pat = re.compile(rf"^{re.escape(args.prefix)}(\d+)\.json$")
        nums = [int(m.group(1)) for p in args.exams_dir.glob(f"{args.prefix}*.json")
                if (m := pat.match(p.name))]
        args.start = (max(nums) + 1) if nums else 1

    written = []
    seen: set[str] = set()
    for i, chosen in enumerate(draws):
        exam, key = to_scraper_shape(chosen)
        name = f"{args.prefix}{args.start + i:02d}"
        ep = args.exams_dir / f"{name}.json"
        kp = args.keys_dir / f"{name}.key.json"
        ep.write_text(json.dumps(exam, indent=2, ensure_ascii=False))
        kp.write_text(json.dumps(key, indent=2))
        seen.update(q["id"] for q in exam)
        written.append((ep, kp))
        print(f"  {ep}  {kp}")
    print(f"{len(written)} exam(s); {len(seen)}/{len(pool)} distinct pool "
          f"questions covered")

    if args.jobs:
        write_manifest(args, written)
    return 0


def existing_exams(args):
    pat = re.compile(rf"^{re.escape(args.prefix)}(\d+)\.json$")
    return sorted((p for p in args.exams_dir.glob(f"{args.prefix}*.json")
                   if pat.match(p.name)), key=lambda p: int(pat.match(p.name).group(1)))


def derive_first(args):
    """<prefix>NN_first<ID>: question ID at position 1, its group's original
    draw removed, the other 49 unchanged and in the original order. Still
    one question per group, still 50 long."""
    pool = load_pool(args.pool)
    q = next((x for x in pool if str(x["id"]) == str(args.first)), None)
    if q is None:
        print(f"error: question {args.first} not in pool", file=sys.stderr)
        return 1
    first_exam, first_key = to_scraper_shape([q])
    first_q, first_k = first_exam[0], first_key[0]
    group = first_q["cluster"]
    tag = f"first{args.first}"
    written = []
    for ep in existing_exams(args):
        exam = json.loads(ep.read_text(encoding="utf-8-sig"))
        key = {str(k["id"]): k["answer"] for k in
               json.loads((args.keys_dir / f"{ep.stem}.key.json").read_text(encoding="utf-8-sig"))}
        rest = [x for x in exam if x["cluster"] != group]
        merged = [dict(first_q)] + [dict(x) for x in rest]
        for n, x in enumerate(merged, 1):
            x["number"] = n
        name = f"{ep.stem}_{tag}"
        out_e = args.exams_dir / f"{name}.json"
        out_k = args.keys_dir / f"{name}.key.json"
        out_e.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        out_k.write_text(json.dumps([first_k] + [{"id": x["id"], "answer": key[x["id"]]}
                                                 for x in rest], indent=2))
        written.append((out_e, out_k))
        print(f"  {out_e}  (dropped {[x['id'] for x in exam if x['cluster'] == group]})")
    print(f"{len(written)} derived exam(s), question {args.first} ({group}) first")
    if args.jobs:
        write_manifest(args, written, tag=tag)
    return 0


def write_manifest(args, written, tag=None):
    if True:
        configs = json.loads(args.config)
        defaults = json.loads(args.defaults) if args.defaults else {
            "model": "claude-sonnet-4-6", "max_tokens": 16000,
            "figures": ["figures/E5_E6.png", "figures/E7_E9-1.png",
                        "figures/E9-2_E9-3.png"],
            "replicas": args.replicas}
        jobs = []
        for ep, kp in written:
            for cfg in configs:
                cfg = dict(cfg)
                suffix = cfg.pop("suffix", None)
                job = {"exam": str(ep), "key": str(kp)}
                if tag:
                    job["tag"] = tag
                if suffix:
                    job["output"] = f"answers/{ep.stem}_{suffix}.answers.json"
                job.update(cfg)
                jobs.append(job)
        args.jobs.write_text(json.dumps({"defaults": defaults, "jobs": jobs}, indent=2))
        print(f"wrote {args.jobs}: {len(jobs)} job line(s) x {args.replicas} replicas "
              f"= {len(jobs) * args.replicas} requests")


if __name__ == "__main__":
    sys.exit(main())
