#!/usr/bin/env python3
"""
Ham-exam-in-a-batch — one Extra-class exam, no tools, one answers .json.

Sibling of polecat_batch.py. The eval prompt forbids the internet and any
scripting, so this request carries NO tools: no web_search/web_fetch, no code
execution, and therefore no pause_turn, no container, no context editing.
One batch, one turn, done. The model's final message is a JSON array of
{"id","answer"} objects, which this script extracts and writes to disk.

    export ANTHROPIC_API_KEY=sk-ant-...
    python3 hamexam_batch.py --exam exams/FirstTest.json \\
        --figure figures/E5_E6.png --figure figures/E7_E9-1.png \\
        --figure figures/E9-2_E9-3.png

Outputs <answers_dir>/<exam stem>.answers.json (default answers_dir: ./answers),
plus a .raw.txt of the full response. Optional --key <key.json> scores the
answers immediately (see hamexam_score.py for the key format).

Message layout (two content blocks in one user turn, cache breakpoint between):
    [exam prompt + figure images]  <- identical across exams -> cached
    [exam json]                     <- varies per exam
"""

import argparse
import base64
import json
import mimetypes
import re
import sys
from pathlib import Path

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"

# Default figure set: the three screenshots of the Extra pool figures
# (E5-1/E6-1/E6-2/E6-3, E7-1/E7-2/E7-3/E9-1, E9-2/E9-3). Override per run
# with --figure, or per job in the fleet manifest.
DEFAULT_FIGURES = [
    Path("figures/E5_E6.png"),
    Path("figures/E7_E9-1.png"),
    Path("figures/E9-2_E9-3.png"),
]

# ---- The exam prompt, verbatim from the human-run version -------------------
EXAM_PROMPT = """This is a ham radio extra class exam. You may not use the interent to complete it, only what you already know, (what's in your model.) Do not use memories of any other chats. The questions are attached in json format. Remember, do not use the internet; do not use the memories of any other chats. You are also not allowed to use scripting beyond simple calculator math. Each question has a multiple choice answer. Choose the answer that fits best. Output your answers in json format, one answer per question. Include the question id with the answer for each question. The exam and associated figure are attached. Write your answers in JSON like
[
{"id":"599","answer":"A"},
{"id":"9","answer":"D"},
{"id":"16","answer":"C"},
{"id":"33","answer":"A"}, and so on
"""


def load_prompt(prompt_path: Path | None) -> str:
    """EXAM_PROMPT by default, or the contents of a prompt file (the
    --bead-prompt equivalent — swap wording without editing this file)."""
    if prompt_path is None:
        return EXAM_PROMPT
    if not prompt_path.exists():
        print(f"error: prompt file not found: {prompt_path}", file=sys.stderr)
        sys.exit(1)
    return prompt_path.read_text()


def image_block(path: Path) -> dict:
    media, _ = mimetypes.guess_type(path.name)
    if media not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        raise ValueError(f"unsupported figure type for {path}: {media}")
    data = base64.standard_b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image",
            "source": {"type": "base64", "media_type": media, "data": data}}


def load_exam(exam_path: Path) -> list[dict]:
    """The scraper dumps a list of {number, cluster, id, header, question,
    choices}. utf-8-sig because the Windows console save leaves a BOM."""
    data = json.loads(exam_path.read_text(encoding="utf-8-sig", errors="replace"))
    if isinstance(data, dict) and "questions" in data:   # window.lastHamExam shape
        data = data["questions"]
    if not isinstance(data, list) or not data:
        raise ValueError(f"{exam_path}: expected a non-empty list of questions")
    return data


def static_blocks(prompt: str, figures: list[Path], cache: bool = True) -> list[dict]:
    """Prompt text + figure images. Identical across exams and replicas, so
    the cache breakpoint goes on the last figure (breakpoints cover everything
    before them). Missing figures are a hard error — a run that silently
    dropped the schematics would corrupt the E7/E9 numbers."""
    blocks: list[dict] = [{"type": "text", "text": prompt}]
    for fig in figures:
        if not fig.exists():
            print(f"error: figure not found: {fig}", file=sys.stderr)
            sys.exit(1)
        blocks.append({"type": "text", "text": f"===== FIGURE FILE: {fig.name} ====="})
        blocks.append(image_block(fig))
    if cache:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def exam_block(exam_path: Path) -> dict:
    questions = load_exam(exam_path)
    text = json.dumps(questions, indent=1, ensure_ascii=False)
    return {"type": "text",
            "text": (f"===== EXAM JSON: {exam_path.name} ({len(questions)} questions) "
                     f"=====\n{text}\n===== END EXAM JSON =====")}


def build_messages(exam_path: Path, figures: list[Path], prompt: str | None = None,
                   cache: bool = True) -> list[dict]:
    content = static_blocks(prompt or EXAM_PROMPT, figures, cache)
    content.append(exam_block(exam_path))
    return [{"role": "user", "content": content}]


# ---- Response parsing ------------------------------------------------------
_FENCE_RE = re.compile(r"```(?:json)?\s*(\[.*?\])\s*```", re.DOTALL)
_PAIR_RE = re.compile(
    r'"id"\s*:\s*"?(\d+)"?\s*,\s*"answer"\s*:\s*"?\s*([A-Da-d])\b')


def extract_answers(text: str) -> tuple[dict[str, str], str]:
    """Return ({id: LETTER}, how). Tries, in order: a fenced JSON array, the
    outermost [...] in the text, then a tolerant regex over id/answer pairs
    (survives trailing 'and so on' commentary, missing brackets, etc.)."""
    candidates = [m.group(1) for m in _FENCE_RE.finditer(text)]
    start, end = text.find("["), text.rfind("]")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for cand in candidates:
        try:
            arr = json.loads(cand)
        except json.JSONDecodeError:
            continue
        if isinstance(arr, list) and arr and all(isinstance(a, dict) for a in arr):
            out = {}
            for a in arr:
                if "id" in a and "answer" in a:
                    out[str(a["id"]).strip()] = str(a["answer"]).strip().upper()[:1]
            if out:
                return out, "json"
    pairs = {i: a.upper() for i, a in _PAIR_RE.findall(text)}
    if pairs:
        print("warning: answers recovered by regex, not clean JSON", file=sys.stderr)
        return pairs, "regex"
    print("warning: no answers found in response", file=sys.stderr)
    return {}, "none"


def message_text(message) -> str:
    return "".join(b.text for b in message.content
                   if getattr(b, "type", None) == "text")


def answers_record(answers: dict[str, str], how: str, exam: list[dict],
                   message=None) -> dict:
    """The on-disk answers file: keeps the exam's question order, notes any
    ids the model skipped or invented, and carries usage for the stats."""
    ids = [str(q["id"]) for q in exam]
    rec = {
        "answers": [{"id": i, "answer": answers.get(i)} for i in ids],
        "parse": how,
        "missing_ids": [i for i in ids if i not in answers],
        "extra_ids": sorted(set(answers) - set(ids)),
    }
    if message is not None:
        u = message.usage
        rec["stop_reason"] = message.stop_reason
        rec["usage"] = {
            "input": u.input_tokens, "output": u.output_tokens,
            "cache_read": getattr(u, "cache_read_input_tokens", 0) or 0,
            "cache_create": getattr(u, "cache_creation_input_tokens", 0) or 0,
        }
    return rec


# ---- Batch plumbing (borrowed shape from polecat_batch) --------------------
def wait_for_batch(client, batch_id, poll_seconds):
    import time
    failures = 0
    while True:
        try:
            batch = client.messages.batches.retrieve(batch_id)
            failures = 0
        except anthropic.APIStatusError:
            raise
        except Exception as e:
            failures += 1
            wait = min(300, poll_seconds * failures)
            print(f"poll failed ({type(e).__name__}: {e}) — retry #{failures} "
                  f"in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        c = batch.request_counts
        print(f"batch {batch_id}: {batch.processing_status} "
              f"(processing={c.processing} succeeded={c.succeeded} errored={c.errored})")
        if batch.processing_status == "ended":
            return batch
        time.sleep(poll_seconds)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exam", type=Path, required=True,
                    help="Exam json dumped by the console scraper")
    ap.add_argument("--figure", type=Path, action="append", default=None,
                    help="Figure image (repeatable). Default: the three pool "
                         "figure screenshots under figures/")
    ap.add_argument("--output", type=Path, default=None,
                    help="Answers path (default: answers/<exam stem>.answers.json)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=16000)
    ap.add_argument("--temperature", type=float, default=None,
                    help="Leave unset for the API default")
    ap.add_argument("--thinking", action="store_true", help="Adaptive thinking")
    ap.add_argument("--effort", default=None,
                    help="low|medium|high|xhigh|max (implies --thinking)")
    ap.add_argument("--prompt", type=Path, default=None,
                    help="File containing an exam prompt to use instead of EXAM_PROMPT")
    ap.add_argument("--key", type=Path, default=None,
                    help="Answer key json; if given, score immediately")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--poll-seconds", type=int, default=60)
    ap.add_argument("--batch-id", default=None,
                    help="Re-attach to an already-submitted batch instead of submitting")
    args = ap.parse_args()

    if not args.exam.exists():
        print(f"error: exam json not found: {args.exam}", file=sys.stderr)
        return 1
    figures = args.figure or DEFAULT_FIGURES
    prompt = load_prompt(args.prompt)
    exam = load_exam(args.exam)
    output = args.output or Path("answers") / f"{args.exam.stem}.answers.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    messages = build_messages(args.exam, figures, prompt, cache=not args.no_cache)
    params = {"model": args.model, "max_tokens": args.max_tokens, "messages": messages}
    thinking = args.thinking or bool(args.effort)
    if thinking:
        if args.temperature not in (None, 1.0):
            print("error: thinking requires temperature unset/1.0", file=sys.stderr)
            return 1
        params["thinking"] = {"type": "adaptive", "display": "summarized"}
        if args.effort:
            params["output_config"] = {"effort": args.effort}
    elif args.temperature is not None:
        params["temperature"] = args.temperature

    print(f"exam: {args.exam.name} ({len(exam)} q)  figures: {len(figures)}  "
          f"model: {args.model}  thinking: {thinking}  -> {output}")

    client = anthropic.Anthropic()
    custom_id = re.sub(r"[^a-zA-Z0-9_-]", "-", args.exam.stem)[:60] + "-t0"
    if args.batch_id:
        batch_id = args.batch_id
        print(f"re-attaching to batch {batch_id}")
    else:
        batch = client.messages.batches.create(
            requests=[{"custom_id": custom_id, "params": params}])
        batch_id = batch.id
        print(f"submitted batch {batch_id} (custom_id={custom_id}) — "
              f"re-attach with --batch-id {batch_id} if interrupted")
    wait_for_batch(client, batch_id, args.poll_seconds)

    message = None
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            print(f"result: {result.result.type} "
                  f"{getattr(result.result, 'error', '')}", file=sys.stderr)
            return 2
        message = result.result.message
        break
    if message is None:
        print("error: batch returned no results", file=sys.stderr)
        return 2

    text = message_text(message)
    answers, how = extract_answers(text)
    rec = answers_record(answers, how, exam, message)
    output.write_text(json.dumps(rec, indent=1))
    output.with_suffix(".raw.txt").write_text(text)
    u = rec["usage"]
    print(f"wrote: {output}  ({len(answers)}/{len(exam)} answered, parse={how}, "
          f"stop={message.stop_reason})")
    print(f"usage: input={u['input']} output={u['output']} "
          f"cache_read={u['cache_read']} cache_create={u['cache_create']}")

    if args.key:
        import hamexam_score as hs
        key = hs.load_key(args.key)
        s = hs.score(rec["answers"], key, exam)
        print(hs.format_score(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
