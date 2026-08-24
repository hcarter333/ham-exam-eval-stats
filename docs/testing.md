# Testing

## Current test coverage

The current pytest suite contains one focused unit test:

- `tests/test_score.py::test_score_counts_correct_wrong_blank_and_subgroups`

That test exercises `ham_exam_eval_stats.hamexam_score.score()` with a small
in-memory exam, answer key, and answer list. It verifies that the scorer:

- counts correct answers
- counts wrong answers
- counts blank answers
- computes the total answered/keyed question count
- aggregates right/total counts by subelement prefix, such as `E1` and `E2`

This gives a quick smoke test for the central scoring behavior without needing
API access, network access, filesystem fixtures, or a SQLite database.

Run it with:

```bash
python -m pytest tests
```

## Additional tests to add

The next tests should cover the pure functions first, then the command-line and
database workflows. Suggested additions:

- `hamexam_score.load_key()` should accept supported key formats:
  dictionaries, lists with `answer`, and lists with `correct`.
- `hamexam_score.load_answers()` should accept both a raw answers list and a
  dictionary containing an `answers` list.
- `hamexam_score.score()` should cover unkeyed answers, unknown clusters,
  lowercase answer letters, and pass/fail threshold behavior.
- `hamexam_batch.extract_answers()` should cover fenced JSON, plain JSON arrays,
  regex recovery, malformed responses, missing IDs, and extra IDs.
- `hamexam_batch.load_exam()` should cover both list-shaped exam files and
  `{"questions": [...]}` files.
- `hamexam_make.to_scraper_shape()` should verify stable question ordering,
  answer-key output, cluster formatting, and whitespace cleanup.
- `hamexam_fleet` database tests should create a temporary SQLite database and
  verify schema creation, replica numbering, and idempotent answer insertion.
- `hamexam_backfill` tests should use temporary CSV and JSON files to verify
  backfilled `runs` and `answers` rows, including rows with missing answer
  files.
- `hamexam_stats` tests should exercise the statistics helpers and report
  generation with a small temporary database.
- CLI tests should call each `main()` through subprocesses or monkeypatched
  arguments, using temporary directories for generated files.

## Tests that should not hit external services

Default tests should avoid Anthropic API calls, GitHub calls, and live downloads.
For code paths that normally require those services, prefer mocked clients,
temporary local files, or cached fixtures.

Networked integration tests can be added later, but they should be clearly
marked so the normal `python -m pytest tests` command remains fast and
repeatable.

## Useful validation before releases

Before tagging or publishing, run:

```bash
python -m pytest tests
python -m compileall -q ham_exam_eval_stats tests docs
```

When report output changes intentionally, regenerate the HTML report and compare
the new file in `reports/` before committing it.
