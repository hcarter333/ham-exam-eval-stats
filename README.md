# ham-exam-eval-stats

Tools for generating, running, scoring, and reporting ham radio Extra-class
exam evals.

This repository packages the existing `hamexam_*` scripts as
`ham_exam_eval_stats` while keeping the original top-level scripts in place.
The project layout follows the same broad Python packaging style used by
Datasette: project metadata in `pyproject.toml`, package code in a Python
package directory, tests in `tests/`, and documentation in `docs/`.

## Repository layout

- `ham_exam_eval_stats/` - importable package copies of the ham exam scripts.
- `tests/` - pytest tests.
- `docs/` - project documentation.
- `reports/` - generated HTML reports checked into the repository.
- `exams/`, `keys/`, `answers/` - JSON fixtures and eval outputs currently
  committed to the repository.

## Development

Install the package and development dependencies with a Python environment that
supports `pyproject.toml` dependency groups, or install the small current test
dependency directly:

```bash
python -m pip install pytest
```

Run the test suite:

```bash
python -m pytest tests
```

## Console scripts

The package exposes these command names when installed:

- `hamexam-backfill`
- `hamexam-batch`
- `hamexam-fleet`
- `hamexam-make`
- `hamexam-score`
- `hamexam-stats`

The original top-level scripts remain available for direct execution.
