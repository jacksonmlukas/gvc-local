"""Aggregate solver result JSONL files into a stratified, CI-aware results table.

Given a results JSONL file (produced by the solver runs) and a tagged puzzles
file (produced by :mod:`gvc_local.eval.tagger`), this module joins them on
``puzzle_id`` and computes:

* Overall solve rate with bootstrapped 95% CI.
* Strikes (failed guesses) per puzzle with bootstrapped 95% CI.
* Per-stratum solve rate with bootstrapped 95% CIs.

The output is a markdown table suitable for pasting into the README.

The point of this script is not to invent new metrics — it's to take the
results we already have (single-point estimates per puzzle) and report
them with the statistical rigor that an eval framework should.  The
stratified breakdown is the operational signal the framework is built for:
when an overall release looks fine but a single stratum has regressed,
the table here is what catches it.

Usage::

    # Aggregate one solver's results into a markdown table.
    python -m gvc_local.eval.aggregate \\
        --results results/snap_gvc_v3_llama8b_groq.jsonl \\
        --puzzles data/puzzles/tagged_connections.json \\
        --solver-name snap_gvc_v3 \\
        --out results/snap_gvc_v3_stratified.md

    # Aggregate every solver in results/ into a combined comparison table.
    python -m gvc_local.eval.aggregate \\
        --results-dir results/ \\
        --puzzles data/puzzles/tagged_connections.json \\
        --combined --out results/combined_stratified.md
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ..eval_harness import RunResult, bootstrap_ci
from .tagger import DEFAULT_STRATUM, STRATA

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Loading + joining
# ---------------------------------------------------------------------------


@dataclass
class AggregatedResult:
    """One puzzle's outcome enriched with its strata label."""

    puzzle_id: int
    solved: bool
    strikes: int
    categories_solved: int
    strata: str


def _read_jsonl(path: Path) -> list[dict]:
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _read_puzzles(path: Path) -> list[dict]:
    text = path.read_text().lstrip()
    if text.startswith("["):
        return json.loads(text)
    return _read_jsonl(path)


def _strata_index(puzzles: Iterable[dict]) -> dict[int, str]:
    """Build a {puzzle_id: strata} lookup from the tagged-puzzles file."""
    return {
        int(p["puzzle_id"]): str(p.get("category", DEFAULT_STRATUM))
        for p in puzzles
        if "puzzle_id" in p
    }


def join_results(
    results: list[dict],
    strata_lookup: dict[int, str],
) -> list[AggregatedResult]:
    """Join solver results with strata labels.

    Results rows without a known puzzle_id fall back to the default stratum
    so they still appear in the overall summary (just not in stratified rows).
    """
    aggregated: list[AggregatedResult] = []
    for row in results:
        pid = int(row["puzzle_id"])
        aggregated.append(
            AggregatedResult(
                puzzle_id=pid,
                solved=bool(row.get("solved", False)),
                strikes=int(row.get("strikes", row.get("guesses", 0))),
                categories_solved=int(row.get("categories_solved", 0)),
                strata=strata_lookup.get(pid, DEFAULT_STRATUM),
            )
        )
    return aggregated


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------


def _to_run_results(rows: Iterable[AggregatedResult]) -> list[RunResult]:
    """Adapt AggregatedResult into the RunResult shape the eval_harness expects."""
    return [
        RunResult(
            solved=r.solved,
            guesses=r.strikes,
            out_of_board_guesses=0,  # not recorded in legacy result schema
            puzzle_id=r.puzzle_id,
            strata=r.strata,
        )
        for r in rows
    ]


@dataclass
class MetricRow:
    """One row of a results table — a stratum, its sample size, and CI-aware metrics."""

    stratum: str
    n: int
    solve_rate: tuple[float, float, float]  # (mean, lo, hi)
    strikes_per_puzzle: tuple[float, float, float]


def _row_for(rows: list[AggregatedResult], stratum: str, n_resamples: int) -> MetricRow:
    solved = [1.0 if r.solved else 0.0 for r in rows]
    strikes = [float(r.strikes) for r in rows]
    return MetricRow(
        stratum=stratum,
        n=len(rows),
        solve_rate=bootstrap_ci(solved, n_resamples=n_resamples),
        strikes_per_puzzle=bootstrap_ci(strikes, n_resamples=n_resamples),
    )


def summarise_stratified(
    aggregated: list[AggregatedResult],
    n_resamples: int = 1000,
) -> list[MetricRow]:
    """Compute one MetricRow per stratum plus an overall row at index 0."""
    rows: list[MetricRow] = []
    rows.append(_row_for(aggregated, stratum="OVERALL", n_resamples=n_resamples))

    by_stratum: dict[str, list[AggregatedResult]] = {}
    for r in aggregated:
        by_stratum.setdefault(r.strata, []).append(r)

    # Emit strata in canonical order, then any unexpected ones alphabetically.
    canonical = [s for s in STRATA if s in by_stratum]
    extra = sorted(set(by_stratum) - set(canonical))
    for stratum in canonical + extra:
        rows.append(_row_for(by_stratum[stratum], stratum=stratum, n_resamples=n_resamples))

    return rows


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_ci(triple: tuple[float, float, float], pct: bool = False) -> str:
    """Render (mean, lo, hi) as 'mean [lo, hi]'. Percentages if pct=True."""
    mean, lo, hi = triple
    if any(v != v for v in (mean, lo, hi)):  # NaN check
        return "n/a"
    if pct:
        return f"{mean * 100:.1f}% [{lo * 100:.1f}, {hi * 100:.1f}]"
    return f"{mean:.2f} [{lo:.2f}, {hi:.2f}]"


def render_markdown(
    rows: list[MetricRow],
    solver_name: str,
    n_bootstrap: int,
) -> str:
    """Render summary rows as a markdown table."""
    lines: list[str] = [
        f"### {solver_name} — stratified results",
        "",
        f"Bootstrapped 95% CIs from {n_bootstrap:,} resamples. Strata assigned by "
        "`gvc_local.eval.tagger` (heuristic, not hand-labelled).",
        "",
        "| Stratum | n | Solve rate | Strikes / puzzle |",
        "|---|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.stratum} | {row.n} | "
            f"{_fmt_ci(row.solve_rate, pct=True)} | "
            f"{_fmt_ci(row.strikes_per_puzzle)} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_combined_markdown(
    per_solver_rows: dict[str, list[MetricRow]],
    n_bootstrap: int,
) -> str:
    """Render multiple solvers' overall rows side-by-side for comparison."""
    lines: list[str] = [
        "### Solver comparison (overall, with bootstrap 95% CIs)",
        "",
        f"Bootstrapped 95% CIs from {n_bootstrap:,} resamples.",
        "",
        "| Solver | n | Solve rate | Strikes / puzzle |",
        "|---|---:|---|---|",
    ]
    for solver_name, rows in per_solver_rows.items():
        overall = rows[0]  # by convention, row 0 is OVERALL
        lines.append(
            f"| {solver_name} | {overall.n} | "
            f"{_fmt_ci(overall.solve_rate, pct=True)} | "
            f"{_fmt_ci(overall.strikes_per_puzzle)} |"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _process_one(
    results_path: Path,
    strata_lookup: dict[int, str],
    n_bootstrap: int,
) -> tuple[str, list[MetricRow]]:
    """Process a single results JSONL file. Returns (solver_name, rows)."""
    rows = _read_jsonl(results_path)
    aggregated = join_results(rows, strata_lookup)
    metric_rows = summarise_stratified(aggregated, n_resamples=n_bootstrap)
    solver_name = results_path.stem
    return solver_name, metric_rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--results", type=Path, help="Single results JSONL to aggregate.")
    group.add_argument(
        "--results-dir",
        type=Path,
        help="Directory of results JSONL files; one solver per file.",
    )
    parser.add_argument(
        "--puzzles",
        type=Path,
        required=True,
        help="Tagged puzzles file (output of gvc_local.eval.tagger).",
    )
    parser.add_argument(
        "--solver-name",
        type=str,
        default=None,
        help="Display name for single-file mode (defaults to filename stem).",
    )
    parser.add_argument(
        "--combined",
        action="store_true",
        help="In --results-dir mode, emit a combined comparison table.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap resamples (default 1000).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write markdown to this path (defaults to stdout).",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    puzzles = _read_puzzles(args.puzzles)
    strata_lookup = _strata_index(puzzles)
    logger.info("Loaded %d tagged puzzles from %s", len(strata_lookup), args.puzzles)

    md_chunks: list[str] = []
    per_solver: dict[str, list[MetricRow]] = {}

    if args.results:
        solver_name, rows = _process_one(args.results, strata_lookup, args.n_bootstrap)
        if args.solver_name:
            solver_name = args.solver_name
        per_solver[solver_name] = rows
        md_chunks.append(render_markdown(rows, solver_name, args.n_bootstrap))
    else:
        result_files = sorted(args.results_dir.glob("*.jsonl"))
        for path in result_files:
            try:
                solver_name, rows = _process_one(path, strata_lookup, args.n_bootstrap)
            except (KeyError, ValueError, json.JSONDecodeError) as exc:
                logger.warning("Skipping %s: %s", path, exc)
                continue
            per_solver[solver_name] = rows
            md_chunks.append(render_markdown(rows, solver_name, args.n_bootstrap))

        if args.combined:
            md_chunks.insert(0, render_combined_markdown(per_solver, args.n_bootstrap))

    output_md = "\n".join(md_chunks)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output_md)
        logger.info("Wrote markdown to %s", args.out)
    else:
        print(output_md)

    return 0


if __name__ == "__main__":
    sys.exit(main())
