"""Evaluation sub-package: harness, tracking, and benchmark adapters.

Public API
----------
.. autosummary::

    RunResult
    EvalConfig
    run_evaluation
    load_results_jsonl
    ExperimentTracker
    GaiaTask
    GaiaEvaluator
    load_gaia_tasks
"""

from ..eval_harness import RunResult, bootstrap_ci, stratified_sample, summarize
from .gaia import GaiaEvaluator, GaiaTask, load_gaia_tasks
from .harness import EvalConfig, load_results_jsonl, run_evaluation
from .tracking import ExperimentTracker, wandb_available

__all__ = [
    # Core types
    "RunResult",
    "EvalConfig",
    # Harness
    "run_evaluation",
    "load_results_jsonl",
    # Statistics (re-exported from eval_harness)
    "bootstrap_ci",
    "stratified_sample",
    "summarize",
    # Tracking
    "ExperimentTracker",
    "wandb_available",
    # GAIA
    "GaiaTask",
    "GaiaEvaluator",
    "load_gaia_tasks",
]
