"""Offline evaluation contracts and quality metrics."""

from .metrics import (
    MAP,
    MRR,
    NDCG,
    CallableEvaluationMetric,
    EvaluationMetric,
    HitRate,
    Precision,
    Recall,
    Success,
)
from .runner import (
    CaseEvaluation,
    EvaluationCase,
    EvaluationDataset,
    EvaluationPerformance,
    EvaluationReport,
    evaluate,
)

__all__ = [
    "MAP",
    "MRR",
    "NDCG",
    "CallableEvaluationMetric",
    "CaseEvaluation",
    "EvaluationCase",
    "EvaluationDataset",
    "EvaluationMetric",
    "EvaluationPerformance",
    "EvaluationReport",
    "HitRate",
    "Precision",
    "Recall",
    "Success",
    "evaluate",
]
