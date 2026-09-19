"""Unified evaluation framework for invoice information extraction models.

Provides standardized token-level, entity-level, and resource profiling metrics
shared across all four deep learning models (GNN, RoBERTa, LayoutLMv3, LiLT).
"""

from app.evaluation.metrics import (
    EntitySpan,
    InferenceProfiler,
    compute_entity_metrics,
    compute_token_metrics,
    extract_entities_from_bio,
)
from app.evaluation.report import (
    export_full_benchmark_report,
    generate_benchmark_summary,
    plot_training_dashboard,
    save_markdown_report,
    save_training_history_csv,
)

__all__ = [
    "EntitySpan",
    "InferenceProfiler",
    "compute_entity_metrics",
    "compute_token_metrics",
    "extract_entities_from_bio",
    "export_full_benchmark_report",
    "generate_benchmark_summary",
    "plot_training_dashboard",
    "save_markdown_report",
    "save_training_history_csv",
]

