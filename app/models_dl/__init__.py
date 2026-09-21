"""Deep Learning models for invoice and document information extraction.

Contains:
  - InvoiceGNN: Spatial Relational Graph Neural Network with edge relations & text semantics.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models_dl.gnn import InvoiceGNN, InvoiceGNNConfig
    from app.models_dl.roberta import RoBERTaWithLayoutConfig, RoBERTaWithLayoutForTokenClassification

__all__ = [
    "InvoiceGNN",
    "InvoiceGNNConfig",
    "RoBERTaWithLayoutConfig",
    "RoBERTaWithLayoutForTokenClassification",
]


def __getattr__(name: str):
    if name in ("InvoiceGNN", "InvoiceGNNConfig"):
        from app.models_dl.gnn import InvoiceGNN, InvoiceGNNConfig
        return {"InvoiceGNN": InvoiceGNN, "InvoiceGNNConfig": InvoiceGNNConfig}[name]
    if name in ("RoBERTaWithLayoutConfig", "RoBERTaWithLayoutForTokenClassification"):
        from app.models_dl.roberta import RoBERTaWithLayoutConfig, RoBERTaWithLayoutForTokenClassification
        return {
            "RoBERTaWithLayoutConfig": RoBERTaWithLayoutConfig,
            "RoBERTaWithLayoutForTokenClassification": RoBERTaWithLayoutForTokenClassification,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

