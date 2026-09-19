"""Deep Learning models for invoice and document information extraction.

Contains:
  - InvoiceGNN: Spatial Relational Graph Neural Network with edge relations & text semantics.
"""

from app.models_dl.gnn import InvoiceGNN, InvoiceGNNConfig

__all__ = ["InvoiceGNN", "InvoiceGNNConfig"]
