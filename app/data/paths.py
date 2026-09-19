"""Portable dataset path resolution.

Finds dataset directories across different machines and environments without relying on
hardcoded absolute paths. Works seamlessly on any developer machine, CI/CD, or Kaggle/Colab.
"""

from __future__ import annotations

import os
from pathlib import Path

# Root of the reconia repository (e.g. .../DACNTT/reconia)
REPO_ROOT = Path(__file__).resolve().parents[2]

# Workspace root containing reconia and its sibling directories (e.g. .../DACNTT)
WORKSPACE_ROOT = REPO_ROOT.parent


def resolve_dataset_path(
    env_var_name: str,
    relative_candidates: list[Path],
    dataset_name: str,
    explicit_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a dataset directory across explicit args, env vars, and auto-discovered paths.

    Priority:
      1. explicit_path passed directly by the caller.
      2. Environment variable (e.g. DOCILE_DIR) if set AND the directory actually exists.
      3. Candidate relative locations (workspace sibling, repo subfolder, cwd).
      4. If none exist, raise FileNotFoundError with clear instructions for teammates.
    """
    if explicit_path is not None:
        p = Path(explicit_path)
        if p.is_dir():
            return p
        raise FileNotFoundError(f"Explicit path for {dataset_name} not found: {p}")

    # Check environment variable if valid and existing
    if env_var_name in os.environ:
        env_p = Path(os.environ[env_var_name])
        if env_p.is_dir():
            return env_p

    # Auto-discover among candidate paths
    for candidate in relative_candidates:
        if candidate.is_dir():
            return candidate

    # Provide helpful error message if missing
    primary = relative_candidates[0] if relative_candidates else (WORKSPACE_ROOT / dataset_name)
    candidates_str = "\n".join(f"  - {c}" for c in relative_candidates)
    raise FileNotFoundError(
        f"Dataset '{dataset_name}' not found.\n"
        f"Searched candidate locations:\n{candidates_str}\n"
        f"To resolve: place the dataset at '{primary}' or set the environment variable {env_var_name}."
    )
