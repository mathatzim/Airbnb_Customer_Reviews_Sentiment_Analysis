from __future__ import annotations

from pathlib import Path

def project_root() -> Path:
    # src/<module>.py -> repo root is two levels up
    return Path(__file__).resolve().parents[2]
