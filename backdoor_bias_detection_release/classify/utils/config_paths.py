"""Expand environment variables embedded in public model-list JSON files."""

from __future__ import annotations

import os
import re


VARIABLE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


def expand_runtime_variables(value):
    """Recursively expand `${NAME}` values and fail clearly when unset."""
    if isinstance(value, dict):
        return {key: expand_runtime_variables(item) for key, item in value.items()}
    if isinstance(value, list):
        return [expand_runtime_variables(item) for item in value]
    if not isinstance(value, str) or not value:
        return value
    missing = sorted({name for name in VARIABLE.findall(value) if name not in os.environ})
    if missing:
        raise ValueError("Missing model-path environment variables: " + ", ".join(missing))
    return os.path.expandvars(os.path.expanduser(value))
