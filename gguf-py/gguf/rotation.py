"""Canonical input-rotation metadata and legacy GGUF compatibility."""
from collections.abc import Mapping
from typing import Any

ROTATION_PREFIX = "quantization.rotation."
LEGACY_ROTATION_PREFIX = "prism.hadamard."
ROTATION_PREFIXES = (ROTATION_PREFIX, LEGACY_ROTATION_PREFIX)


def rotation_key(key: str) -> str:
    """Translate an old metadata key without changing its value or type."""
    if key.startswith(LEGACY_ROTATION_PREFIX):
        return ROTATION_PREFIX + key[len(LEGACY_ROTATION_PREFIX):]
    return key


def rotation_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Read one complete namespace; mixed contracts are ambiguous."""
    present = [prefix for prefix in ROTATION_PREFIXES
               if any(key.startswith(prefix) for key in metadata)]
    if len(present) > 1:
        raise ValueError("mixed canonical and legacy rotation metadata")
    if not present:
        return {}
    prefix = present[0]
    result = {rotation_key(key): value for key, value in metadata.items()
              if key.startswith(prefix)}
    if ROTATION_PREFIX + "version" not in result:
        raise ValueError("rotation metadata requires a version")
    return result
