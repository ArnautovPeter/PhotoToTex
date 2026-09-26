"""Optional future document detection interface.

The baseline assumes one cropped formula per image. A detector can implement
``detect_regions`` and pass each crop to the recognizer before combining text.
"""

from __future__ import annotations

from typing import Any, Callable


def detect_regions(image: Any) -> list[tuple[int, int, int, int]]:
    """Return text/formula boxes; detection is intentionally not implemented."""
    raise NotImplementedError("Document region detection is a future extension.")


def recognize_document(image: Any, recognizer: Callable[[Any], str]) -> str:
    """Recognize detected crops in reading order once a detector is supplied."""
    regions = detect_regions(image)
    return "\n".join(recognizer(region) for region in regions)
