"""Single point of validation and resolution for image-size parameters.

Previously this logic was scattered across:
  - app/routers/openai_compat.py  (VALID_RESOLUTIONS, VALID_SIZES, _resolve_size_params)
  - app/providers.py              (_build_size_payload — consumer of already-resolved values)
  - app/service.py                (signature/docstring, with no validation of its own)
  - app/pipe.py                   (its own aspect_ratio value dropdown, not wired to the server)

Now any calling code (routers, service.py, future endpoints) should use
resolve_and_validate_size(...) instead of duplicating the logic.

The module does not depend on FastAPI: on error it raises ValueError, and the
routers themselves convert that into HTTPException(400, ...) — as they already
do for the rest of the validation errors (see direct_image.py, openai_compat.py).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

VALID_RESOLUTIONS: set[str] = {"512", "1K", "2K", "4K"}

_DEFAULT_SIZES = "1024x1024,864x1184,1184x864,768x1344,1344x768"
VALID_SIZES: set[str] = {
    s.strip()
    for s in os.getenv("VALID_SIZES", _DEFAULT_SIZES).split(",")
    if s.strip()
}

VALID_ASPECT_RATIOS: set[str] = {
    "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9",
}


@dataclass(frozen=True, slots=True)
class SizeParams:
    resolution: str | None
    aspect_ratio: str | None
    width: int | None
    height: int | None

    def __iter__(self):
        return iter((self.resolution, self.aspect_ratio, self.width, self.height))


def resolve_and_validate_size(
    size: str | None = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
) -> SizeParams:
    """Validate and normalize size/resolution/aspect_ratio.

    :param size: An OpenRouter resolution tier (e.g. ``"1K"``) or an explicit
        ``WIDTHxHEIGHT`` from ``VALID_SIZES``.
    :param resolution: An OpenRouter resolution tier; must be in ``VALID_RESOLUTIONS``.
    :param aspect_ratio: Aspect ratio; must be in ``VALID_ASPECT_RATIOS``.
    :raises ValueError: if any value fails validation.
    :returns: SizeParams with the resolved (resolution, aspect_ratio, width, height).
    """
    if aspect_ratio is not None and aspect_ratio not in VALID_ASPECT_RATIOS:
        raise ValueError(
            f"Invalid aspect_ratio {aspect_ratio!r}. Must be one of: {sorted(VALID_ASPECT_RATIOS)}"
        )

    if resolution is not None:
        if resolution not in VALID_RESOLUTIONS:
            raise ValueError(
                f"Invalid resolution {resolution!r}. Must be one of: {sorted(VALID_RESOLUTIONS)}"
            )
        return SizeParams(resolution, aspect_ratio, None, None)

    if size is not None:
        if size in VALID_RESOLUTIONS:
            return SizeParams(size, aspect_ratio, None, None)
        if size not in VALID_SIZES:
            raise ValueError(
                f"Invalid size {size!r}. Must be one of: {sorted(VALID_SIZES)} "
                f"or resolution tier: {sorted(VALID_RESOLUTIONS)}"
            )
        try:
            w, h = map(int, size.split("x"))
        except ValueError:
            raise ValueError(f"Malformed size {size!r}, expected WxH format")
        return SizeParams(None, aspect_ratio, w, h)

    return SizeParams(None, aspect_ratio, 1024, 1024)
