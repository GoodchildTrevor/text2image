"""Dynamic capability discovery for OpenRouter image models.

At startup, ``refresh_caps()`` fetches ``GET /api/v1/images/models`` from
OpenRouter and builds a mapping of model slug → list of supported sizes
(largest first).  The result is cached in ``MODEL_SIZES`` and used by
``clamp_size()`` to pick the best allowed size for each model.

If the endpoint is unreachable or returns unexpected data, a built-in
fallback table is used so the service still starts cleanly.

Usage::

    # at startup (inside an async context)
    await openrouter_caps.refresh_caps()

    # at request time
    best = openrouter_caps.clamp_size("openai/gpt-5-image", "4096x4096")
    # → "1792x1024"  (landscape clamped to largest allowed landscape)
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fallback table — used when the live endpoint is unreachable.
# Sorted largest-first within each model.  Only models with known hard limits
# need entries; unconstrained models are simply absent.
# ---------------------------------------------------------------------------
_FALLBACK_SIZES: dict[str, list[str]] = {
    "openai/gpt-image-1": [
        "1536x1024", "1024x1536", "1024x1024",
    ],
    "openai/gpt-5-image": [
        "1792x1024", "1024x1792", "1024x1024",
    ],
    "openai/gpt-5-image-mini": [
        "1792x1024", "1024x1792", "1024x1024",
    ],
    "openai/gpt-5.4-image-2": [
        "1792x1024", "1024x1792", "1024x1024",
    ],
    "google/gemini-3.1-flash-image-preview": ["1024x1024"],
    "google/gemini-3-pro-image-preview":     ["1024x1024"],
    "google/gemini-2.5-flash-image":         ["1024x1024"],
}

# ---------------------------------------------------------------------------
# Live cache — populated by refresh_caps(), falls back to _FALLBACK_SIZES
# ---------------------------------------------------------------------------
MODEL_SIZES: dict[str, list[str]] = dict(_FALLBACK_SIZES)

# Will be set to True once a successful live fetch has been done
_live_loaded: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_models_response(data: list[dict]) -> dict[str, list[str]]:
    """Extract model→sizes mapping from the /api/v1/images/models response.

    Each entry in the list may look like::

        {
            "id": "openai/gpt-5-image",
            "capabilities": {
                "sizes": ["1024x1024", "1792x1024", "1024x1792"],
                ...
            }
        }

    We sort sizes largest-first (by max side) so ``clamp_size`` can iterate
    top-to-bottom to find the best fit.

    :param data: Parsed JSON list from the endpoint.
    :returns: Dict mapping model slug → sorted size list.
    """
    result: dict[str, list[str]] = {}
    for entry in data:
        model_id = entry.get("id") or entry.get("model")
        if not model_id:
            continue
        caps = entry.get("capabilities") or entry.get("parameters") or {}
        sizes_raw: list[str] = (
            caps.get("sizes")
            or caps.get("supported_sizes")
            or entry.get("sizes")
            or []
        )
        if not sizes_raw:
            continue
        # Sort by max(w, h) descending
        def _max_side(s: str) -> int:
            try:
                w, h = map(int, s.split("x"))
                return max(w, h)
            except ValueError:
                return 0
        result[model_id] = sorted(sizes_raw, key=_max_side, reverse=True)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def refresh_caps(
    api_key: str | None = None,
    base_url: str = "https://openrouter.ai",
    timeout: float = 15.0,
) -> None:
    """Fetch live model capabilities from OpenRouter and update ``MODEL_SIZES``.

    On success, ``MODEL_SIZES`` is updated in-place: live data takes precedence
    for known models, and new models are added.  Fallback entries for models
    absent in the live response are preserved.

    On any error (network, auth, unexpected schema), the fallback table remains
    active and a warning is logged — the service continues normally.

    :param api_key: OpenRouter API key.  Defaults to ``OPENROUTER_API_KEY`` env var.
    :param base_url: OpenRouter base URL.
    :param timeout: HTTP timeout in seconds.
    """
    global _live_loaded

    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        logger.warning("openrouter_caps: OPENROUTER_API_KEY not set — using fallback sizes")
        return

    url = f"{base_url.rstrip('/')}/api/v1/images/models"
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(
                url,
                headers={"Authorization": f"Bearer {key}"},
            )
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.warning(f"openrouter_caps: failed to fetch {url}: {exc} — using fallback sizes")
        return

    # The endpoint may return a list directly or wrap it in {"data": [...]}
    if isinstance(payload, dict):
        entries = payload.get("data") or payload.get("models") or []
    elif isinstance(payload, list):
        entries = payload
    else:
        logger.warning(f"openrouter_caps: unexpected response shape ({type(payload)}) — using fallback sizes")
        return

    if not entries:
        logger.warning("openrouter_caps: endpoint returned empty model list — using fallback sizes")
        return

    live = _parse_models_response(entries)
    if not live:
        logger.warning("openrouter_caps: could not parse any model sizes — using fallback sizes")
        return

    # Merge: live data wins, fallback entries for absent models are kept
    MODEL_SIZES.update(live)
    _live_loaded = True
    logger.info(
        f"openrouter_caps: loaded {len(live)} models from live API "
        f"({'live' if _live_loaded else 'fallback'} total={len(MODEL_SIZES)}). "
        f"Sample: { {k: v[:2] for k, v in list(live.items())[:5]} }"
    )


def clamp_size(model: str, requested: str | None) -> str | None:
    """Return the best allowed size for *model* given *requested* ``WxH``.

    Reads from ``MODEL_SIZES`` (live or fallback).  If the model is not in the
    table, the requested size is returned unchanged (no constraints known).

    Strategy:
    - Exact match → keep as-is.
    - Otherwise find the largest allowed size with the same orientation
      whose long side ≤ requested long side.
    - If nothing fits → return the square ``1024x1024`` if available, else
      the smallest entry in the allowed list.

    :param model: OpenRouter model slug.
    :param requested: Pixel string ``"WxH"`` or ``None``.
    :returns: Clamped or original size string, or ``None`` if input is ``None``.
    """
    if requested is None:
        return None

    allowed = MODEL_SIZES.get(model)
    if not allowed:
        return requested  # no constraints for this model

    if requested in allowed:
        return requested

    try:
        req_w, req_h = map(int, requested.split("x"))
    except ValueError:
        return allowed[0]  # malformed → use largest allowed

    req_long = max(req_w, req_h)
    req_landscape = req_w > req_h
    req_portrait = req_h > req_w
    req_square = req_w == req_h

    for candidate in allowed:  # sorted largest-first
        try:
            c_w, c_h = map(int, candidate.split("x"))
        except ValueError:
            continue
        c_landscape = c_w > c_h
        c_portrait = c_h > c_w
        c_square = c_w == c_h

        same_orientation = (
            (req_landscape and c_landscape)
            or (req_portrait and c_portrait)
            or (req_square and c_square)
        )
        if same_orientation and max(c_w, c_h) <= req_long:
            logger.info(f"size clamp: model={model!r} {requested!r} → {candidate!r}")
            return candidate

    # Fallback: square or last entry
    fallback = "1024x1024" if "1024x1024" in allowed else allowed[-1]
    logger.info(f"size clamp fallback: model={model!r} {requested!r} → {fallback!r}")
    return fallback
