"""Dynamic capability discovery for OpenRouter-compatible image models.

At startup, ``refresh_caps()`` fetches ``GET /api/v1/images/models`` from
the URL set in ``OPENROUTER_BASE_URL``.  The result is cached in
``MODEL_SIZES`` and used by ``clamp_size()`` to pick the best allowed size.

If the endpoint is unreachable or returns unexpected data, a built-in
fallback table is used so the service still starts cleanly.

``OPENROUTER_BASE_URL`` is **required** when OpenRouter models are used.
No URL is hardcoded in this module.

Usage::

    # at startup (inside an async context)
    await openrouter_caps.refresh_caps()

    # at request time
    best = openrouter_caps.clamp_size("openai/gpt-5-image", "4096x4096")
    # → "1792x1024"
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)


def _require_base_url() -> str | None:
    """Return OPENROUTER_BASE_URL from env, or None if not set."""
    return os.getenv("OPENROUTER_BASE_URL", "").rstrip("/") or None


# ---------------------------------------------------------------------------
# Fallback table
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

MODEL_SIZES: dict[str, list[str]] = dict(_FALLBACK_SIZES)
_live_loaded: bool = False


def _parse_models_response(data: list[dict]) -> dict[str, list[str]]:
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

        def _max_side(s: str) -> int:
            try:
                w, h = map(int, s.split("x"))
                return max(w, h)
            except ValueError:
                return 0

        result[model_id] = sorted(sizes_raw, key=_max_side, reverse=True)
    return result


async def refresh_caps(
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 15.0,
) -> None:
    """Fetch live model capabilities and update ``MODEL_SIZES``.

    URL resolution order:
    1. ``base_url`` argument
    2. ``OPENROUTER_BASE_URL`` env var
    3. Skip fetch entirely (log info, use fallback) — no hardcoded URL.

    :param api_key: Defaults to ``OPENROUTER_API_KEY`` env var.
    :param base_url: Explicit override; otherwise read from env.
    :param timeout: HTTP timeout in seconds.
    """
    global _live_loaded

    key = api_key or os.getenv("OPENROUTER_API_KEY", "")
    if not key:
        logger.warning("openrouter_caps: OPENROUTER_API_KEY not set — using fallback sizes")
        return

    resolved_base = (base_url or _require_base_url())
    if not resolved_base:
        logger.info(
            "openrouter_caps: OPENROUTER_BASE_URL not set — skipping caps fetch, using fallback sizes"
        )
        return

    url = f"{resolved_base}/api/v1/images/models"
    logger.info(f"openrouter_caps: fetching caps from {url}")

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {key}"})
            resp.raise_for_status()
            payload = resp.json()
    except Exception as exc:
        logger.warning(f"openrouter_caps: failed to fetch {url}: {exc} — using fallback sizes")
        return

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

    MODEL_SIZES.update(live)
    _live_loaded = True
    logger.info(
        f"openrouter_caps: loaded {len(live)} models from {resolved_base} "
        f"(total={len(MODEL_SIZES)}). "
        f"Sample: { {k: v[:2] for k, v in list(live.items())[:5]} }"
    )


def clamp_size(model: str, requested: str | None) -> str | None:
    """Return the best allowed size for *model* given *requested* ``WxH``."""
    if requested is None:
        return None

    allowed = MODEL_SIZES.get(model)
    if not allowed:
        return requested

    if requested in allowed:
        return requested

    try:
        req_w, req_h = map(int, requested.split("x"))
    except ValueError:
        return allowed[0]

    req_long = max(req_w, req_h)
    req_landscape = req_w > req_h
    req_portrait = req_h > req_w
    req_square = req_w == req_h

    for candidate in allowed:
        try:
            c_w, c_h = map(int, candidate.split("x"))
        except ValueError:
            continue
        same_orientation = (
            (req_landscape and c_w > c_h)
            or (req_portrait and c_h > c_w)
            or (req_square and c_w == c_h)
        )
        if same_orientation and max(c_w, c_h) <= req_long:
            logger.info(f"size clamp: model={model!r} {requested!r} → {candidate!r}")
            return candidate

    fallback = "1024x1024" if "1024x1024" in allowed else allowed[-1]
    logger.info(f"size clamp fallback: model={model!r} {requested!r} → {fallback!r}")
    return fallback
