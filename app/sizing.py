"""Единая точка валидации и разрешения параметров размера изображения.

Раньше логика была разбросана по:
  - app/routers/openai_compat.py  (VALID_RESOLUTIONS, VALID_SIZES, _resolve_size_params)
  - app/providers.py              (_build_size_payload — потребитель уже готовых значений)
  - app/service.py                (сигнатура/докстринг, без собственной валидации)
  - app/pipe.py                   (свой dropdown значений aspect_ratio, без связи с сервером)

Теперь любой вызывающий код (роутеры, service.py, будущие эндпоинты) должен
использовать resolve_and_validate_size(...) вместо копирования логики.

Модуль не зависит от FastAPI: при ошибке кидает ValueError, а роутеры сами
конвертируют это в HTTPException(400, ...) — как они уже делают для остальных
ошибок валидации (см. direct_image.py, openai_compat.py).
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

# Тот же набор, что в dropdown UserValves.ASPECT_RATIO в app/pipe.py.
# pipe.py — отдельный standalone-файл для OpenWebUI Function и не может
# импортировать app.sizing напрямую, поэтому список там дублируется намеренно.
# При изменении этого сета — обновите dropdown в app/pipe.py вручную.
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
        # Позволяет распаковку: resolution, aspect_ratio, width, height = resolve_and_validate_size(...)
        return iter((self.resolution, self.aspect_ratio, self.width, self.height))


def resolve_and_validate_size(
    size: str | None = None,
    resolution: str | None = None,
    aspect_ratio: str | None = None,
) -> SizeParams:
    """Провалидировать и нормализовать size/resolution/aspect_ratio.

    :param size: Тир резолюции OpenRouter (например ``"1K"``) либо явный
        ``WIDTHxHEIGHT`` из ``VALID_SIZES``.
    :param resolution: Тир резолюции OpenRouter, должен быть в ``VALID_RESOLUTIONS``.
    :param aspect_ratio: Соотношение сторон, должен быть в ``VALID_ASPECT_RATIOS``.
    :raises ValueError: если любое значение не проходит валидацию.
    :returns: SizeParams с разрешёнными (resolution, aspect_ratio, width, height).
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
