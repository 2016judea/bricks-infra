"""Image generation via fal.ai (Qwen-Image-2512 by default).

Qwen-Image is the open text-to-image model that renders legible embedded text and
accurate brand names/slogans — exactly what ad creative and mailer hero images
need. We call fal's synchronous REST endpoint directly (no SDK dep): POST a prompt,
get back image URLs.

Used by bricks to generate the sector hero images (ad_creative/.../ai_*.png) and
creative variants for A/B testing.
"""

from __future__ import annotations

import time

import httpx

from bricks_infra.config import get_settings

_MAX_ATTEMPTS = 3


def generate_image(
    prompt: str,
    *,
    image_size: str = "square_hd",
    num_images: int = 1,
    seed: int | None = None,
    timeout: float = 180.0,
) -> list[str]:
    """Generate image(s) and return their URLs.

    image_size is a fal preset: "square_hd" (1024x1024), "landscape_16_9",
    "portrait_4_3", etc. Callers download the bytes from the returned URLs.
    """
    s = get_settings()
    if not s.fal_api_key:
        raise RuntimeError("image generation requested but BRICKS_INFRA_FAL_API_KEY is unset")

    payload: dict = {"prompt": prompt, "image_size": image_size, "num_images": num_images}
    if seed is not None:
        payload["seed"] = seed

    url = f"{s.image_base_url}/{s.image_model}"
    headers = {"Authorization": f"Key {s.fal_api_key}"}

    last_exc: Exception | None = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=timeout)
            # Retry server-side hiccups (5xx) but fail fast on 4xx (auth, balance, bad input).
            if resp.status_code >= 500:
                resp.raise_for_status()
            resp.raise_for_status()
            return [img["url"] for img in resp.json().get("images", [])]
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            # Don't retry deterministic client errors (4xx).
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                raise
            last_exc = exc
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"fal image request failed after {_MAX_ATTEMPTS} attempts: {last_exc}")


def model_name() -> str:
    return get_settings().image_model
