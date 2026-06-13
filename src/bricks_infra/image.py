"""Image generation via fal.ai (Qwen-Image-2512 by default).

Qwen-Image is the open text-to-image model that renders legible embedded text and
accurate brand names/slogans — exactly what ad creative and mailer hero images
need. We call fal's synchronous REST endpoint directly (no SDK dep): POST a prompt,
get back image URLs.

Used by bricks to generate the sector hero images (ad_creative/.../ai_*.png) and
creative variants for A/B testing.
"""

from __future__ import annotations

import httpx

from bricks_infra.config import get_settings


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

    resp = httpx.post(
        f"{s.image_base_url}/{s.image_model}",
        headers={"Authorization": f"Key {s.fal_api_key}"},
        json=payload,
        timeout=timeout,
    )
    resp.raise_for_status()
    return [img["url"] for img in resp.json().get("images", [])]


def model_name() -> str:
    return get_settings().image_model
