"""Caption extracted diagrams/images with a vision-capable model on Groq.

This is what makes retrieval "multimodal": a diagram's visual content
becomes a plain-text description that gets embedded and searched
alongside ordinary text chunks. No vision model is loaded locally --
the image is sent to Groq's hosted API -- so this step barely touches
local RAM, which matters on a 4GB machine.
"""
import base64
from pathlib import Path

from groq import Groq

from src.config import GROQ_API_KEY, VISION_MODEL

_client = None

CAPTION_PROMPT = (
    "Describe this technical diagram in detail for someone who cannot "
    "see it. Mention every labeled component, how they connect, and any "
    "numbers or text visible in the image."
)


def get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _guess_mime(image_path: str) -> str:
    ext = Path(image_path).suffix.lower().lstrip(".")
    return "jpeg" if ext in ("jpg", "jpeg") else (ext or "png")


def caption_image(image_path: str) -> str:
    """Send one image to the vision model and return a text description."""
    client = get_client()
    image_bytes = Path(image_path).read_bytes()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    mime = _guess_mime(image_path)

    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/{mime};base64,{b64_image}"},
                    },
                ],
            }
        ],
        temperature=0.2,
        max_tokens=400,
    )
    return response.choices[0].message.content
