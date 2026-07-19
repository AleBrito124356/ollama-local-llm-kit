"""Vision: ask a local multimodal model about an image.

Uses ``llava:7b`` through Ollama's OpenAI-compatible endpoint, sending the image
as a base64 ``data:`` URL in a content part. If you don't pass an image path, the
script draws a small test image with Pillow so it runs out of the box.

Pull the model first:
    python -m src.model_manager pull llava:7b

Run:
    python examples/vision.py
    python examples/vision.py path/to/photo.jpg "What objects are in this image?"
"""

from __future__ import annotations

import base64
import io
import mimetypes
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.client import LLMClient  # noqa: E402


def make_sample_image() -> bytes:
    """Draw a simple labeled shape image with Pillow and return PNG bytes."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (480, 320), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 200, 200], fill=(37, 99, 235))          # blue square
    draw.ellipse([260, 60, 420, 220], fill=(220, 38, 38))            # red circle
    draw.polygon([(150, 300), (250, 230), (350, 300)], fill=(22, 163, 74))  # green triangle
    draw.text((150, 15), "shapes", fill=(30, 30, 30))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_to_data_url(image_bytes: bytes, mime: str = "image/png") -> str:
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def load_image(path: str | None) -> str:
    """Return a data URL for the given image path, or a generated sample."""
    if path:
        mime = mimetypes.guess_type(path)[0] or "image/png"
        with open(path, "rb") as fh:
            return image_to_data_url(fh.read(), mime)
    print("No image path given; generating a sample image with Pillow.\n")
    return image_to_data_url(make_sample_image(), "image/png")


def describe(data_url: str, question: str) -> None:
    # Vision runs locally on llava; the "vision" role maps to the vision model per backend.
    client = LLMClient.create()
    model = client.model_for("vision")
    print(f"[backend={client.backend}  model={model}]\n")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    print("assistant ", end="")
    for token in client.stream(messages, role="vision"):
        print(token, end="", flush=True)
    print()


if __name__ == "__main__":
    args = sys.argv[1:]
    path = args[0] if args and os.path.exists(args[0]) else None
    if path:
        question = args[1] if len(args) > 1 else "Describe this image in detail."
    else:
        question = args[0] if args else "What shapes and colors do you see in this image?"

    data_url = load_image(path)
    describe(data_url, question)
