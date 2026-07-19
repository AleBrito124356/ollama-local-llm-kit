"""Structured output: free text into validated JSON.

Asks a local model to extract fields from an unstructured sentence, requests JSON
mode via ``response_format``, then validates the result against a Pydantic model
so downstream code gets typed, guaranteed-shaped data or a clear error.

Run:
    python examples/structured_output.py
    python examples/structured_output.py "Ada Lovelace, 36, London, wrote the first algorithm."
"""

from __future__ import annotations

import json
import os
import sys

from pydantic import BaseModel, Field, ValidationError

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.client import LLMClient  # noqa: E402


class Person(BaseModel):
    name: str = Field(description="Full name")
    age: int | None = Field(default=None, description="Age in years if stated")
    city: str | None = Field(default=None, description="City if stated")
    known_for: str | None = Field(default=None, description="What the person is known for")


SCHEMA_HINT = json.dumps(Person.model_json_schema(), indent=2)

SYSTEM = (
    "You extract structured data. Reply with a single JSON object only, no prose, "
    "matching this JSON schema:\n" + SCHEMA_HINT
)


def extract(text: str) -> Person:
    client = LLMClient.create()
    print(f"[backend={client.backend}  model={client.model_for('chat')}]\n")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": f"Extract a person from: {text}"},
    ]

    # JSON mode is honored by Ollama and NIM; the schema in the system prompt guides fields.
    raw = client.chat(messages, temperature=0, response_format={"type": "json_object"})

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model did not return valid JSON: {raw!r}") from exc

    return Person.model_validate(data)


if __name__ == "__main__":
    sentence = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "Grace Hopper, 79, from New York, popularized the term 'debugging'."
    )
    print(f"Input: {sentence}\n")
    try:
        person = extract(sentence)
    except (ValueError, ValidationError) as exc:
        print(f"Extraction failed: {exc}")
        sys.exit(1)

    print("Validated Person object:")
    print(person.model_dump_json(indent=2))
