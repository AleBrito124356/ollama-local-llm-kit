"""Tool / function calling on a local model.

Models like ``llama3.1:8b`` and ``qwen2.5:7b`` support the OpenAI tool-calling
protocol through Ollama. This example defines two small tools, lets the model
decide which to call, executes them locally, feeds the results back, and prints
the final natural-language answer.

Because it goes through the unified client, the exact same script runs against
cloud NVIDIA NIM with:  LLM_BACKEND=nim python examples/function_calling.py

Run:
    python examples/function_calling.py
    python examples/function_calling.py "What is 128 * 47, and is it raining in Panama City?"
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.client import LLMClient  # noqa: E402


# -- the actual tool implementations --------------------------------------
def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely (no names, no calls)."""
    allowed = set("0123456789+-*/(). %")
    if not expression or set(expression) - allowed:
        return json.dumps({"error": "unsupported characters in expression"})
    try:
        # eval is constrained to arithmetic only: empty builtins and a char allowlist.
        value = eval(expression, {"__builtins__": {}}, {})  # noqa: S307
    except Exception as exc:  # noqa: BLE001
        return json.dumps({"error": str(exc)})
    return json.dumps({"expression": expression, "result": value})


def get_weather(city: str) -> str:
    """Return a canned forecast. A real tool would call a weather API here."""
    fake = {
        "panama city": {"temp_c": 30, "condition": "thunderstorms", "humidity": 0.84},
        "reykjavik": {"temp_c": 6, "condition": "overcast", "humidity": 0.71},
    }
    data = fake.get(city.strip().lower(), {"temp_c": 22, "condition": "clear", "humidity": 0.5})
    return json.dumps({"city": city, **data})


TOOL_IMPLS = {"calculate": calculate, "get_weather": get_weather}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a basic arithmetic expression and return the numeric result.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "e.g. '128 * 47'"},
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name, e.g. 'Panama City'"},
                },
                "required": ["city"],
            },
        },
    },
]


def run(question: str) -> None:
    client = LLMClient.create()
    print(f"[backend={client.backend}  model={client.model_for('chat')}]\n")

    messages = [{"role": "user", "content": question}]

    # First round: let the model request tool calls.
    resp = client.raw_chat(messages, tools=TOOLS, tool_choice="auto")
    msg = resp.choices[0].message
    tool_calls = msg.tool_calls or []

    if not tool_calls:
        print(msg.content or "(no answer)")
        return

    # Record the assistant's tool-call message, then run each tool.
    messages.append(
        {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ],
        }
    )

    for tc in tool_calls:
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        impl = TOOL_IMPLS.get(name)
        result = impl(**args) if impl else json.dumps({"error": f"unknown tool {name}"})
        print(f"-> {name}({args}) = {result}")
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": result})

    # Second round: the model answers using the tool results.
    final = client.chat(messages)
    print("\n" + final)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is 128 * 47, and what's the weather in Panama City?"
    run(q)
