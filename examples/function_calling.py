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

import ast
import json
import math
import operator
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.client import LLMClient  # noqa: E402


# -- the actual tool implementations --------------------------------------
# The model writes the expression, so it is untrusted input. It is parsed into a
# Python AST and walked by hand: only numbers, + - * / // % ** and unary +/- are
# allowed. No eval, no names, no calls, no attribute access. Size limits stop
# pathological inputs such as 9**9**9**9 (which never finishes under eval) and
# results too large to print.
MAX_EXPRESSION_CHARS = 200
MAX_NODES = 100
MAX_RESULT_BITS = 1024          # ints up to ~308 decimal digits
MAX_FLOAT = 1e300

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


class CalcError(ValueError):
    """The expression is not plain arithmetic or its result is out of bounds."""


def _check_size(value):
    if isinstance(value, int) and value.bit_length() > MAX_RESULT_BITS:
        raise CalcError("result is too large")
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value) or abs(value) > MAX_FLOAT):
        raise CalcError("result is too large")
    return value


def _pow(base, exponent):
    """``base ** exponent`` without ever computing a result that would be rejected anyway."""
    magnitude = abs(base)
    # log2(|result|) = exponent * log2(|base|): reject before doing any work.
    if magnitude not in (0, 1) and exponent != 0 and math.log2(magnitude) * exponent > MAX_RESULT_BITS:
        raise CalcError("result is too large")
    try:
        result = operator.pow(base, exponent)
    except OverflowError as exc:
        raise CalcError("result is too large") from exc
    if isinstance(result, complex):
        raise CalcError("complex numbers are not supported")
    return result


def _eval_node(node):
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return _check_size(node.value)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow):
            return _check_size(_pow(left, right))
        return _check_size(_BIN_OPS[type(node.op)](left, right))
    raise CalcError(f"unsupported syntax: {type(node).__name__}")


def safe_eval(expression: str) -> int | float:
    """Evaluate plain arithmetic, or raise CalcError / ZeroDivisionError."""
    if not expression or not expression.strip():
        raise CalcError("empty expression")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalcError(f"expression is longer than {MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalcError("not a valid arithmetic expression") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_NODES:
        raise CalcError("expression is too complex")
    return _eval_node(tree)


def calculate(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely; always returns a JSON string."""
    try:
        value = safe_eval(expression)
    except ZeroDivisionError:
        return json.dumps({"expression": expression, "error": "division by zero"})
    except (CalcError, RecursionError, MemoryError) as exc:
        return json.dumps({"expression": expression, "error": str(exc) or "invalid expression"})
    if isinstance(value, float) and value.is_integer() and abs(value) < 2**53:
        value = int(value)
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


def run_tool(name: str, raw_arguments: str | None) -> str:
    """Run one tool call from the model; any mistake becomes a JSON error the model can read."""
    impl = TOOL_IMPLS.get(name)
    if impl is None:
        return json.dumps({"error": f"unknown tool {name!r}"})
    try:
        args = json.loads(raw_arguments or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "tool arguments are not valid JSON"})
    if not isinstance(args, dict):
        return json.dumps({"error": "tool arguments must be a JSON object"})
    try:
        return impl(**args)
    except TypeError as exc:  # missing or unexpected argument names
        return json.dumps({"error": f"bad arguments for {name}: {exc}"})

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
        result = run_tool(name, tc.function.arguments)
        print(f"-> {name}({tc.function.arguments}) = {result}")
        messages.append({"role": "tool", "tool_call_id": tc.id, "name": name, "content": result})

    # Second round: the model answers using the tool results.
    final = client.chat(messages)
    print("\n" + final)


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "What is 128 * 47, and what's the weather in Panama City?"
    run(q)
