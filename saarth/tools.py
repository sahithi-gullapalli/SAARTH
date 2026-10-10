"""Tools the agents can call. Each tool has an OpenAI-style JSON schema."""

from __future__ import annotations

import ast
import operator
import re
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

_OPS: dict[type, Callable[..., Any]] = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow,
    ast.USub: operator.neg, ast.UAdd: operator.pos,
}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ValueError("exponent too large")
        return _OPS[type(node.op)](_eval(node.left), right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    """Safely evaluate an arithmetic expression (no names, calls or attributes allowed)."""
    expr = expression.replace("^", "**").replace("×", "*")
    try:
        result = _eval(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, SyntaxError, TypeError):
        return f"Error: could not evaluate '{expression}'"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return f"{expression} = {result}"


def current_datetime(timezone_name: str = "UTC") -> str:
    try:
        tz = ZoneInfo(timezone_name)
    except (ZoneInfoNotFoundError, ValueError):
        tz, timezone_name = ZoneInfo("UTC"), "UTC"
    return f"Current date/time ({timezone_name}): {datetime.now(timezone.utc).astimezone(tz):%Y-%m-%d %H:%M}"


def word_count(text: str) -> str:
    return f"Word count: {len(text.split())}"


TOOLS: dict[str, Callable[..., str]] = {
    "calculator": calculator, "current_datetime": current_datetime, "word_count": word_count,
}

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "calculator", "description": "Evaluate an arithmetic expression, e.g. '(12+8)*3'.",
        "parameters": {"type": "object", "properties": {"expression": {"type": "string"}},
                       "required": ["expression"]}}},
    {"type": "function", "function": {
        "name": "current_datetime", "description": "Get the current date and time in an IANA timezone.",
        "parameters": {"type": "object", "properties": {"timezone_name": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {
        "name": "word_count", "description": "Count the words in a piece of text.",
        "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}}},
]


def run_tool(name: str, arguments: dict[str, Any]) -> str:
    fn = TOOLS.get(name)
    if fn is None:
        return f"Error: unknown tool '{name}'"
    try:
        return fn(**arguments)
    except TypeError as exc:
        return f"Error: bad arguments for {name}: {exc}"


_MATH = re.compile(r"[-(]*\d[\d.]*(?:\s*[-+*/%^×]\s*[-(]*\d[\d.]*\)*)+")


def detect_tool_calls(question: str) -> list[tuple[str, dict[str, Any]]]:
    """Rule-based tool selection used when the LLM has no native tool calling."""
    calls: list[tuple[str, dict[str, Any]]] = []
    m = _MATH.search(question)
    if m:
        calls.append(("calculator", {"expression": m.group(0).strip()}))
    if re.search(r"\b(time|date)\b", question, re.I) and re.search(r"\b(what|current|today|now)\b", question, re.I):
        tz = re.search(r"\b([A-Z][a-z]+/[A-Z][A-Za-z_]+)\b", question)
        calls.append(("current_datetime", {"timezone_name": tz.group(1) if tz else "UTC"}))
    return calls
