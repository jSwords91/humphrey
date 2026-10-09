"""Small example tools - a starting point for adding your own.

get_weather shows a tool that calls an external HTTP API (Open-Meteo, free,
no API key); calculate shows a pure local tool with careful input handling.
Both follow the same handler contract as app/tools/platform.py: take a dict
of arguments, return a JSON string, and report failures as {"error": ...}.
"""

from __future__ import annotations

import ast
import json
import math
import operator
from typing import Any

import httpx

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


def _ok(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _err(message: str) -> str:
    return _ok({"error": message})


# --- weather -----------------------------------------------------------------


def _get_weather(args: dict[str, Any]) -> str:
    location = str(args.get("location", "")).strip()
    if not location:
        return _err("location is required, e.g. 'London' or 'Paris, France'")

    try:
        geo = httpx.get(GEOCODE_URL, params={"name": location, "count": 1}, timeout=15)
        geo.raise_for_status()
        matches = geo.json().get("results") or []
        if not matches:
            return _err(f"no place found matching {location!r}")
        place = matches[0]

        forecast = httpx.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,precipitation,wind_speed_10m",
                "hourly": "temperature_2m,precipitation",
                "forecast_hours": 6,
                "timezone": "auto",
            },
            timeout=15,
        )
        forecast.raise_for_status()
        data = forecast.json()
    except httpx.HTTPError as exc:
        return _err(f"failed to fetch weather: {exc}")

    hourly = data.get("hourly") or {}
    next_hours = zip(
        hourly.get("time") or [],
        hourly.get("temperature_2m") or [],
        hourly.get("precipitation") or [],
        strict=False,
    )
    return _ok(
        {
            "place": ", ".join(
                part for part in (place.get("name"), place.get("admin1"), place.get("country")) if part
            ),
            "timezone": data.get("timezone"),
            "current": data.get("current"),
            "next_hours": [
                {"time": t, "temperature_c": temp, "precipitation_mm": p} for t, temp, p in next_hours
            ],
        }
    )


# --- calculator --------------------------------------------------------------

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    name: getattr(math, name)
    for name in ("sqrt", "log", "log10", "exp", "sin", "cos", "tan", "floor", "ceil")
} | {"abs": abs, "round": round}
_CONSTANTS = {"pi": math.pi, "e": math.e}
MAX_EXPRESSION_CHARS = 200
MAX_EXPONENT = 1_000


def _evaluate(node: ast.AST) -> float:
    """Walk an arithmetic AST. Anything else (names, attributes, calls to
    unknown functions) is rejected, so this never evaluates arbitrary code.
    """
    if isinstance(node, ast.Expression):
        return _evaluate(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent too large (max {MAX_EXPONENT})")
        return _BINARY_OPS[type(node.op)](left, right)
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        return _FUNCTIONS[node.func.id](*(_evaluate(arg) for arg in node.args))
    raise ValueError(f"unsupported expression element: {ast.dump(node)[:60]}")


def _calculate(args: dict[str, Any]) -> str:
    expression = str(args.get("expression", "")).strip()
    if not expression:
        return _err("expression is required, e.g. '(3 + 4) * 2'")
    if len(expression) > MAX_EXPRESSION_CHARS:
        return _err(f"expression too long (max {MAX_EXPRESSION_CHARS} characters)")
    try:
        result = _evaluate(ast.parse(expression.replace("^", "**"), mode="eval"))
    except (SyntaxError, ValueError, TypeError, ZeroDivisionError, OverflowError) as exc:
        return _err(f"could not evaluate {expression!r}: {exc}")
    return _ok({"expression": expression, "result": result})
