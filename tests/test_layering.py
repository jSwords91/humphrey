"""The dependency arrow, enforced: app -> providers -> brain, never back.

brain/ is the portable agent core. It may use the standard library,
pydantic and itself, and nothing that knows about this app, a provider, a
UI or storage. providers/ may use brain/, its vendor SDKs, and one
documented piece of app/: its settings (see providers/__init__.py).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _imports(package: str) -> list[tuple[str, str]]:
    """(file, imported module) for every import in a package's modules."""
    found: list[tuple[str, str]] = []
    for path in sorted((ROOT / package).rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                found += [(path.name, alias.name) for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                found.append((path.name, node.module))
    return found


def _top(module: str) -> str:
    return module.split(".")[0]


def test_brain_imports_only_the_standard_library_pydantic_and_itself() -> None:
    allowed = set(sys.stdlib_module_names) | {"__future__", "pydantic", "brain"}
    outside = [(f, m) for f, m in _imports("brain") if _top(m) not in allowed]
    assert outside == []


@pytest.mark.parametrize("package", ["brain", "providers"])
def test_nothing_below_app_imports_app_beyond_its_settings(package: str) -> None:
    allowed_from_app = {"app.settings"} if package == "providers" else set()
    reaching_up = [
        (f, m) for f, m in _imports(package) if _top(m) == "app" and m not in allowed_from_app
    ]
    assert reaching_up == []


def test_brain_does_not_import_providers() -> None:
    assert [(f, m) for f, m in _imports("brain") if _top(m) == "providers"] == []
