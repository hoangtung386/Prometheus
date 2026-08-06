"""Static contract for the training notebook.

The notebook is the supported training workstation, but nothing else checks it: a rewritten
cell can drop a variable that a later cell depends on, and the failure only shows up on a
Colab GPU minutes into a session. That happened — an edit to the audit cell removed the
``build_datamodule`` call, and three later cells raised ``NameError: name 'batch' is not
defined``.

These checks run in CI, in milliseconds, without a GPU or a dataset:

* every ``prometheus`` symbol the notebook imports exists;
* no module-level name is used before a preceding cell binds it.

They are static. They do not prove the notebook runs, only that it cannot fail in these two
specific ways.
"""

from __future__ import annotations

import ast
import builtins
import importlib
import json
import re
from pathlib import Path

import pytest

NOTEBOOK = Path(__file__).resolve().parents[2] / "notebooks" / "train.ipynb"

# Names Colab/IPython injects, plus the placeholders that replace `!shell` and `%magic` lines.
_IPYTHON_NAMES = frozenset({"get_ipython", "display", "In", "Out", "exit", "quit"})
_BUILTINS = frozenset(dir(builtins))


def _code_cells() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _strip_magics(source: str) -> str:
    """Replace IPython shell/magic lines with ``pass``, preserving block structure."""
    return "\n".join(
        (re.match(r"\s*", line).group(0) + "pass") if line.strip().startswith(("!", "%")) else line
        for line in source.splitlines()
    )


def _bound_names(node: ast.AST) -> set[str]:
    """Every name this statement binds, anywhere inside it, including nested scopes.

    Being generous here is deliberate: comprehension and function-local targets cannot leak,
    so counting them only risks missing an error, never inventing one.
    """
    names: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
            names.add(child.id)
        elif isinstance(child, ast.alias):
            names.add((child.asname or child.name).split(".")[0])
        elif isinstance(child, ast.arg):
            names.add(child.arg)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) or (
            isinstance(child, ast.ExceptHandler) and child.name
        ):
            names.add(child.name)
        elif isinstance(child, (ast.Global, ast.Nonlocal)):
            names.update(child.names)
    return names


def _module_level_loads(node: ast.AST) -> set[str]:
    """Names read at module level, skipping deferred bodies.

    A function body may reference a global defined by a later cell, so its loads are not a
    use-before-definition. Module-level loads execute immediately and must already be bound.
    """
    names: set[str] = set()
    stack: list[ast.AST] = [node]
    while stack:
        current = stack.pop()
        for child in ast.iter_child_nodes(current):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                names.add(child.id)
            stack.append(child)
    return names


def _assignment_targets(node: ast.AST) -> set[str]:
    """Top-level assignment targets of a statement, which bind only *after* its value runs.

    Without this, ``batch = batch.to(device)`` looks safe because the statement binds
    ``batch`` somewhere — even though the right-hand side reads a ``batch`` that an earlier
    cell was supposed to define. Comprehension and loop targets are deliberately excluded;
    they are scoped to the expression and must stay maskable.
    """
    if isinstance(node, ast.Assign):
        targets: list[ast.AST] = list(node.targets)
    elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
    else:
        return set()
    return {
        child.id
        for target in targets
        for child in ast.walk(target)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store)
    }


def _deleted_names(node: ast.AST) -> set[str]:
    return {
        target.id
        for child in ast.walk(node)
        if isinstance(child, ast.Delete)
        for target in child.targets
        if isinstance(target, ast.Name)
    }


@pytest.mark.parametrize(("index", "source"), list(enumerate(_code_cells())))
def test_every_code_cell_parses(index: int, source: str) -> None:
    ast.parse(_strip_magics(source))


def test_no_name_is_used_before_a_previous_cell_defines_it() -> None:
    bound = set(_IPYTHON_NAMES | _BUILTINS)
    problems: list[str] = []

    for index, source in enumerate(_code_cells()):
        for statement in ast.parse(_strip_magics(source)).body:
            local = _bound_names(statement) - _assignment_targets(statement)
            undefined = _module_level_loads(statement) - bound - local
            if undefined:
                problems.append(f"cell {index}, line {statement.lineno}: {sorted(undefined)}")
            bound |= local | _assignment_targets(statement)
            bound -= _deleted_names(statement)

    assert not problems, "names used before definition:\n  " + "\n  ".join(problems)


def test_every_imported_prometheus_symbol_exists() -> None:
    problems: list[str] = []
    for index, source in enumerate(_code_cells()):
        for node in ast.walk(ast.parse(_strip_magics(source))):
            if not (isinstance(node, ast.ImportFrom) and node.module):
                continue
            if not node.module.startswith("prometheus"):
                continue
            module = importlib.import_module(node.module)
            for alias in node.names:
                if hasattr(module, alias.name):
                    continue
                try:  # a submodule is importable even before it is an attribute
                    importlib.import_module(f"{node.module}.{alias.name}")
                except ImportError:
                    problems.append(f"cell {index}: {node.module}.{alias.name}")

    assert not problems, "notebook imports that no longer exist:\n  " + "\n  ".join(problems)


def test_the_notebook_uses_the_official_tissue_metric() -> None:
    # The old `result.tissue_dice` reported a NaN-skipping average that hid a collapsed class.
    source = "\n".join(_code_cells())

    assert "tissue_micro_dice" in source
    assert "result.tissue_dice" not in source
