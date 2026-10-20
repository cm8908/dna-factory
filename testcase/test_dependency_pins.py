"""
Test cases for the dependency pins in pyproject.toml

`[tool.uv] override-dependencies` re-pins some core packages so the optional `unsloth` extra (which caps
them lower) resolves. An override silently wins over the core pin in `uv lock`, so bumping only the core pin
leaves the lock on the old version with no error. These tests keep the two in sync without a GPU or network.
"""

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def find_override_mismatches(pyproject):
    """Return (package, core specifier, override specifier) for every override that differs from its core pin"""
    core = {}
    for dep in pyproject.get("project", {}).get("dependencies", []):
        req = Requirement(dep)
        core[canonicalize_name(req.name)] = req.specifier

    mismatches = []
    for dep in pyproject.get("tool", {}).get("uv", {}).get("override-dependencies", []):
        req = Requirement(dep)
        name = canonicalize_name(req.name)
        # Overrides for packages outside the core dependencies (e.g. unsloth-only) are allowed
        if name in core and set(core[name]) != set(req.specifier):
            mismatches.append((name, str(core[name]), str(req.specifier)))
    return mismatches


def test_overrides_match_core_pins():
    """Every override of a core dependency in the real pyproject.toml must use the identical specifier"""
    with open(PYPROJECT, "rb") as f:
        mismatches = find_override_mismatches(tomllib.load(f))

    details = "\n".join(
        f"  {name}: core '{core}' vs override '{override}'"
        for name, core, override in mismatches
    )
    assert not mismatches, (
        "Core pins and `[tool.uv] override-dependencies` disagree:\n"
        f"{details}\n"
        "Keep `[tool.uv] override-dependencies` in sync with the core pins "
        "(an override silently wins over the core pin in `uv lock`)."
    )


def test_find_override_mismatches_detects_drift():
    """A bumped core pin with a stale override is reported; non-core overrides and no overrides are ignored"""
    pyproject = {
        "project": {"dependencies": ["transformers==5.18.0", "Datasets == 5.1.0"]},
        "tool": {
            "uv": {
                "override-dependencies": [
                    "transformers==5.19.0",
                    "datasets==5.1.0",
                    "unsloth==1.0",
                ]
            }
        },
    }
    assert find_override_mismatches(pyproject) == [
        ("transformers", "==5.18.0", "==5.19.0")
    ]

    del pyproject["tool"]["uv"]["override-dependencies"]
    assert find_override_mismatches(pyproject) == []
