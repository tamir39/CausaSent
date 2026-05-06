"""Standalone test runner — works without pytest installed.

Discovers every `test_*` function in `tests/test_*.py`, runs them, prints
pass/fail summary, exits non-zero on any failure. Compatible with pytest:
just `pytest tests/` once pytest is installed.

Usage:
    python -m tests.run_all
"""
from __future__ import annotations

import importlib
import inspect
import pkgutil
import sys
import traceback
from pathlib import Path


def _discover_modules() -> list[str]:
    pkg = Path(__file__).parent
    sys.path.insert(0, str(pkg.parent))  # repo root
    mods = []
    for info in pkgutil.iter_modules([str(pkg)]):
        if info.name.startswith("test_"):
            mods.append(f"tests.{info.name}")
    return sorted(mods)


def main() -> int:
    mods = _discover_modules()
    total = passed = failed = 0
    failures: list[tuple[str, str, str]] = []

    for mod_name in mods:
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            failed += 1
            total += 1
            failures.append((mod_name, "<import>", traceback.format_exc()))
            print(f"  [IMPORT FAIL] {mod_name}")
            continue

        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            if fn.__module__ != mod_name:
                continue
            total += 1
            try:
                fn()
            except Exception:
                failed += 1
                failures.append((mod_name, name, traceback.format_exc()))
                print(f"  FAIL  {mod_name}.{name}")
            else:
                passed += 1
                print(f"  ok    {mod_name}.{name}")

    print("\n" + "=" * 60)
    print(f"{passed}/{total} tests passed, {failed} failed")
    if failures:
        print("\n--- failures ---")
        for mod, name, tb in failures:
            print(f"\n[{mod}.{name}]")
            print(tb)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
