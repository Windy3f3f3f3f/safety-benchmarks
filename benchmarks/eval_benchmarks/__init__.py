# eval_benchmarks - Local security benchmarks for inspect_ai evaluation
#
# This package is registered as an inspect_ai plugin via entry points
# in pyproject.toml. The _registry module imports all @task functions
# to make them discoverable by inspect_ai.

# Import benchmarks with graceful error handling for missing dependencies
_all_modules = []

try:
    from . import raccoon  # noqa: F401
    _all_modules.append("raccoon")
except Exception as e:
    print(f"⚠️  Skipping raccoon (missing dependencies): {type(e).__name__}")

try:
    from . import overthink  # noqa: F401
    _all_modules.append("overthink")
except Exception as e:
    print(f"⚠️  Skipping overthink (missing dependencies): {type(e).__name__}")

try:
    from . import privacylens  # noqa: F401
    _all_modules.append("privacylens")
except Exception as e:
    print(f"⚠️  Skipping privacylens (missing dependencies): {type(e).__name__}")

try:
    from . import personalized_safety  # noqa: F401
    _all_modules.append("personalized_safety")
except Exception as e:
    print(f"⚠️  Skipping personalized_safety (missing dependencies): {type(e).__name__}")

try:
    from . import mm_safety_bench  # noqa: F401
    _all_modules.append("mm_safety_bench")
except Exception as e:
    print(f"⚠️  Skipping mm_safety_bench (missing dependencies): {type(e).__name__}")

__all__ = _all_modules
