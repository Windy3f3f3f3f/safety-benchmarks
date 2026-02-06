# eval_benchmarks - Local security benchmarks for inspect_ai evaluation
#
# This package is registered as an inspect_ai plugin via entry points
# in pyproject.toml. The _registry module imports all @task functions
# to make them discoverable by inspect_ai.

from . import raccoon  # noqa: F401
from . import overthink  # noqa: F401
from . import privacylens  # noqa: F401
from . import personalized_safety  # noqa: F401
from . import wmdp  # noqa: F401


__all__ = ["raccoon", "overthink", "privacylens", "personalized_safety", "wmdp"]
