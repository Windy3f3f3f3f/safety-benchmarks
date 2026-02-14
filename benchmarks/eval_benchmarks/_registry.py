# ruff: noqa: F401
# Import all @task functions to register them with inspect_ai's registry.
# When this module is loaded via the entry point, all tasks become discoverable.

# Temporarily disabled - missing dependencies
# from eval_benchmarks.raccoon import raccoon
# from eval_benchmarks.overthink import overthink
# from eval_benchmarks.privacylens import privacylens_probing, privacylens_action
# from eval_benchmarks.personalized_safety import personalized_safety, personalized_safety_context_free, personalized_safety_context_rich

# MM-SafetyBench - enabled
from eval_benchmarks.mm_safety_bench import mm_safety_bench_illegal_activity
