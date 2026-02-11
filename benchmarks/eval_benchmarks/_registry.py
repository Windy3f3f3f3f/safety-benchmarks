# ruff: noqa: F401
# Import all @task functions to register them with inspect_ai's registry.
# When this module is loaded via the entry point, all tasks become discoverable.

from eval_benchmarks.raccoon import raccoon
from eval_benchmarks.overthink import overthink
from eval_benchmarks.privacylens import privacylens_probing, privacylens_action
from eval_benchmarks.personalized_safety import personalized_safety, personalized_safety_context_free, personalized_safety_context_rich
from eval_benchmarks.hallulens import (
    hallulens_task1_precise_wikiqa,
    hallulens_task2_longwiki,
    hallulens_task3_nonsense_mixed_entities,
    hallulens_task3_round_robin_nonsense_name,
)
