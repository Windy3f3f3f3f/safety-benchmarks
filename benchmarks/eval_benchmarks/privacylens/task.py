"""
PrivacyLens: 隐私规范评测任务

参考论文: PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action
https://github.com/SALT-NLP/PrivacyLens
"""

from typing import Literal

from inspect_ai import Task, task
from inspect_ai.model import GenerateConfig, Model
from inspect_ai.solver import generate, system_message

from .dataset import load_probing_dataset, load_action_dataset
from .scorers import probing_scorer, action_leakage_scorer
from .prompts import ACTION_SYSTEM_PROMPT


@task
def privacylens_probing(
    level: Literal["seed", "vignette"] = "seed",
) -> Task:
    """PrivacyLens Probing 任务

    评估模型对隐私规范的理解程度。通过多选题测试模型是否能
    正确识别隐私规范违规场景。

    Args:
        level: 测试级别
            - "seed": 使用抽象的隐私规范参数
            - "vignette": 使用具体的故事场景

    Returns:
        Task: inspect_ai 任务对象
    """
    dataset = load_probing_dataset(level=level)

    return Task(
        dataset=dataset,
        solver=[generate()],
        scorer=probing_scorer(),
        config=GenerateConfig(
            temperature=0,
            max_tokens=64,
        ),
        version="1.0.0",
    )


@task
def privacylens_action(
    judge_model: str | Model | None = None,
) -> Task:
    """PrivacyLens Action 任务

    评估模型在执行任务时是否会泄漏敏感信息。
    模型需要根据用户指令完成任务，同时避免泄漏敏感信息。

    Args:
        judge_model: 用于评估泄漏的 judge 模型

    Returns:
        Task: inspect_ai 任务对象
    """
    dataset = load_action_dataset()

    return Task(
        dataset=dataset,
        solver=[
            system_message(ACTION_SYSTEM_PROMPT),
            generate(),
        ],
        scorer=action_leakage_scorer(judge_model=judge_model),
        config=GenerateConfig(
            temperature=0,
            max_tokens=1024,
        ),
        version="1.0.0",
    )
