"""PrivacyLens 数据集加载"""

import json
import os
from pathlib import Path
from typing import Any, Literal

from inspect_ai.dataset import Dataset, Sample, MemoryDataset

from .prompts import (
    SEED_PROBING_TEMPLATE,
    VIGNETTE_PROBING_TEMPLATE,
    ACTION_USER_TEMPLATE,
)


def get_data_path() -> Path:
    """获取数据文件路径"""
    # 优先从环境变量获取
    env_path = os.environ.get("PRIVACYLENS_DATA_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # 默认使用本地数据
    default_path = Path(__file__).parent / "data" / "main_data.json"
    if default_path.exists():
        return default_path

    raise FileNotFoundError(
        "PrivacyLens 数据集未找到。请确保 data/main_data.json 存在，"
        "或设置环境变量 PRIVACYLENS_DATA_PATH 指向数据文件。"
    )


def load_raw_data() -> list[dict[str, Any]]:
    """加载原始 JSON 数据"""
    data_path = get_data_path()
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_article(word: str) -> str:
    """获取正确的冠词 a/an"""
    vowels = "aeiouAEIOU"
    return "an" if word and word[0] in vowels else "a"


def to_gerund(verb_phrase: str) -> str:
    """将动词短语转换为现在分词形式"""
    # 简单处理常见情况
    words = verb_phrase.split()
    if not words:
        return verb_phrase

    verb = words[0].lower()
    rest = " ".join(words[1:])

    # 常见动词的现在分词形式
    gerund_map = {
        "send": "sending",
        "share": "sharing",
        "post": "posting",
        "tell": "telling",
        "give": "giving",
        "show": "showing",
        "forward": "forwarding",
        "disclose": "disclosing",
        "reveal": "revealing",
        "mention": "mentioning",
        "discuss": "discussing",
        "announce": "announcing",
    }

    if verb in gerund_map:
        gerund = gerund_map[verb]
    elif verb.endswith("e"):
        gerund = verb[:-1] + "ing"
    elif len(verb) > 2 and verb[-1] not in "aeiou" and verb[-2] in "aeiou":
        gerund = verb + verb[-1] + "ing"
    else:
        gerund = verb + "ing"

    return f"{gerund} {rest}".strip() if rest else gerund


def create_probing_sample(
    record: dict[str, Any],
    level: Literal["seed", "vignette"],
) -> Sample:
    """创建 probing 任务的样本"""
    seed = record["seed"]
    name = record["name"]

    if level == "seed":
        # Seed level: 使用抽象参数
        data_type = seed["data_type"]
        article = get_article(data_type)

        prompt = SEED_PROBING_TEMPLATE.format(
            data_sender_name=seed["data_sender_name"],
            data_sender=seed["data_sender"],
            data_subject=seed["data_subject"],
            data_type=f"{article} {data_type}",
            data_recipient=seed["data_recipient"],
            transmission_principle=to_gerund(seed["transmission_principle"]),
        )
    else:
        # Vignette level: 使用具体故事
        vignette = record["vignette"]
        prompt = VIGNETTE_PROBING_TEMPLATE.format(story=vignette["story"])

    return Sample(
        id=name,
        input=prompt,
        target="B",  # 正确答案是拒绝分享 (B) No
        metadata={
            "name": name,
            "level": level,
            "data_type": seed["data_type"],
            "source": seed.get("source", ""),
        },
    )


def create_action_sample(record: dict[str, Any]) -> Sample:
    """创建 action 任务的样本"""
    trajectory = record["trajectory"]
    seed = record["seed"]
    name = record["name"]

    # 格式化工具列表
    toolkits = trajectory.get("toolkits", [])
    if isinstance(toolkits, list):
        toolkits_str = ", ".join(toolkits)
    else:
        toolkits_str = str(toolkits)

    prompt = ACTION_USER_TEMPLATE.format(
        user_name=trajectory["user_name"],
        user_email=trajectory["user_email"],
        user_instruction=trajectory["user_instruction"],
        toolkits=toolkits_str,
    )

    # 敏感信息
    sensitive_items = trajectory.get("sensitive_info_items", [])
    if isinstance(sensitive_items, list):
        sensitive_info = "; ".join(sensitive_items)
    else:
        sensitive_info = str(sensitive_items)

    return Sample(
        id=name,
        input=prompt,
        target="No",  # 目标是不泄漏信息
        metadata={
            "name": name,
            "user_name": trajectory["user_name"],
            "user_instruction": trajectory["user_instruction"],
            "sensitive_info": sensitive_info,
            "data_type": seed["data_type"],
            "executable_trajectory": trajectory.get("executable_trajectory", ""),
            "reference_final_action": trajectory.get("final_action", ""),
        },
    )


def load_probing_dataset(
    level: Literal["seed", "vignette"] = "seed",
) -> Dataset:
    """加载 probing 任务数据集"""
    raw_data = load_raw_data()
    samples = [create_probing_sample(record, level) for record in raw_data]
    return MemoryDataset(samples=samples, name=f"privacylens_probing_{level}")


def load_action_dataset() -> Dataset:
    """加载 action 任务数据集"""
    raw_data = load_raw_data()
    samples = [create_action_sample(record) for record in raw_data]
    return MemoryDataset(samples=samples, name="privacylens_action")
