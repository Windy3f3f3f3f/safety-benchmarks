from pydantic import BaseModel, Field
from typing import Literal

# 这个列表应该和你评分标准中的选项保持一致
CHOICES = ["A", "B", "C", "D", "E"]

class MSSScorerResponse(BaseModel):
    """
    一个 Pydantic 模型，用于解析和验证 Judge 模型对安全评估的响应。
    """
    choice: Literal[tuple(CHOICES)] = Field(
        description="对应于安全评估的多项选择题答案（例如 'A', 'B' 等）。"
    )
    reasoning: str = Field(
        description="对做出该选择的详细解释和理由。"
    )