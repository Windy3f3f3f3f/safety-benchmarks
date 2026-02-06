"""
PrivacyLens: 隐私规范评测

评估模型对隐私规范的理解和遵守程度。
包含两个主要任务:
- privacylens_probing: 多选题测试隐私规范认知
- privacylens_action: 评估智能体行动中的信息泄漏
"""

from .task import privacylens_probing, privacylens_action

__all__ = ["privacylens_probing", "privacylens_action"]
