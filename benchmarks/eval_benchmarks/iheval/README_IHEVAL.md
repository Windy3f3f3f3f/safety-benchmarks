# IHEval Benchmark

Instruction Hierarchy Evaluation benchmark - 评估模型遵循指令层级的能力。

## 目录结构

```
iheval/
├── dataset.py              # 数据加载模块
├── scorer.py               # 评分模块（集成 IFEval）
├── iheval.py               # @task 定义
├── __init__.py             # 模块导出
├── requirements.txt        # 依赖包
│
├── data/                   # 数据目录
│   ├── rule-following/     # 规则遵循数据 (182M)
│   ├── task-execution/     # 任务执行数据 (63M)
│   ├── safety/             # 安全场景数据 (55M)
│   └── tool-use/          # 工具使用数据 (61M)
│
└── evaluate/               # IFEval 评估脚本
    ├── evaluation_main.py
    ├── instructions.py
    ├── instructions_registry.py
    └── instructions_util.py
```

## 数据说明

- **rule-following**: 单轮和多轮指令遵循（基于 IFEval）
- **task-execution**: NLP 任务（动词提取、翻译、语言检测）
- **safety**: 安全防御场景（劫击攻击、提示词提取）
- **tool-use**: 函数调用场景（intrinsic、injected）

## 数据来源

数据来自 [IHEval](https://arxiv.org/pdf/2502.08745) benchmark，包含约 3,538 个样本，涵盖 9 个任务。

## 评估设置

每个任务支持三种设置：
- **aligned**: 所有输入与系统指令一致
- **conflict**: 低优先级输入与系统指令冲突
- **reference**: 所有指令合并到用户消息（基线）

## 使用方法

```bash
# 使用默认参数
./run-eval.py iheval --model openai/gpt-4o-mini --limit 100

# 使用预配置任务
./run-eval.py iheval:iheval_rule_multi_aligned --model openai/gpt-4o-mini --limit 100
./run-eval.py iheval:iheval_rule_multi_conflict --model openai/gpt-4o-mini --limit 100
```

## 评分说明

- **rule-following**: 使用 IFEval 官方评分（strict + loose 平均）
- **task-execution**: 基于任务类型的评分（F1、精确匹配等）
- **safety**: 基于标签的评分（访问控制）
- **tool-use**: 基于规则的评分

所有评分归一化到 [0, 1] 区间，高分表示模型更安全。
