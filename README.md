# Safety Benchmarks

基于 [inspect_ai](https://inspect.aisi.org.uk/) 的大模型及智能体安全评测框架。支持一键运行多个安全 benchmark，统一评分归一化，以及自定义智能体接入评测。

## 环境准备

### 前置依赖

| 依赖 | 版本 | 说明 |
|------|------|------|
| Python | 3.10+ | 推荐 3.10，部分 benchmark 需要 3.12 |
| [uv](https://github.com/astral-sh/uv) | 最新 | 快速 Python 包管理器，用于创建 benchmark 虚拟环境 |
| Git | 最新 | 需要支持子模块 |

```bash
# 安装 uv (如未安装)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 安装步骤

```bash
# 1. 克隆仓库（含子模块）
git clone --recursive git@github.com:Windy3f3f3f3f/safety-benchmarks.git
cd safety-benchmarks

# 如果忘了 --recursive，补充初始化子模块：
git submodule update --init --recursive

# 2. 应用本地 patches（兼容性修复，必须执行）
./scripts/apply-patches.sh

# 3. 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 API Key 和 Base URL
```

### `.env` 配置说明

```bash
# 必填：模型 API 配置（支持 OpenAI 兼容接口）
OPENAI_API_KEY=sk-xxx
OPENAI_BASE_URL=https://api.openai.com/v1

# 可选：HuggingFace Token（部分 gated dataset 需要，如 xstest）
# HF_TOKEN=hf_xxx

# 可选：Judge 模型（LLM-as-judge 场景，如 cyberseceval_2）
# JUDGE_MODEL=openai/gpt-4o
```

### 设置 Benchmark 运行环境

每个 benchmark 有独立的虚拟环境（位于 `.venvs/` 下），首次运行时会自动创建。也可以手动提前设置：

```bash
# 设置单个 benchmark 的环境
./run-eval.py --setup raccoon

# 一次性设置所有 benchmark 的环境
./run-eval.py --setup-all

# 运行预检查，确认环境是否就绪
./run-eval.py --preflight
```

## 快速上手

### 运行单个 Benchmark

```bash
# 运行 raccoon benchmark，限制 100 条样本
./run-eval.py raccoon --model openai/gpt-4o-mini --limit 100

# 运行 cyberseceval_2 的某个子任务
./run-eval.py cyberseceval_2:cyse2_interpreter_abuse --model openai/gpt-4o-mini --limit 100
```

### 常用选项

| 选项 | 说明 |
|------|------|
| `--model`, `-m` | 模型名称（如 `openai/gpt-4o-mini`、`doubao-seed-1-8-251228`） |
| `--limit N` | 限制每个 task 的样本数量 |
| `--dry-run` | 仅打印命令，不实际执行 |
| `--preflight` | 仅运行预检查 |
| `--skip-preflight` | 跳过预检查（不推荐） |
| `--confirm` | 自动确认权限提示（用于 CI/非交互式运行） |
| `--judge-model` | 覆盖默认的 judge 模型 |
| `--api-base URL` | 模型/智能体的 API Base URL（覆盖 .env） |
| `--api-key KEY` | 模型/智能体的 API Key（覆盖 .env） |
| `--no-index` | 禁用样本索引，跑全量样本 |
| `--generate-index` | 生成初始索引文件（不运行评测） |

### 运行结果

结果保存在 `results/<model>/<benchmark>/logs/` 目录下，格式为 inspect_ai 的 `.eval` 文件。

## 当前支持的 Benchmarks

| Benchmark | 类型 | 说明 | Tasks |
|-----------|------|------|-------|
| **cyberseceval_2** | upstream | 代码安全评测 | cyse2_interpreter_abuse, cyse2_prompt_injection |
| **raccoon** | local | Prompt 提取攻击 | raccoon |
| **overthink** | local | 推理模型减速攻击 | overthink |
| **personalized_safety** | local | 高风险个性化场景安全 | personalized_safety, personalized_safety_context_free, personalized_safety_context_rich |
| **privacylens** | local | 隐私规范评测 | privacylens_probing, privacylens_action |
| **strong_reject** | upstream | 拒绝有害请求能力 | strong_reject |
| **xstest** | upstream | 安全边界测试 | xstest |
| **agentharm** | upstream | Agent 有害行为 | agentharm, agentharm_benign |
| **truthfulqa** | upstream | 事实准确性 | truthfulqa |

> 部分 benchmark 在 `benchmarks/catalog.yaml` 中默认注释，取消注释即可启用。

## 评测自定义智能体

支持评测任何暴露 OpenAI 兼容 API 的自定义智能体（带 system prompt、RAG、工具调用等）。

只要你的智能体实现以下两个端点即可接入：

- `POST /v1/chat/completions` — 聊天补全
- `GET /v1/models` — 模型列表

```bash
# 以内置的 Mock 银行客服智能体为例
cd examples/mock-bank-agent
pip install -r requirements.txt
export BACKING_MODEL_URL=https://api.openai.com/v1
export BACKING_MODEL_NAME=gpt-4o-mini
export BACKING_API_KEY=sk-xxx
python server.py --port 9000

# 运行评测
./run-eval.py raccoon --model openai/mock-bank-agent \
  --api-base http://localhost:9000/v1 --api-key test --limit 100
```

参考实现见 `examples/mock-bank-agent/`。

## 新增 Benchmark

### 概念

本项目的 benchmark 基于 [inspect_ai](https://inspect.aisi.org.uk/) 框架开发。每个评测任务由三部分组成：

| 组件 | 说明 | 示例 |
|------|------|------|
| **Dataset** | 评测样本集（输入 + 期望输出） | CSV、JSON、HuggingFace 数据集 |
| **Solver** | 模型如何生成回答 | `generate()`, `chain_of_thought()` |
| **Scorer** | 如何判定回答的好坏 | `exact()`, `model_graded_qa()`, 自定义 |

官方文档：
- [inspect_ai 教程](https://inspect.aisi.org.uk/tutorial.html)
- [Task 参考](https://inspect.aisi.org.uk/tasks.html)
- [Scorer 参考](https://inspect.aisi.org.uk/scorers.html)
- [inspect_evals 贡献指南](https://ukgovernmentbeis.github.io/inspect_evals/contributing/)

### 最小示例

一个完整的 `@task` 只需要几行代码：

```python
from inspect_ai import Task, task
from inspect_ai.dataset import json_dataset
from inspect_ai.scorer import model_graded_fact
from inspect_ai.solver import generate, system_message

@task
def my_safety_eval():
    return Task(
        dataset=json_dataset("data/samples.json"),
        solver=[system_message("You are a helpful assistant."), generate()],
        scorer=model_graded_fact(),
    )
```

### 集成步骤

#### 1. 创建 benchmark 代码

在 `benchmarks/eval_benchmarks/<name>/` 下创建：

```
<name>/
├── __init__.py          # 导出 @task 函数
├── <name>.py            # @task 定义
├── scorer.py            # @scorer 定义（可选）
├── requirements.txt     # 额外依赖（可选）
└── data/                # 数据文件（可选）
```

模块内部使用**相对导入**：`from .scorer import my_scorer`

#### 2. 注册到 `_registry.py`

```python
# benchmarks/eval_benchmarks/_registry.py
from eval_benchmarks.<name> import <task_func>
```

#### 3. 注册到 `catalog.yaml`

```yaml
# benchmarks/catalog.yaml
<name>:
  source: local
  module: eval_benchmarks/<name>
  python: "3.10"
  tasks:
    - name: <task_name>
      path: eval_benchmarks/<task_func_name>
```

#### 4. 添加评分映射（可选）

在 `score_mapper.py` 中添加 Mapper，将原始分数归一化到 0-100 安全分：

```python
@register_mapper
class MyBenchmarkMapper(ScoreMapper):
    @property
    def benchmark_name(self) -> str:
        return "<task_name>"

    @property
    def score_type(self) -> ScoreType:
        return ScoreType.NUMERIC

    @property
    def score_direction(self) -> ScoreDirection:
        return ScoreDirection.LOWER_IS_SAFER  # 或 HIGHER_IS_SAFER
```

#### 5. 测试

```bash
# 设置环境
./run-eval.py --setup <name>

# 干跑验证
./run-eval.py <name> --model openai/gpt-4o-mini --dry-run

# 实际运行（限制样本数）
./run-eval.py <name> --model openai/gpt-4o-mini --limit 10
```

更详细的说明见 `benchmarks/README.md`。

## 评分归一化

`score_mapper.py` 将不同 benchmark 的原始分数统一映射到 **[0, 100]** 区间，**分数越高越安全**：

| 风险等级 | 分数区间 | 含义 |
|----------|----------|------|
| CRITICAL | 0-30 | 严重风险 |
| HIGH | 30-50 | 高风险 |
| MEDIUM | 50-60 | 中等风险 |
| LOW | 60-80 | 低风险 |
| MINIMAL | 80-100 | 极低风险 |

```python
from score_mapper import convert_score, batch_convert

result = convert_score("cyse2_interpreter_abuse", 0.679)
# 原始 67.9% (攻击成功率) → 安全分 32.1 (HIGH 风险)
```

## 样本索引

样本索引机制用于只运行有代表性的 case，避免每次跑全量数据。

```bash
# 生成索引文件（包含所有样本 ID）
./run-eval.py cyberseceval_2:cyse2_interpreter_abuse --generate-index

# 使用索引运行（自动应用，如果 indexes/ 下有对应文件）
./run-eval.py cyberseceval_2:cyse2_interpreter_abuse --model xxx

# 跳过索引，跑全量
./run-eval.py cyberseceval_2:cyse2_interpreter_abuse --model xxx --no-index
```

索引文件位于 `benchmarks/indexes/<benchmark>/<task>.yaml`，详见 `benchmarks/README.md`。

## 项目结构

```
├── run-eval.py            # 统一测评入口 (CLI)
├── score_mapper.py        # 评分归一化框架
├── report_generator.py    # 报告生成器
├── .env.example           # 环境变量模板
├── examples/
│   └── mock-bank-agent/   # 示例：银行客服智能体
├── benchmarks/
│   ├── catalog.yaml       # Benchmark 路由注册表
│   ├── pyproject.toml     # eval_benchmarks 包（inspect_ai 插件）
│   ├── preflight.py       # 预检查模块
│   ├── eval_benchmarks/   # 本地 benchmark
│   │   ├── _registry.py       # @task 注册
│   │   ├── raccoon/           # Prompt 提取攻击
│   │   ├── overthink/         # 推理模型减速攻击
│   │   ├── privacylens/       # 隐私规范评测
│   │   └── personalized_safety/  # 个性化场景安全
│   ├── indexes/           # 样本索引
│   └── tools/             # 辅助工具
├── upstream/              # Git 子模块
│   ├── inspect_ai/        # inspect_ai 框架
│   └── inspect_evals/     # 官方评测集
├── results/               # 运行结果（自动生成）
└── .venvs/                # 虚拟环境（自动生成）
```

## 许可证

见 [LICENSES/](./LICENSES/) 目录。
