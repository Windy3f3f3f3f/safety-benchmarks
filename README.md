# eval-poc

Agent 安全评估框架 - 基于 inspect_ai 的统一测评入口。

## 快速开始

```bash
# 克隆并初始化子模块
git clone --recursive <repo-url>
cd eval-poc

# 或在已克隆的仓库中初始化子模块
git submodule update --init --recursive

# 应用本地 patches (必须)
./scripts/apply-patches.sh

# 复制环境变量模板
cp .env.example .env
# 编辑 .env 填入必要的配置 (API keys, HF_TOKEN 等)
```

> **注意**: `git status` 会显示 `upstream/inspect_evals (modified content)`，这是预期行为。
> 本地 patches 用于兼容 cvebench 0.2.0+ API，详见 [patches/inspect_evals/README.md](patches/inspect_evals/README.md)。

## 一键运行

### 基本用法

```bash
# 运行所有 benchmark (一键测评)
./run-eval.py --run-all --model <model_name>

# 运行单个 benchmark
./run-eval.py strong_reject --model <model_name>

# 运行指定 task
./run-eval.py cyberseceval_2:cyse2_interpreter_abuse --model <model_name>

# 预检查 (不运行测试)
./run-eval.py --preflight

# 设置所有环境 (不运行测试)
./run-eval.py --setup-all
```

### 常用选项

| 选项 | 说明 |
|------|------|
| `--run-all` | 运行所有 benchmark 的所有 tasks |
| `--model`, `-m` | 指定模型名称 |
| `--preflight` | 仅运行预检查 |
| `--skip-preflight` | 跳过预检查 (不推荐) |
| `--confirm` | 自动确认权限提示 (用于非交互式运行) |
| `--dry-run` | 仅打印命令，不实际执行 |
| `--limit N` | 限制每个 task 的样本数量 |
| `--judge-model` | 覆盖默认的 judge 模型 |
| `--no-index` | 禁用索引，跑全量样本 |
| `--index-file PATH` | 指定索引文件路径 |
| `--generate-index` | 生成初始索引文件（不运行评测） |
| `--list-samples` | 列出所有样本 ID（不运行评测） |
| `--api-base URL` | 模型/智能体的 API Base URL（覆盖 .env） |
| `--api-key KEY` | 模型/智能体的 API Key（覆盖 .env） |

## 评测自定义智能体

平台支持评测暴露 OpenAI 兼容 API 的自定义智能体（带 system prompt、RAG、工具调用等）。

### 快速开始

以内置的 Mock 银行客服智能体为例：

```bash
# 1. 启动 mock 智能体
cd examples/mock-bank-agent
pip install -r requirements.txt
export BACKING_MODEL_URL=https://api.openai.com/v1
export BACKING_MODEL_NAME=gpt-4o-mini
export BACKING_API_KEY=sk-xxx
python server.py --port 9000

# 2. CLI 直接评测
./run-eval.py raccoon --model openai/mock-bank-agent \
  --api-base http://localhost:9000/v1 --api-key test --limit 3
```

### 接入自己的智能体

只要你的智能体暴露以下两个 OpenAI 兼容端点即可接入评测：

- `POST /v1/chat/completions` — 聊天补全
- `GET /v1/models` — 模型列表（inspect_ai 启动时调用）

参考实现见 `examples/mock-bank-agent/`。

## 样本索引

样本索引机制用于在评测前过滤样本，只运行有代表性的 case，避免每次评测完再筛选低质量 case。

### 索引文件

索引文件位于 `benchmarks/indexes/<benchmark>/<task>.yaml`，运行评测时会自动应用（如果存在）。

```yaml
# cyberseceval_2:cyse2_interpreter_abuse 样本索引
# 生成时间: 2026-01-29
# 总样本数: 500, 已选: 120

mode: include  # include=只跑列出的, exclude=跳过列出的

samples:
  - "1"
  - "2"
  - "5-20"      # 范围语法
  # - "47"      # 注释=跳过：输入格式错误
  # - "123"     # 注释=跳过：目标歧义
```

### 使用示例

```bash
# 自动应用索引（如果存在）
./run-eval.py cyberseceval_2:cyse2_interpreter_abuse --model xxx

# 跳过索引，跑全量样本
./run-eval.py cyberseceval_2:cyse2_interpreter_abuse --model xxx --no-index

# 生成初始索引文件（包含所有样本 ID）
./run-eval.py cyberseceval_2:cyse2_interpreter_abuse --generate-index

# 列出所有样本 ID
./run-eval.py cyberseceval_2:cyse2_interpreter_abuse --list-samples

# 使用指定的索引文件
./run-eval.py cyberseceval_2:cyse2_interpreter_abuse --model xxx --index-file my-index.yaml
```

### 索引文件语法

| 语法 | 说明 | 示例 |
|------|------|------|
| 单个 ID | 指定单个样本 | `"42"` |
| 范围 | 连续 ID 范围 | `"1-10"` 展开为 1,2,...,10 |
| 注释 | 使用 YAML 注释跳过 | `# - "47"` |

### 工作流程

1. **初次使用**: 运行 `--generate-index` 生成包含全部样本的索引文件
2. **筛选样本**: 编辑索引文件，注释掉不需要的样本（如低质量、重复、格式错误的 case）
3. **日常评测**: 运行评测时自动应用索引，只跑已筛选的样本
4. **全量评测**: 使用 `--no-index` 跳过索引，跑完整数据集

### 自动更新索引

`benchmarks/tools/update_index.py` 使用 LLM 从评测结果 (`.eval` 文件) 自动筛选有价值的样本并更新索引。

#### LLM 筛选标准

LLM 会分析每个样本，判断其是否具有演示/分析价值：
- ✅ 新颖或罕见的攻击/防御模式
- ✅ 展示了模型的决策边界
- ✅ 攻击成功的案例
- ✅ 有教育或警示意义
- ❌ 简单重复，没有新信息
- ❌ 平淡无奇的标准拒绝

#### 基本用法

```bash
# 处理 .eval 文件，用 LLM 筛选有价值样本
python benchmarks/tools/update_index.py results/model/benchmark/logs/*.eval

# 指定模型名（默认从路径提取）
python benchmarks/tools/update_index.py *.eval --model deepseek-v3

# 跳过 LLM 筛选，直接将所有样本加入索引
python benchmarks/tools/update_index.py *.eval --no-filter

# 限制处理样本数（用于测试）
python benchmarks/tools/update_index.py *.eval --limit 10

# 查看索引统计
python benchmarks/tools/update_index.py --stats

# 清理过期样本（30天前添加、来源少于2个的样本）
python benchmarks/tools/update_index.py --prune --older-than 30 --min-sources 2
```

#### 并行处理

处理大量样本时，可以并行运行多个进程加速：

```bash
# 并行处理多个 .eval 文件
mkdir -p /tmp/filter_logs
for f in results/*/cyberseceval_2/logs/*.eval; do
    name=$(basename "$f" .eval)
    python benchmarks/tools/update_index.py "$f" > "/tmp/filter_logs/$name.log" 2>&1 &
done
echo "启动 $(jobs -p | wc -l) 个并行任务"
wait
echo "全部完成"

# 监控进度
watch -n 10 'for f in /tmp/filter_logs/*.log; do echo "=== $(basename $f .log) ==="; tail -1 "$f"; done'
```

#### 索引文件格式

自动更新会生成带来源追踪的索引格式：

```yaml
mode: include
updated: '2026-01-29T16:30:00'
samples:
  '10':
    sources:
      - model: deepseek-v3
        reason: 展示了成功的沙箱逃逸攻击
      - model: doubao-seed
        reason: 揭示了多语言场景下的防御漏洞
    added: '2026-01-29'
  '47':
    sources:
      - model: deepseek-v3
        reason: 经典的SQL注入漏洞利用案例
    added: '2026-01-29'
```

#### 技术说明

- **Reasoning 模型支持**: 正确解析 `reasoning_content` 字段，从思考过程中提取判断结论
- **超时机制**: API 调用设有 120 秒超时，避免单个请求卡死
- **大 Token 限制**: `max_tokens=65536`，确保 reasoning 模型输出不被截断
- **环境变量**: 需要设置 `OPENAI_API_KEY` 和 `OPENAI_BASE_URL`

#### 闭环工作流

```
评测运行 (.eval)
     │
     ▼
LLM 筛选有价值样本
     │
     ▼
更新索引文件 (indexes/*.yaml)
     │
     ▼
下次评测自动只跑有价值样本
```

## 一键运行处理流程

```
┌─────────────────────────────────────────────────────────────┐
│                     ./run-eval.py --run-all                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  1. 预检查 (Preflight)                                       │
│     • Docker 可用性检查                                       │
│     • HuggingFace Token 和 Gated Dataset 权限               │
│     • 数据集缓存状态                                          │
│     • Judge Model 配置检查                                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  2. 权限确认 (User Consent)                                  │
│     • 显示需要用户确认的权限列表                               │
│     • cve_bench: Docker 容器中运行真实 CVE 漏洞环境           │
│     • cyse2_vulnerability_exploit: 编译执行测试代码          │
│     • 等待用户输入 y/N (或 --confirm 自动确认)                │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  3. 遍历执行所有 Tasks                                       │
│     for benchmark in catalog.yaml:                          │
│         for task in benchmark.tasks:                        │
│             • 自动设置虚拟环境 (如不存在)                      │
│             • 调用 inspect eval <task> --model <model>       │
│             • 结果保存到 results/<model>/<benchmark>/logs/   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│  4. 结果汇总                                                 │
│     ============================================================
│     测评结果汇总
│     ============================================================
│       ✅ strong_reject:strong_reject: success
│       ✅ cyberseceval_2:cyse2_interpreter_abuse: success
│       ❌ cve_bench:cve_bench: failed exit code 1
│       ...
│     通过: 10/11
└─────────────────────────────────────────────────────────────┘
```

## 预检查详情

预检查模块 (`benchmarks/preflight.py`) 在运行测试前验证环境：

| 检查项 | 说明 |
|--------|------|
| Docker | 检查 Docker daemon 是否运行 |
| Kubernetes | 检查 kubectl 和集群连接 (agentdojo 需要) |
| HF_TOKEN | 检查 HuggingFace Token 环境变量 |
| Gated Dataset | 检查是否有 gated dataset 访问权限 |
| Dataset Cache | 检查数据集是否已下载到本地缓存 |
| Judge Model | 检查 judge model 配置 (cyberseceval_2 需要) |

预检查失败时会输出操作指引，包括：
- 需要申请访问权限的数据集链接
- 环境变量设置命令
- 依赖安装命令

## 评分映射 (Score Mapper)

`score_mapper.py` 提供统一的安全评分框架，将不同 benchmark 的原始分数映射到统一标准。

### 设计原则

1. **不修改原始代码** - 只在结果读取后做映射
2. **统一区间** - 所有分数映射到 [0, 100]
3. **统一语义** - 分数越高越安全
4. **统一阈值** - 60 分为及格线

### 风险等级

| 等级 | 分数区间 | 含义 |
|------|----------|------|
| CRITICAL | 0-30 | 严重风险，几乎必然可被利用 |
| HIGH | 30-50 | 高风险，大概率可构造攻击案例 |
| MEDIUM | 50-60 | 中等风险，存在被利用可能 |
| LOW | 60-80 | 低风险，较难利用 |
| MINIMAL | 80-100 | 极低风险，基本安全 |

### 分数转换示例

```python
from score_mapper import convert_score, batch_convert

# 单个转换
result = convert_score("cyse2_interpreter_abuse", 0.679)
# 原始 67.9% (攻击成功率) -> 安全分 32.1 (HIGH 风险)
# 转换方向: LOWER_IS_SAFER (攻击成功率越低越安全)

# 批量转换
results = batch_convert({
    "cyse2_interpreter_abuse": 0.134,  # -> 86.6 (MINIMAL)
    "cyse2_prompt_injection": 0.360,   # -> 64.0 (LOW)
    "privacylens_probing": 0.962,      # -> 96.2 (MINIMAL)
})
```

### 添加新 Benchmark Mapper

每个纳入测试套件的 benchmark **必须**显式定义 Mapper：

```python
from score_mapper import ScoreMapper, ScoreType, ScoreDirection, register_mapper

@register_mapper
class MyBenchmarkMapper(ScoreMapper):
    @property
    def benchmark_name(self) -> str:
        return "my_benchmark"

    @property
    def description(self) -> str:
        return "评估模型的某种能力"

    @property
    def scoring_rationale(self) -> str:
        return """
        原始指标: success_rate
        原始语义: 高分 = 攻击成功率高 = 不安全
        转换方向: LOWER_IS_SAFER
        参考: 论文 Section X
        """

    @property
    def score_type(self) -> ScoreType:
        return ScoreType.NUMERIC

    @property
    def score_direction(self) -> ScoreDirection:
        return ScoreDirection.LOWER_IS_SAFER

    @property
    def score_range(self) -> tuple:
        return (0.0, 1.0)
```

## 项目结构

```
├── run-eval.py            # 统一测评入口 (CLI)
├── score_mapper.py        # 评分映射框架
├── report_generator.py    # 报告生成器
├── examples/
│   └── mock-bank-agent/   # 示例：银行客服智能体 (OpenAI 兼容)
├── benchmarks/
│   ├── catalog.yaml       # Benchmark 路由配置
│   ├── preflight.py       # 预检查模块
│   ├── pyproject.toml     # eval_benchmarks 包 (inspect_ai 插件入口)
│   ├── eval_benchmarks/   # 本地 benchmark (inspect_ai 插件)
│   │   ├── _registry.py       # 集中注册 @task
│   │   ├── raccoon/           # Prompt 提取攻击
│   │   ├── overthink/         # 推理模型减速攻击
│   │   ├── privacylens/       # 隐私规范评测
│   │   └── personalized_safety/  # 高风险个性化场景安全
│   ├── indexes/           # 样本索引文件
│   │   └── <benchmark>/
│   │       └── <task>.yaml
│   └── tools/             # 辅助工具
│       ├── list_samples.py    # 枚举样本 ID
│       └── update_index.py    # 从评测结果更新索引
├── upstream/              # 上游依赖 (Git 子模块)
│   ├── inspect_ai/        # Inspect AI 框架
│   └── inspect_evals/     # Inspect 评估集
├── results/               # 测试结果 (运行时生成)
│   └── <model>/
│       └── <benchmark>/
│           └── logs/      # inspect_ai 日志
└── .venvs/                # 虚拟环境 (运行时生成)
    └── <benchmark>/       # 每个 benchmark 独立环境
```

## 当前支持的 Benchmarks

| Benchmark | Tasks | 说明 |
|-----------|-------|------|
| strong_reject | 1 | 拒绝有害请求能力 |
| xstest | 1 | 安全边界测试 |
| cyberseceval_2 | 3 | 代码安全 (interpreter_abuse, prompt_injection, vulnerability_exploit) |
| bbq | 1 | 偏见检测 |
| cve_bench | 1 | CVE 漏洞利用 (需要 Docker) |
| agentdojo | 1 | Agent 安全 |
| agentharm | 2 | Agent 有害行为 (agentharm, agentharm_benign) |
| truthfulqa | 1 | 事实准确性 |

## 许可证

见 [LICENSES/](./LICENSES/) 目录。
