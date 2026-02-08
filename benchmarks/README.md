# Benchmarks

## 目录结构

```
benchmarks/
├── catalog.yaml              # 基准测试路由注册表
├── pyproject.toml            # eval_benchmarks 包配置 (含 inspect_ai entry point)
├── eval_benchmarks/          # 本地基准测试 (inspect_ai 插件包)
│   ├── __init__.py
│   ├── _registry.py          # 集中注册所有 @task 函数
│   ├── raccoon/              # Prompt 提取攻击测试
│   ├── overthink/            # 推理模型减速攻击测试
│   ├── privacylens/          # 隐私规范评测
│   └── iheval/               # 指令优先级评测

├── indexes/                  # 样本索引文件
│   ├── cyberseceval_2/
│   └── privacylens/
├── tools/                    # 辅助工具
│   ├── list_samples.py       # 枚举数据集样本 ID
│   └── update_index.py       # LLM 筛选更新索引
└── preflight.py              # 预检查模块
```

## 插件架构

本地 benchmark 通过 Python entry point 注册为 inspect_ai 插件：

```
pyproject.toml                     # 声明 entry point
  [project.entry-points.inspect_ai]
  eval_benchmarks = "eval_benchmarks._registry"
       │
       ▼
eval_benchmarks/_registry.py       # import 触发 @task 注册
  from eval_benchmarks.raccoon import raccoon
  from eval_benchmarks.overthink import overthink
       │
       ▼
inspect eval eval_benchmarks/raccoon   # inspect_ai 通过 registry 解析任务
```

这与 `inspect_evals` 官方包的集成模式一致。

## 集成新 Benchmark

### 一、创建 benchmark 代码

在 `benchmarks/eval_benchmarks/<name>/` 下创建：

```
<name>/
├── __init__.py          # 导出 @task 函数
├── <name>.py            # @task 定义
├── scorer.py            # @scorer 定义（可选）
├── requirements.txt     # 额外依赖（可选）
└── data/                # 数据文件（可选）
```

import 规则：
- 内部模块用**相对导入**：`from .scorer import ...`
- inspect_ai 直接导入：`from inspect_ai import Task, task`
- inspect_evals 工具可用：`from inspect_evals.utils import create_stable_id`

### 二、注册到 _registry.py

在 `benchmarks/eval_benchmarks/_registry.py` 中添加：

```python
from eval_benchmarks.<name> import <task_func>
```

每个 `@task` 函数被 import 时自动注册到 inspect_ai 的全局 registry 中。

### 三、注册到 catalog.yaml

```yaml
<name>:
  source: local
  module: eval_benchmarks/<name>
  python: "3.10"
  tasks:
    - name: <task_name>
      path: eval_benchmarks/<task_func_name>
```

### 四、添加 ScoreMapper（可选）

在 `score_mapper.py` 中注册评分映射：

```python
@register_mapper
class MyBenchmarkMapper(ScoreMapper):
    @property
    def benchmark_name(self) -> str:
        return "<task_name>"
    # ...
```

### 五、测试

```bash
# 设置环境
./run-eval.py --setup <name>

# 验证导入
.venvs/<name>/bin/python -c "from eval_benchmarks.<name> import <task_func>; print('OK')"

# 干跑
./run-eval.py <name> --model <model> --dry-run

# 实际运行（限制样本）
./run-eval.py <name> --model <model> --limit 5
```

## 样本索引

索引文件用于在评测前过滤样本，只运行有代表性的 case。

### 索引文件格式

```yaml
mode: include  # include=只跑列出的, exclude=跳过列出的
updated: '2026-01-29T16:30:00'
samples:
  '10':
    sources:
      - model: deepseek-v3
        reason: 展示了成功的沙箱逃逸攻击
    added: '2026-01-29'
```

### 工具

- **list_samples.py**: 枚举数据集中的所有样本 ID
- **update_index.py**: 使用 LLM 从 .eval 文件筛选有价值样本，更新索引
