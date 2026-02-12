# Bug 修复：Step 编号从 2 开始的问题

> **修复时间**: 2026-02-10
> **影响文件**: `react_agent_attack.py`, `react_agent.py`
> **Bug 类型**: 逻辑错误 - 变量复用

---

## 问题描述

**原始行为**: Agent 执行 workflow 时，步骤编号从 "At step 2" 开始，而不是 "At step 1"

**用户报告的日志**:
```
[example/financial_analyst_agent] At step 2, {'role': 'assistant', 'content': '[Action]: ...'}
```

---

## 根本原因

`self.rounds` 变量被用于两个不同的目的：

1. **Workflow 生成阶段**: 记录生成 workflow 的尝试次数
   - 在 `automatic_workflow()` 函数中递增
   - 代码位置: [base_agent.py:116](../ASB_SOURCE_DATA/pyopenagi/agents/base_agent.py#L116)

2. **Workflow 执行阶段**: 记录当前执行的步骤编号
   - 在执行循环中用于显示 "At step {self.rounds + 1}"
   - 代码位置: [react_agent_attack.py:276](../ASB_SOURCE_DATA/pyopenagi/agents/react_agent_attack.py#L276)

**执行流程**:
```
初始化: self.rounds = 0
↓
调用 automatic_workflow()
  └─ self.rounds += 1  # rounds 变为 1
↓
开始执行 workflow
  ├─ 第 1 个步骤: self.rounds = 1 → 显示 "At step 2" ❌
  ├─ 第 2 个步骤: self.rounds = 2 → 显示 "At step 3" ❌
  └─ ...
```

---

## 修复方案

在 workflow 生成完成后、执行开始前，重置 `self.rounds = 0`

### 修复文件

#### 1. react_agent_attack.py (第 269-274 行)

**修复前**:
```python
if workflow:
    workflow_failure = False

    for i, step in enumerate(workflow):
```

**修复后**:
```python
if workflow:
    # Reset rounds counter for execution (it was used during workflow generation attempts)
    self.rounds = 0
    workflow_failure = False

    for i, step in enumerate(workflow):
```

#### 2. react_agent.py (第 149-154 行)

**修复前**:
```python
if workflow:
    final_result = ""

    for i, step in enumerate(workflow):
```

**修复后**:
```python
if workflow:
    # Reset rounds counter for execution (it was used during workflow generation attempts)
    self.rounds = 0
    final_result = ""

    for i, step in enumerate(workflow):
```

---

## 验证修复

**预期行为**:
```
初始化: self.rounds = 0
↓
调用 automatic_workflow()
  └─ self.rounds += 1  # rounds 变为 1
↓
重置: self.rounds = 0  # ✅ 新增
↓
开始执行 workflow
  ├─ 第 1 个步骤: self.rounds = 0 → 显示 "At step 1" ✅
  ├─ 第 2 个步骤: self.rounds = 1 → 显示 "At step 2" ✅
  └─ ...
```

---

## 影响范围

### 正面影响
- ✅ Step 编号从 1 开始，符合直觉
- ✅ 与标准 ReAct 循环一致（迁移到 inspect_ai 时对齐）
- ✅ 日志输出更清晰

### 潜在风险
- ⚠️ 如果有代码依赖 `self.rounds` 在执行前的值，可能会受影响
- ⚠️ 需要重新运行测试，验证评分不受影响

---

## 测试建议

运行以下命令验证修复：

```bash
cd E:\code\aisafety\ASB1\ASB_SOURCE_DATA
python main_attacker.py \
  --llm_name deepseek-v3.2 \
  --attacker_tools_path data/attack_tools_test.jsonl \
  --tasks_path data/agent_task_test.jsonl \
  --tools_info_path data/all_normal_tools.jsonl \
  --task_num 1 \
  --direct_prompt_injection \
  --attack_type naive \
  --res_file logs/test_fix.csv
```

**检查日志输出**: 应该看到 "At step 1" 而不是 "At step 2"

---

## 相关代码位置

- [base_agent.py:96-119](../ASB_SOURCE_DATA/pyopenagi/agents/base_agent.py#L96-L119) - `automatic_workflow()` 函数
- [react_agent_attack.py:251-272](../ASB_SOURCE_DATA/pyopenagi/agents/react_agent_attack.py#L251-L272) - workflow 执行入口
- [react_agent.py:135-152](../ASB_SOURCE_DATA/pyopenagi/agents/react_agent.py#L135-L152) - workflow 执行入口（非攻击版本）

---

## 更新日志

**2026-02-10** - 初始修复
- 在 `react_agent_attack.py` 中添加 `self.rounds = 0` 重置
- 在 `react_agent.py` 中添加 `self.rounds = 0` 重置
- 创建本修复文档

---

## 后续工作

- [ ] 运行完整测试套件，验证修复不影响评分
- [ ] 对比修复前后的 CSV 结果，确认指标一致
- [ ] 更新 CLAUDE.md 文档，记录此修复
- [ ] 考虑重构，使用独立变量（如 `workflow_generation_attempts`）代替复用 `self.rounds`
