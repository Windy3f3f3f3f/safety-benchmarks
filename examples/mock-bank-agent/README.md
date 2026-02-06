# Mock Bank Agent

模拟银行客服智能体，用于演示评测平台对自定义 Agent 的接入能力。

## 架构

```
用户/评测请求 → mock-bank-agent (port 9000)
                    │
                    ├── 1. 关键词匹配 RAG：从知识库检索相关产品信息
                    ├── 2. 注入 system prompt + RAG 上下文
                    └── 3. 转发给后端 LLM，返回响应
```

Agent 暴露 OpenAI 兼容 API（`/v1/chat/completions`），inspect_ai 可直接调用。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置后端 LLM
export BACKING_MODEL_URL=https://api.openai.com/v1
export BACKING_MODEL_NAME=gpt-4o-mini
export BACKING_API_KEY=sk-xxx

# 3. 启动
python server.py --port 9000
```

## 测试

```bash
# 直接调用
curl http://localhost:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mock-bank-agent","messages":[{"role":"user","content":"有什么存款产品？"}]}'

# 通过 inspect 评测
export OPENAI_BASE_URL=http://localhost:9000/v1
export OPENAI_API_KEY=test
./run-eval.py raccoon --model openai/mock-bank-agent --limit 3
```

## 知识库

包含 4 类银行产品：存款、贷款、信用卡、理财。每类产品有：
- **公开信息**：利率、期限、申请条件等（可告知客户）
- **内部信息**：审批阈值、风控参数、佣金比例（system prompt 要求不可泄露）

这些内部信息会通过 RAG 注入上下文，但 system prompt 明确禁止泄露——这正是安全评测要检验的。

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `BACKING_MODEL_URL` | 后端 LLM base URL | `https://api.openai.com/v1` |
| `BACKING_MODEL_NAME` | 后端 LLM 模型名 | `gpt-4o-mini` |
| `BACKING_API_KEY` | 后端 LLM API Key | （必填） |
