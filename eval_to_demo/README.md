# CyberSecEval2 Eval → POC-Demo 转换流程

## 前置条件

- Inspect AI 评测已完成（`.eval` 日志文件，或已 dump 的 JSON）
- `uv` 已安装（用于运行 `inspect log dump`）
- poc-demo 后端运行中（`localhost:8000`）
- Python 3.8+

## 端到端流程

从 `.eval` 日志文件一步到位：

```bash
# .eval → JSON dump → poc-demo dataset，一条命令完成
python convert.py path/to/logs/xxx.eval -o output/result.json

# 保留中间 JSON dump 到 evaldumps/ 供复用
python convert.py path/to/logs/xxx.eval -o output/result.json --keep-dump
```

脚本自动检测 `.eval` 后缀，调用 `uv run inspect log dump --resolve-attachments full` 导出 JSON，然后转换。

## 分步执行

如果已有 JSON dump 文件，可以跳过 dump 步骤：

```bash
# 手动 dump（可选，convert.py 会自动做）
uv run inspect log dump --resolve-attachments full path/to/logs/xxx.eval > evaldumps/xxx.json

# 从 JSON 转换
python convert.py evaldumps/xxx.json -o output/result.json
```

## 其他选项

```bash
# 限制样本数（调试用）
python convert.py evaldumps/doubao-cyse2-1.json --limit 5 -o output/test-5.json

# 仅验证不输出
python convert.py evaldumps/doubao-cyse2-1.json --dry-run

# LLM 增强攻击描述（需要 ../.env 中配置 OPENAI_API_KEY）
python convert.py evaldumps/doubao-cyse2-1.json --enrich -o output/enriched.json
```

## 支持的任务类型

| eval 文件 | task | Capability | 样本数 |
|---|---|---|---|
| doubao-cyse2-1.json | cyse2_vulnerability_exploit | F3 | 124 |
| doubao-cyse2-2.json | cyse2_prompt_injection | F1 | 251 |
| doubao-cyse2-3.json | cyse2_interpreter_abuse | F3 | 500 |

## 导入 poc-demo

转换后的 JSON 直接拷贝到 poc-demo 数据目录：

```bash
# 数据目录位置（由 POC_DATA_ROOT 环境变量决定，默认 poc-demo 同级的 poc-data/）
DATA_DIR=../../poc-data/datasets

# 文件名必须与 meta.datasetId 一致
cp output/result.json "$DATA_DIR/cyse2-vulnerability_exploit-doubao-seed-1-8-251228.json"
```

后端自动识别新文件，无需重启。通过 API 验证：

```bash
curl -s http://localhost:8000/datasets | python3 -m json.tool
```

## 批量转换

```bash
mkdir -p output
for f in evaldumps/doubao-cyse2-*.json; do
  python convert.py "$f" -o "output/$(basename "$f")"
done
```

## 输出格式

v2.2.0 Dataset JSON。相比 v2.1.0 新增字段（需同事完成 schema 升级后前端才会渲染）：

- `case.benchmarkMeta` — eval 溯源信息
- `criteria.referenceAnswer / referenceCode` — 参考答案（仅 vulnerability_exploit）
- `recording.tokenUsage` — token 用量统计
- `recording.evaluation` — 评分详情（rawScore, riskLevel, isSuccessfulAttack 等）

当前前端已验证可正常回放 recording（F1 类型已手动测试通过）。
