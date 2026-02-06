# inspect_evals Patches

本目录包含对 `upstream/inspect_evals` 子模块的本地修改 patch。

## Patch 列表

| Patch | 说明 |
|-------|------|
| `001-cve-bench-api.patch` | 兼容 cvebench 0.2.0+ API 变更 |
| `002-gitignore.patch` | 添加 HuggingFace 缓存目录忽略规则 |

## 001-cve-bench-api.patch

**文件**: `src/inspect_evals/cve_bench/cve_bench.py`

**变更内容**:
- 添加 cvebench 0.2.0+ 所需的额外环境变量 (`CVEBENCH_DOCKER_DIR`, `CVEBENCH_SANDBOXES_DIR` 等)
- 修改 `cvebench()` 调用方式，传递 `challenges_dir` 参数
- 移除不再支持的 `sandbox_type` 参数

**背景**: cvebench 0.2.0 重构了 API，上游 inspect_evals 尚未更新。此 patch 提供兼容性修复。

## 002-gitignore.patch

**文件**: `.gitignore`

**变更内容**:
- 添加 `**/hf-cache/` 忽略规则

**背景**: agentharm benchmark 会在本地生成 HuggingFace 缓存目录，不应提交到版本控制。

## 使用方法

### 应用 Patches

```bash
# 从项目根目录运行
./scripts/apply-patches.sh
```

### 更新 Patch

当需要修改 patch 内容时：

1. 进入子模块目录
   ```bash
   cd upstream/inspect_evals
   ```

2. 恢复到干净状态
   ```bash
   git checkout .
   ```

3. 手动进行修改

4. 生成新的 patch
   ```bash
   git diff HEAD -- <file> > ../../patches/inspect_evals/<patch-name>.patch
   ```

5. 提交更新后的 patch 到主仓库

### 何时需要更新 Patch

- 上游 `inspect_evals` 更新了相关文件，导致 patch 无法应用
- cvebench 包的 API 再次变更
- 发现新的兼容性问题需要修复

## 维护注意事项

1. **Patch 顺序**: 文件名前缀 (001-, 002-) 决定应用顺序
2. **上游同步**: 更新子模块后需重新应用 patches
3. **冲突处理**: 如果 patch 失败，需手动检查上游变更并更新 patch
