# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from pathlib import Path
from tqdm.contrib.concurrent import thread_map
import os

from utils import lm


def run_exp(
    task: str,
    model_path: str,
    all_prompts,
    generations_file_path=None,
    base_path="output",
    inference_method="vllm",
    max_workers=64,
    max_tokens=512,
    return_gen=False
):
    """
    One-click stable version:
    - Keep interface unchanged
    - For inference_method == "vllm": route to lm.call_openai_compat() with capped workers
      so 503 / connection hiccups won't crash the whole run.
    """

    if not generations_file_path:
        base_path = Path(base_path)
        model_name = model_path.split("/")[-1]
        output_folder = base_path / task / model_name
        output_folder.mkdir(exist_ok=True, parents=True)
        generations_file_path = output_folder / "generation.jsonl"

    generations_file_path = str(generations_file_path)
    print("generations_file_path", generations_file_path)

    prompts = all_prompts.prompt.to_list()

    # get the response from the model
    if inference_method == "openai":
        # 统一走 openai_compat：避免 OPENAI_KEY 依赖，也避免真的请求 gpt-4o 等模型名
        user_cap = int(os.getenv("HALLULENS_MAX_WORKERS", "1"))
        safe_workers = max(1, min(int(max_workers), user_cap))

        all_prompts["generation"] = thread_map(
            lambda p: lm.call_openai_compat(
                p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens
            ),
            prompts,
            max_workers=safe_workers,
            desc=f"Predict on openai_compat (workers={safe_workers})",
        )
    elif inference_method == "vllm":
        # ===== 关键修改点 =====
        # 1) 不再直接 call_vllm_api（它遇到 503 会直接抛异常导致任务中断）
        # 2) 改走 call_openai_compat（你已经在 lm.py 里做了环境变量 OPENAI_BASE_URL/KEY/MODEL 的兼容）
        # 3) 默认把并发压到 1（免费/不稳定线路下更稳）；你可用环境变量手动放大
        #
        # 你可以在 cmd 里这样调：
        #   set HALLULENS_MAX_WORKERS=4
        user_cap = int(os.getenv("HALLULENS_MAX_WORKERS", "1"))
        safe_workers = max(1, min(int(max_workers), user_cap))

        all_prompts["generation"] = thread_map(
            lambda p: lm.call_openai_compat(
                p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens
            ),
            prompts,
            max_workers=safe_workers,
            desc=f"Predict on openai_compat (workers={safe_workers})",
        )

    elif inference_method == "custom":
        all_prompts["generation"] = thread_map(
            lambda p: lm.generate(
                p, model=model_path, temperature=0.0, top_p=1.0, max_tokens=max_tokens
            ),
            prompts,
            max_workers=max_workers,
            desc="Predict on custom API",
        )
    else:
        raise NotImplementedError(f"No method {inference_method}")

    # save the results
    all_prompts.to_json(generations_file_path, lines=True, orient="records")

    if return_gen:
        return all_prompts