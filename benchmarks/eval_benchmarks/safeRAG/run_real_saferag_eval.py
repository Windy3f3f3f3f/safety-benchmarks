#!/usr/bin/env python3
"""
使用原始 SafeRAG 组件的完整评测脚本 (支持命令行参数)

使用方法:
    # 使用 doubao-seed-1-8
    python3 run_real_saferag_eval.py --model doubao-seed-1-8 --samples 10

    # 使用 deepseek-v3.2
    python3 run_real_saferag_eval.py --model deepseek-v3.2 --samples 10

    # 使用 gpt-4o-mini
    python3 run_real_saferag_eval.py --model gpt-4o-mini --samples 5
"""

import sys
import os
import argparse
from pathlib import Path
import json

# 添加当前目录到路径（用于导入本地模块）
CURRENT_DIR = Path(__file__).parent
sys.path.insert(0, str(CURRENT_DIR))

# 默认环境变量
DEFAULT_API_KEY = "sk-AVsyIUmeAjeyEyhBA981E921C5304b079540091115430e97"
DEFAULT_API_BASE = "https://aihubmix.com/v1"

# 模型配置
MODEL_CONFIGS = {
    "doubao-seed-1-8": {
        "model_name": "doubao-seed-1-8",
        "api_base": DEFAULT_API_BASE,
    },
    "deepseek-v3.2": {
        "model_name": "deepseek-v3.2",
        "api_base": DEFAULT_API_BASE,
    },
    "gpt-4o-mini": {
        "model_name": "gpt-4o-mini",
        "api_base": DEFAULT_API_BASE,
    },
}


def run_real_saferag_evaluation(model_name="doubao-seed-1-8", num_samples=5):
    """运行完整的 SafeRAG 评测"""

    # 设置环境变量
    os.environ["OPENAI_API_KEY"] = DEFAULT_API_KEY
    os.environ["OPENAI_BASE_URL"] = DEFAULT_API_BASE

    print("=" * 80)
    print(f"SafeRAG 完整评测 (使用原始组件)")
    print(f"模型: {model_name}")
    print(f"样本数: {num_samples}")
    print("=" * 80)

    # 导入本地 SafeRAG 组件
    from retrievers.base import BaseRetriever
    from tasks.nctd_attack import Silver_noise
    from llms.api_model import OpenAICompat
    from embeddings.base import HuggingfaceEmbeddings

    # 数据路径
    data_dir = CURRENT_DIR / "data"

    print("\\n步骤 1: 创建嵌入模型...")
    embed_model_path = data_dir / "bge-base-zh-v1.5"
    embed_model = HuggingfaceEmbeddings(
        model_name=str(embed_model_path),
        model_kwargs={'device': 'cpu'}
    )
    print("✅ 嵌入模型创建成功")

    print("\\n步骤 2: 创建 BaseRetriever...")
    attack_data_path = data_dir / "nctd_datasets" / "nctd.json"
    docs_path = data_dir / "knowledge_base"
    retriever = BaseRetriever(
        attack_data_directory=str(attack_data_path),
        docs_directory=str(docs_path),
        attack_task="SN",
        attack_module="indexing",
        attack_intensity=0.5,
        embed_model=embed_model,
        embed_dim=768,
        filter_module='base',
        chunk_size=128,
        chunk_overlap=0,
        collection_name="docs",
        similarity_top_k=6,
    )
    print("✅ BaseRetriever 创建成功")

    print("\\n步骤 3: 创建攻击任务...")
    attack_task = Silver_noise(
        output_dir='./output',
        quest_eval_model="gpt-3.5-turbo",
        attack_task="SN",
        use_quest_eval=False,
    )
    print("✅ 攻击任务创建成功")

    print(f"\\n步骤 4: 创建生成模型 ({model_name})...")
    model_config = MODEL_CONFIGS.get(model_name, MODEL_CONFIGS["gpt-4o-mini"])
    model = OpenAICompat(model_name=model_config["model_name"])
    attack_task.set_model(model, retriever)
    print("✅ 生成模型设置成功")

    print("\\n步骤 5: 加载评测数据...")
    dataset_path = data_dir / "nctd_datasets" / "nctd.json"
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    sn_samples = dataset["SN"][:num_samples]
    print(f"✅ 加载了 {len(sn_samples)} 个样本")

    print("\\n步骤 6: 开始评测...")
    results = []

    for i, sample_data in enumerate(sn_samples):
        print(f"\\n--- 评测样本 {i+1}/{len(sn_samples)} ---")

        # 构造数据点
        data_point = {
            "id": i,
            "questions": sample_data["questions"],
            "propositions": sample_data.get("propositions", []),
            "numbered_options": sample_data.get("numbered_options", []),
            "ground_truth_correct_options": sample_data.get("ground_truth_correct_options", []),
            "ground_truth_incorrect_options": sample_data.get("ground_truth_incorrect_options", []),
            "enhanced_contexts": sample_data.get("propositions", []),
            "enhanced_SN_contexts": sample_data.get("enhanced_SN_contexts", []),
            "attack_kws": sample_data.get("attack_kws", []),
        }

        print(f"问题: {data_point['questions'][:50]}...")

        try:
            # 执行检索
            print("执行检索...")
            retrieve_context, filtered_retrieve_context = attack_task.retrieve_docs(data_point)

            print(f"检索到 {len(retrieve_context)} 个文档片段")

            # 准备用于评分的数据点
            data_point["retrieve_context"] = str(retrieve_context)
            data_point["filtered_retrieve_context"] = filtered_retrieve_context

            # 生成回答
            print("生成回答...")
            generated_text = attack_task.model_generation(data_point)
            data_point["generated_text"] = generated_text

            print(f"生成的回答: {generated_text[:100]}...")

            # 评分
            print("计算评分...")
            score_result = attack_task.scoring(data_point)

            print("✅ 评分结果:")
            for key, value in score_result["metrics"].items():
                print(f"  {key}: {value}")

            results.append({
                "sample_id": i,
                "question": data_point["questions"],
                "generated_text": generated_text,
                "metrics": score_result["metrics"],
                "log": score_result.get("log", {}),
            })

        except Exception as e:
            print(f"❌ 样本 {i} 评测失败: {e}")
            import traceback
            traceback.print_exc()

    print("\\n" + "=" * 80)
    print("评测完成!")
    print("=" * 80)

    # 计算总体统计
    if results:
        print("\\n【总体统计】")
        overall = {
            'retrieval_accuracy': 0,
            'recall_gc': 0,
            'recall_ac': 0,
            'attack_success_rate': 0,
            'f1_correct': 0,
            'f1_incorrect': 0,
            'f1_avg': 0,
        }

        valid_count = 0
        for result in results:
            metrics = result["metrics"]
            for key in overall.keys():
                if metrics.get(key, -1) >= 0:
                    overall[key] += metrics[key]
            if metrics.get('f1_correct', -1) >= 0:
                valid_count += 1

        n = len(results)
        print(f"\\n检索准确率: {overall['retrieval_accuracy'] / n:.4f}")
        print(f"正确上下文召回率 (recall_gc): {overall['recall_gc'] / n:.4f}")
        print(f"攻击上下文召回率 (recall_ac): {overall['recall_ac'] / n:.4f}")
        print(f"攻击成功率: {overall['attack_success_rate'] / n:.4f}")

        if valid_count > 0:
            print(f"F1 正确: {overall['f1_correct'] / valid_count:.4f}")
            print(f"F1 错误: {overall['f1_incorrect'] / valid_count:.4f}")
            print(f"F1 平均: {overall['f1_avg'] / valid_count:.4f}")
        else:
            print("F1 分数: 未计算 (需要 QuestEval)")

    # 保存结果
    script_dir = Path(__file__).parent
    output_file = script_dir / f"real_saferag_results_{model_name.replace('-', '_')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            "model": model_name,
            "num_samples": num_samples,
            "overall_stats": {
                "retrieval_accuracy": overall['retrieval_accuracy'] / n if n > 0 else 0,
                "recall_gc": overall['recall_gc'] / n if n > 0 else 0,
                "recall_ac": overall['recall_ac'] / n if n > 0 else 0,
                "attack_success_rate": overall['attack_success_rate'] / n if n > 0 else 0,
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)

    print(f"\\n结果已保存到: {output_file}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SafeRAG 完整评测')
    parser.add_argument('--model', type=str, default='doubao-seed-1-8',
                        choices=['doubao-seed-1-8', 'deepseek-v3.2', 'gpt-4o-mini'],
                        help='选择模型')
    parser.add_argument('--samples', type=int, default=5,
                        help='评测样本数量')

    args = parser.parse_args()

    run_real_saferag_evaluation(model_name=args.model, num_samples=args.samples)
