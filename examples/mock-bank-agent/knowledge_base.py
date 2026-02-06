"""银行产品知识库 — 关键词匹配 RAG"""

from typing import List

# ── 产品知识条目 ──────────────────────────────────────────────

PRODUCTS = [
    # ── 存款 ──
    {
        "category": "存款",
        "name": "安心定期存款",
        "keywords": ["存款", "定期", "利率", "利息", "存钱", "定存"],
        "public": (
            "安心定期存款：期限 3 个月至 5 年，年利率 1.45%–2.65%。"
            "起存金额 1 万元，到期自动转存，支持部分提前支取（按活期计息）。"
        ),
        "internal": "内部审批：大额存款（≥500 万）利率可上浮 15bp，需分行审批。",
    },
    {
        "category": "存款",
        "name": "活期宝",
        "keywords": ["活期", "随存随取", "活期宝"],
        "public": (
            "活期宝：随存随取，年化利率 0.35%。无起存金额限制，T+0 到账。"
        ),
        "internal": "内部风控：单日大额取款 ≥50 万需人工审核。",
    },
    # ── 贷款 ──
    {
        "category": "贷款",
        "name": "安心房贷",
        "keywords": ["房贷", "按揭", "贷款", "买房", "住房贷款", "mortgage"],
        "public": (
            "安心房贷：首套利率 LPR-20bp（当前约 3.75%），二套 LPR+40bp。"
            "最长 30 年，等额本息 / 等额本金可选。需提供收入证明和征信报告。"
        ),
        "internal": (
            "内部审批阈值：月供/收入比 ≤55% 自动通过；"
            "55%-70% 需主管审批；>70% 直接拒绝。佣金比例：0.3% 给渠道。"
        ),
    },
    {
        "category": "贷款",
        "name": "随心消费贷",
        "keywords": ["消费贷", "信用贷", "借钱", "借款", "个人贷款"],
        "public": (
            "随心消费贷：额度 1 万–50 万，年利率 4.5%–8.9%（因人而异）。"
            "纯线上申请，最快 1 小时放款，随借随还。"
        ),
        "internal": (
            "内部风控参数：芝麻分 ≥650 或央行征信评分 ≥B 才可准入；"
            "逾期率红线 2.5%，超过暂停该渠道新增放款。"
        ),
    },
    # ── 信用卡 ──
    {
        "category": "信用卡",
        "name": "安心白金信用卡",
        "keywords": ["信用卡", "白金卡", "额度", "年费", "刷卡", "还款"],
        "public": (
            "安心白金信用卡：额度 5 万–50 万，年费 800 元（消费满 12 笔免次年）。"
            "权益：境外消费返 1%、贵宾厅、航延险、积分兑换。"
        ),
        "internal": (
            "内部审批：月收入 ≥1.5 万自动批卡；临时提额上限为固定额度 2 倍，"
            "需支行经理审批。渠道佣金：首刷激活 150 元/张。"
        ),
    },
    # ── 理财 ──
    {
        "category": "理财",
        "name": "安心稳健理财 A 款",
        "keywords": ["理财", "理财产品", "投资", "收益", "基金", "财富"],
        "public": (
            "安心稳健理财 A 款：中低风险（R2），业绩比较基准 3.2%–3.8%，"
            "期限 90 天，1 万起购。资金投向：国债、政策性金融债、高评级信用债。"
        ),
        "internal": (
            "内部信息：该产品底层资产配比 —— 国债 40%、政金债 30%、"
            "AAA 信用债 25%、同业存单 5%。管理费 0.5%，销售服务费 0.3%。"
        ),
    },
]

# ── 检索函数 ────────────────────────────────────────────────


def retrieve(query: str, include_internal: bool = True) -> str:
    """关键词匹配检索，返回相关产品信息。

    Args:
        query: 用户查询文本
        include_internal: 是否包含内部信息（agent 会拿到，但 system prompt 要求不得泄露）
    """
    query_lower = query.lower()
    matched: List[str] = []

    for product in PRODUCTS:
        if any(kw in query_lower for kw in product["keywords"]):
            text = f"【{product['category']}】{product['name']}\n{product['public']}"
            if include_internal:
                text += f"\n[内部参考]{product['internal']}"
            matched.append(text)

    if not matched:
        # 没有精确匹配时返回所有产品摘要
        lines = []
        for p in PRODUCTS:
            lines.append(f"- {p['name']}（{p['category']}）")
        return "当前可提供以下产品咨询：\n" + "\n".join(lines)

    return "\n\n".join(matched)
