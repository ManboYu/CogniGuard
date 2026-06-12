from __future__ import annotations

from datetime import datetime
from statistics import mean
from typing import Any, Optional

from core.schemas import (
    COGNITIVE_DOMAINS,
    DISCLAIMER,
    display_risk_level,
    normalize_domain_scores,
)


COGNIGUARD_SCORE_EXPLANATION = (
    "基于可用认知域得分和画钟结构化特征的技术原型提示分，"
    "不等同于 MoCA、MMSE 或 Mini-Cog 正式量表。"
)
DIALOGUE_SCORE_EXPLANATION = (
    "对话评估参考分仅基于本次对话可用认知域得分计算，"
    "是技术原型指标，不等同于 MoCA、MMSE 或 Mini-Cog 正式量表。"
)


def generate_mock_dialog_report(messages: list[str]) -> dict[str, Any]:
    message_count = len([message for message in messages if message.strip()])
    confidence_bonus = min(message_count, 3) * 0.03

    raw_scores = {
        "orientation": 0.78 + confidence_bonus,
        "memory": 0.72 + confidence_bonus,
        "language": 0.82 + confidence_bonus,
        "executive_function": 0.74 + confidence_bonus,
        "attention": 0.76 + confidence_bonus,
        "visuospatial": 0.70 + confidence_bonus,
    }
    concern_profile = _dialogue_concern_profile(messages)
    for domain in concern_profile["domains"]:
        if domain in raw_scores:
            raw_scores[domain] = min(raw_scores[domain], 0.56)
    if concern_profile["vague_count"] >= 2:
        raw_scores["executive_function"] = min(raw_scores["executive_function"], 0.62)
        raw_scores["visuospatial"] = min(raw_scores["visuospatial"], 0.60)
    if concern_profile["vague_count"] >= 4:
        raw_scores["memory"] = min(raw_scores["memory"], 0.58)
        raw_scores["attention"] = min(raw_scores["attention"], 0.58)

    scores = normalize_domain_scores(raw_scores)

    evidence = [
        {
            "domain": "orientation",
            "source": "dialog",
            "text": "回答中能提到日期、日程或当天活动线索。",
        },
        {
            "domain": "memory",
            "source": "dialog",
            "text": "能复述前面提到的饮食或活动信息，但本阶段仅为 mock 判断。",
        },
        {
            "domain": "language",
            "source": "dialog",
            "text": "回答句子基本完整，能表达日常意图。",
        },
    ]

    if concern_profile["vague_count"] >= 2:
        risk_level = "medium"
        evidence.append(
            {
                "domain": "executive_function",
                "source": "dialog",
                "text": "连续多轮回答出现不确定、模糊或跑题表达，建议补充画钟观察执行步骤和空间布局。",
            }
        )
        evidence.append(
            {
                "domain": "visuospatial",
                "source": "dialog",
                "text": "对方向、路线或任务步骤的回答不够稳定，本次 mock 规则建议继续画钟测试。",
            }
        )
        explanation = (
            "本次 mock 对话出现连续模糊或不确定回答，演示规则建议补充画钟测试，"
            "用于观察视觉空间和执行功能线索。"
        )
    else:
        risk_level = "low" if message_count >= 2 else "unknown"
        explanation = (
            "本次 mock 对话显示回答内容较连贯，暂未呈现明显连续下降信号。"
            if message_count >= 2
            else "输入轮次较少，本阶段仅生成演示用 mock 结果。"
        )

    return {
        "session_id": "mock-dialog-session",
        "participant_id": "demo-person-live",
        "is_mock": True,
        "domain_scores": scores,
        "evidence": evidence,
        "risk_level": risk_level,
        "explanation": explanation,
        "disclaimer": DISCLAIMER,
    }


def _dialogue_concern_profile(messages: list[str]) -> dict[str, Any]:
    domains: list[str] = []
    vague_count = 0
    current_domain = ""
    for message in messages:
        text = str(message or "").strip()
        if not text:
            continue
        if text.startswith("AI访谈问题："):
            current_domain = _infer_dialog_domain_from_text(text)
            continue
        if text.startswith("老人回答："):
            answer = text.split("老人回答：", 1)[1].strip()
        else:
            answer = text
        if not _looks_uncertain_or_vague(answer):
            continue
        vague_count += 1
        domain = current_domain or _infer_dialog_domain_from_text(answer)
        if domain in COGNITIVE_DOMAINS and domain not in domains:
            domains.append(domain)
    return {"vague_count": vague_count, "domains": domains}


def _looks_uncertain_or_vague(text: str) -> bool:
    vague_keywords = (
        "不太清楚",
        "不清楚",
        "不知道",
        "说不好",
        "说不上",
        "想不起来",
        "拿不准",
        "不确定",
        "记不住",
        "弄混",
        "混了",
        "分不清",
        "不太想算",
        "不会",
        "差不多",
        "看情况",
    )
    return any(keyword in text for keyword in vague_keywords)


def _infer_dialog_domain_from_text(text: str) -> str:
    keyword_domains = [
        ("orientation", ("星期", "日期", "今天", "时间", "几号")),
        ("memory", ("刚才", "记得", "记忆", "复述", "早饭", "早餐")),
        ("language", ("描述", "一句话", "看到", "东西", "房间")),
        ("executive_function", ("准备", "出门", "计划", "步骤", "安排", "先")),
        ("attention", ("往回数", "倒数", "数", "计算", "注意")),
        ("visuospatial", ("左", "右", "方向", "位置", "路线", "经过", "客厅", "厨房", "门口")),
    ]
    for domain, keywords in keyword_domains:
        if any(keyword in text for keyword in keywords):
            return domain
    return "language"


def generate_mock_clock_report(filename: Optional[str] = None) -> dict[str, Any]:
    return {
        "session_id": "mock-clock-session",
        "uploaded_filename": filename or "not_saved",
        "is_mock": True,
        "clock_findings": {
            "number_placement": "数字基本完整，但间距略不均匀。",
            "hand_accuracy": "指针方向可表达目标时间，但长度区分不够清楚。",
            "visuospatial_evidence": [
                "部分数字略集中在右侧。",
                "圆形轮廓轻微不规则。",
                "长短针区分不够明显。",
            ],
        },
        "risk_level": "medium",
        "disclaimer": DISCLAIMER,
    }


def summarize_trend(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    if not sessions:
        return {
            "trend_label": "unknown",
            "summary": "没有可用于趋势判断的模拟 session。",
            "domain_changes": {},
            "disclaimer": DISCLAIMER,
        }

    sessions = sort_sessions_chronologically(sessions)
    averages = [_session_average(session) for session in sessions]
    first_average = averages[0]
    last_average = averages[-1]
    delta = round(last_average - first_average, 3)
    spread = round(max(averages) - min(averages), 3)

    domain_changes = _domain_changes(sessions[0], sessions[-1])

    decline_domains = [
        domain
        for domain, change in domain_changes.items()
        if change is not None and change <= -0.08
    ]

    if decline_domains:
        trend_label = "下降"
        summary = (
            "最近一次平均得分低于首次记录，且部分认知域出现连续下降信号。"
        )
    elif spread >= 0.07 and abs(delta) < 0.05:
        trend_label = "波动"
        summary = "多次 session 得分有起伏，但没有形成简单单调下降趋势。"
    elif delta >= 0.05:
        trend_label = "改善"
        summary = "最近一次平均得分高于首次记录，mock 数据显示有改善趋势。"
    else:
        trend_label = "稳定"
        summary = "多次 session 得分接近，mock 数据未显示明显下降趋势。"

    return {
        "trend_label": trend_label,
        "summary": summary,
        "average_scores": averages,
        "domain_changes": domain_changes,
        "disclaimer": DISCLAIMER,
    }


def compute_cogniguard_score(record_or_sessions: Any) -> dict[str, Any]:
    record = _latest_record(record_or_sessions)
    if not isinstance(record, dict):
        return _empty_cogniguard_score()

    scores = record.get("domain_scores")
    if not isinstance(scores, dict):
        return _empty_cogniguard_score()

    values = [
        float(score)
        for score in scores.values()
        if isinstance(score, (int, float)) and not isinstance(score, bool)
    ]
    if not values:
        return _empty_cogniguard_score()

    raw_score = int(round(mean(values) * 100))
    risk_level = record.get("risk_level", "unknown")
    score = _apply_risk_score_bounds(raw_score, risk_level)
    if score is None:
        return _empty_cogniguard_score(risk_level)

    explanation = COGNIGUARD_SCORE_EXPLANATION
    if (
        risk_level == "low"
        and score < 90
        and _dialogue_result_is_low_risk(record)
        and _clock_structure_score_value(record) >= 9
    ):
        score = 90
    elif risk_level == "low" and score < 85 and _clock_structure_score_value(record) >= 9:
        score = 85

    if score != raw_score:
        explanation += " 当前分数已根据风险等级约束进行提示性调整。"

    return {
        "score": score,
        "band": _score_band(score),
        "risk_label": display_risk_level(risk_level),
        "explanation": explanation,
    }


def compute_dialogue_score(record_or_result: Any) -> dict[str, Any]:
    record = _latest_record(record_or_result)
    if not isinstance(record, dict):
        return {
            "score": None,
            "band": "无法评估",
            "explanation": DIALOGUE_SCORE_EXPLANATION,
        }

    dialogue_result = record.get("dialogue_result")
    source = dialogue_result if isinstance(dialogue_result, dict) else record
    scores = source.get("domain_scores")
    if not isinstance(scores, dict):
        return {
            "score": None,
            "band": "无法评估",
            "explanation": DIALOGUE_SCORE_EXPLANATION,
        }

    values = [
        float(score)
        for score in scores.values()
        if isinstance(score, (int, float)) and not isinstance(score, bool)
    ]
    if not values:
        return {
            "score": None,
            "band": "无法评估",
            "explanation": DIALOGUE_SCORE_EXPLANATION,
        }

    score = int(round(mean(values) * 100))
    return {
        "score": score,
        "band": _score_band(score),
        "explanation": DIALOGUE_SCORE_EXPLANATION,
    }


def compute_clock_structure_score(record: dict[str, Any]) -> dict[str, Any]:
    cdt_features = record.get("cdt_features")
    if not isinstance(cdt_features, dict):
        clock_result = record.get("clock_result")
        if isinstance(clock_result, dict):
            cdt_features = clock_result.get("cdt_features")
    if not isinstance(cdt_features, dict) or not cdt_features:
        return {
            "score": None,
            "explanation": "暂无足够 CDT 结构化特征生成画钟结构分。",
        }

    score = 0.0
    score += 1.5 if cdt_features.get("numbers_complete") is True else 0
    score += 1.5 if cdt_features.get("number_order_correct") is True else 0
    score += 1.5 if cdt_features.get("number_spacing") == "normal" else 0
    score += 1.5 if cdt_features.get("number_distribution") == "balanced" else 0
    score += 1.0 if cdt_features.get("hands_present") is True else 0
    score += 2.0 if cdt_features.get("target_time_match") is True else 0
    score += 1.0 if cdt_features.get("center_anchor_clear") is True else 0

    return {
        "score": round(score, 1),
        "explanation": "画钟结构分为演示用结构化提示分，不是正式 CDT 临床量表。",
    }


def build_trend_chart_rows(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    sessions = sort_sessions_chronologically(sessions)
    total = len(sessions)
    for index, session in enumerate(sessions, start=1):
        label = "最近一次" if index == total and total > 1 else f"第{index}次"
        cogniguard_score = compute_cogniguard_score(session)
        row = {
            "display_label": label,
            "session_id": session.get("session_id", ""),
            "created_at": session.get("created_at", ""),
            "测试时间": format_session_time(session.get("created_at")),
            "测试类型": infer_session_test_type(session),
            "风险等级": display_risk_level(session.get("risk_level", "unknown")),
            "CogniGuard 综合提示分": _format_score_for_table(cogniguard_score),
            "CogniGuard 综合提示分数值": cogniguard_score.get("score"),
        }
        domain_scores = session.get("domain_scores", {})
        if isinstance(domain_scores, dict):
            for domain, score in domain_scores.items():
                row[domain] = score
        rows.append(row)
    return rows


def sort_sessions_chronologically(sessions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed_sessions = [
        (index, session)
        for index, session in enumerate(sessions)
        if isinstance(session, dict)
    ]
    indexed_sessions.sort(key=lambda item: (_session_sort_key(item[1]), item[0]))
    return [session for _index, session in indexed_sessions]


def format_session_time(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        return "未知时间"

    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return text[:16]
    return parsed.strftime("%Y-%m-%d %H:%M")


def _session_sort_key(session: dict[str, Any]) -> tuple[int, int, Any]:
    value = session.get("created_at")
    if not isinstance(value, str) or not value.strip():
        return (1, 0, "")
    text = value.strip()
    try:
        return (0, 0, datetime.fromisoformat(text).timestamp())
    except ValueError:
        return (0, 1, text)


def infer_session_test_type(record: dict[str, Any]) -> str:
    components = record.get("components")
    if isinstance(components, list):
        component_set = set(components)
        if {"dialogue", "clock"} <= component_set:
            return "综合评估"
        if "clock" in component_set:
            return "画钟测试"
        if "dialogue" in component_set:
            return "对话评估"

    has_clock = isinstance(record.get("clock_result"), dict) or isinstance(
        record.get("cdt_features"), dict
    )
    evidence = record.get("evidence", [])
    has_dialog = False
    if isinstance(evidence, list):
        has_dialog = any(
            isinstance(item, dict) and item.get("source") == "dialog"
            for item in evidence
        )
    if has_clock and has_dialog:
        return "综合评估"
    if has_clock:
        return "画钟测试"
    return "对话评估"


def _format_score_for_table(score_result: dict[str, Any]) -> str:
    score = score_result.get("score")
    if score is None:
        return "暂无"
    return f"{score} / 100"


def build_family_brief(sessions: list[dict[str, Any]]) -> dict[str, Any]:
    trend = summarize_trend(sessions)
    label = trend["trend_label"]
    latest = sort_sessions_chronologically(sessions)[-1] if sessions else {}
    latest_risk_level = latest.get("risk_level", "unknown") if isinstance(latest, dict) else "unknown"

    if latest_risk_level == "high":
        reminders = [
            "最近一次记录里出现需要重点关注的信号，建议家属留意近几天的记忆、日期判断、画钟或日常安排是否有明显变化。",
            "可以把最近几次记录和家人观察到的日常表现一起整理；如果变化持续或影响安全，建议咨询专业医生。",
            "本系统只做技术原型风险提示，不作为诊断结论。",
        ]
    elif latest_risk_level == "medium":
        reminders = [
            "最近一次记录提示有一些波动，建议先观察是否与当天睡眠、情绪、身体状态或环境变化有关。",
            "建议在相近时间段再做一次低压力记录，重点看记忆、定向和画钟表现是否持续变化。",
            "如果家属也观察到日常生活受到影响，再考虑咨询专业医生。",
        ]
    elif label == "下降":
        reminders = [
            "近期趋势有下降信号，建议家属结合日常观察，留意记忆、时间判断、画钟或安排事情的变化是否持续。",
            "可以连续保留几次记录，减少单次状态波动带来的误判。",
            "如变化持续或影响生活安全，建议咨询专业医生。",
        ]
    elif label == "波动":
        reminders = [
            "记录存在波动，建议先留意睡眠、情绪、身体状态或环境变化是否影响当天表现。",
            "可尽量在相近时间段复测，观察波动是否反复出现。",
            "如果波动伴随明显日常困难，建议咨询专业医生。",
        ]
    else:
        reminders = [
            "当前记录整体平稳，建议继续用轻松聊天和画钟记录保持观察。",
            "家属可留意日常作息、社交互动和独立完成小任务的情况。",
            "如后续出现持续变化，再带着记录咨询专业医生。",
        ]

    return {
        "trend_label": label,
        "summary": trend["summary"],
        "family_reminders": reminders,
        "disclaimer": DISCLAIMER,
    }


def _session_average(session: dict[str, Any]) -> float:
    values = [
        float(score)
        for score in session["domain_scores"].values()
        if score is not None
    ]
    return round(mean(values), 3) if values else 0.0


def _latest_record(record_or_sessions: Any) -> Optional[dict[str, Any]]:
    if isinstance(record_or_sessions, list):
        if not record_or_sessions:
            return None
        candidate = record_or_sessions[-1]
        return candidate if isinstance(candidate, dict) else None
    return record_or_sessions if isinstance(record_or_sessions, dict) else None


def _empty_cogniguard_score(risk_level: Any = "unknown") -> dict[str, Any]:
    return {
        "score": None,
        "band": "无法评估",
        "risk_label": display_risk_level(risk_level),
        "explanation": COGNIGUARD_SCORE_EXPLANATION,
    }


def _apply_risk_score_bounds(raw_score: int, risk_level: Any) -> Optional[int]:
    if risk_level == "low":
        return min(100, max(raw_score, 75))
    if risk_level == "medium":
        return min(74, max(raw_score, 50))
    if risk_level == "high":
        return min(raw_score, 49)
    return None


def _dialogue_result_is_low_risk(record: dict[str, Any]) -> bool:
    dialogue_result = record.get("dialogue_result")
    if isinstance(dialogue_result, dict):
        return dialogue_result.get("risk_level") == "low"
    components = record.get("components")
    if isinstance(components, list) and "dialogue" in components:
        return record.get("risk_level") == "low"
    return False


def _clock_structure_score_value(record: dict[str, Any]) -> float:
    score_result = compute_clock_structure_score(record)
    score = score_result.get("score")
    return float(score) if isinstance(score, (int, float)) and not isinstance(score, bool) else 0.0


def _score_band(score: int) -> str:
    if score >= 85:
        return "整体稳定"
    if score >= 75:
        return "轻微波动"
    if score >= 50:
        return "建议关注"
    return "明显异常"


def _domain_changes(
    first_session: dict[str, Any],
    last_session: dict[str, Any],
) -> dict[str, Optional[float]]:
    first_scores = first_session.get("domain_scores", {})
    last_scores = last_session.get("domain_scores", {})
    changes: dict[str, Optional[float]] = {}
    for domain in COGNITIVE_DOMAINS:
        first_value = first_scores.get(domain)
        last_value = last_scores.get(domain)
        if first_value is None or last_value is None:
            changes[domain] = None
            continue
        try:
            changes[domain] = round(float(last_value) - float(first_value), 3)
        except (TypeError, ValueError):
            changes[domain] = None
    return changes
