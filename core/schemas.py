from __future__ import annotations

from copy import deepcopy
from typing import Any, Optional


COGNITIVE_DOMAINS: tuple[str, ...] = (
    "orientation",
    "memory",
    "language",
    "executive_function",
    "attention",
    "visuospatial",
)

DOMAIN_LABELS: dict[str, str] = {
    "orientation": "时间定向",
    "memory": "记忆",
    "language": "语言",
    "executive_function": "执行功能",
    "attention": "注意力",
    "visuospatial": "视觉空间",
}

RISK_LEVEL_LABELS: dict[str, str] = {
    "low": "低风险",
    "medium": "中等风险",
    "high": "高风险",
    "unknown": "无法评估",
}

SOURCE_LABELS: dict[str, str] = {
    "qwen": "Qwen 文本模型",
    "qwen-vl": "Qwen-VL 视觉模型",
    "mock": "模拟结果",
    "preset": "预设问题",
    "fallback": "兜底结果",
    "sqlite_current_user": "SQLite 当前用户记录",
    "empty_current_user": "当前用户暂无记录",
    "sqlite": "SQLite 数据库记录",
    "fixtures": "演示模拟数据",
    "unknown": "未知来源",
}

RISK_LEVELS: tuple[str, ...] = ("low", "medium", "high", "unknown")
RISK_ORDER: dict[str, int] = {
    "unknown": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
}

CDT_NUMBER_SPACING_VALUES = ("normal", "crowded", "shifted", "irregular", "unknown")
CDT_NUMBER_DISTRIBUTION_VALUES = (
    "balanced",
    "right_shifted",
    "left_shifted",
    "clustered",
    "unknown",
)

CDT_VALUE_LABELS: dict[str, str] = {
    "balanced": "分布均衡",
    "right_shifted": "向右偏移",
    "left_shifted": "向左偏移",
    "clustered": "聚集",
    "crowded": "拥挤",
    "shifted": "偏移",
    "irregular": "不规则",
    "normal": "正常",
    "unknown": "无法判断",
}

DISCLAIMER = (
    "本系统仅为技术原型，输出内容仅作认知健康风险提示参考，"
    "不构成医学诊断或治疗建议。如有健康疑虑，请咨询专业医生。"
)

FALLBACK_EXPLANATION = (
    "本次模型输出无法可靠解析，系统未生成有效评估结果。请重试或使用人工检查。"
)

FALLBACK_RESULT: dict[str, Any] = {
    "domain_scores": {domain: None for domain in COGNITIVE_DOMAINS},
    "evidence": [],
    "risk_level": "unknown",
    "explanation": FALLBACK_EXPLANATION,
    "disclaimer": DISCLAIMER,
}


def empty_domain_scores() -> dict[str, None]:
    return {domain: None for domain in COGNITIVE_DOMAINS}


def normalize_domain_scores(scores: dict[str, Optional[float]]) -> dict[str, Optional[float]]:
    normalized: dict[str, Optional[float]] = {}
    for domain in COGNITIVE_DOMAINS:
        value = scores.get(domain)
        normalized[domain] = None if value is None else float(value)
    return normalized


def fallback_result() -> dict[str, Any]:
    return deepcopy(FALLBACK_RESULT)


def display_risk_level(value: Any) -> str:
    return RISK_LEVEL_LABELS.get(str(value).strip(), RISK_LEVEL_LABELS["unknown"])


def display_source(value: Any) -> str:
    return SOURCE_LABELS.get(str(value).strip(), SOURCE_LABELS["unknown"])


def display_cdt_feature_value(feature_name: str, value: Any) -> str:
    if isinstance(value, bool):
        if feature_name == "target_time_match":
            return "符合目标时间" if value else "不符合目标时间"
        return "是" if value else "否"
    if value is None:
        return "暂无"
    return CDT_VALUE_LABELS.get(str(value).strip(), str(value))


def validate_dialogue_assessment(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("dialogue assessment must be a JSON object")

    risk_level = payload.get("risk_level")
    if risk_level not in RISK_LEVELS:
        raise ValueError("risk_level is invalid")

    explanation = payload.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation is required")

    disclaimer = payload.get("disclaimer")
    if not isinstance(disclaimer, str) or "不构成医学诊断" not in disclaimer:
        raise ValueError("valid disclaimer is required")

    return {
        "domain_scores": _validate_domain_scores(payload.get("domain_scores")),
        "evidence": _validate_dialogue_evidence(payload.get("evidence")),
        "risk_level": risk_level,
        "explanation": explanation.strip(),
        "disclaimer": DISCLAIMER,
        "is_mock": False,
    }


def calibrate_dialogue_result(result: dict[str, Any]) -> dict[str, Any]:
    calibrated = deepcopy(result)
    scores = calibrated.get("domain_scores")
    if not isinstance(scores, dict):
        calibrated["calibrated"] = False
        calibrated["calibration_notes"] = []
        return calibrated

    notes: list[str] = []
    severe_key_domains = [
        domain
        for domain in ("orientation", "memory", "executive_function")
        if _score_at_or_below(scores.get(domain), 0.3)
    ]
    low_domains = [
        domain
        for domain in COGNITIVE_DOMAINS
        if _score_at_or_below(scores.get(domain), 0.4)
    ]
    severe_signal = _has_severe_dialogue_signal(calibrated)

    if len(severe_key_domains) >= 2 and _key_severe_combo_is_high(severe_key_domains):
        if _raise_risk_level(calibrated, "high"):
            notes.append(
                "risk_level raised to high because at least two key domains "
                f"were <= 0.3: {', '.join(severe_key_domains)}"
            )
    elif len(low_domains) >= 4:
        if _raise_risk_level(calibrated, "high"):
            notes.append(
                "risk_level raised to high because at least four domains "
                f"were <= 0.4: {', '.join(low_domains)}"
            )
    elif severe_signal:
        if _raise_risk_level(calibrated, "high"):
            notes.append("risk_level raised to high because severe dialogue signal was detected")

    if _should_boost_high_confidence_dialogue(calibrated, scores):
        boosted_domains = []
        for domain, value in scores.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value < 0.95:
                scores[domain] = 0.95
                boosted_domains.append(domain)
        if boosted_domains:
            notes.append(
                "domain scores raised to >= 0.95 because low-risk dialogue "
                f"contained clear fully-correct signals: {', '.join(boosted_domains)}"
            )

    calibrated["domain_scores"] = scores
    calibrated["calibrated"] = bool(notes)
    calibrated["calibration_notes"] = notes
    return calibrated


def _validate_domain_scores(value: Any) -> dict[str, Optional[float]]:
    if not isinstance(value, dict):
        raise ValueError("domain_scores must be an object")

    scores: dict[str, Optional[float]] = {}
    for domain in COGNITIVE_DOMAINS:
        if domain not in value:
            raise ValueError(f"domain_scores missing {domain}")

        score = value[domain]
        if score is None:
            scores[domain] = None
            continue

        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError(f"domain_scores.{domain} must be a number or null")

        numeric_score = float(score)
        if not 0 <= numeric_score <= 1:
            raise ValueError(f"domain_scores.{domain} must be between 0 and 1")

        scores[domain] = numeric_score

    return scores


def _validate_dialogue_evidence(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("evidence must be an array")

    evidence: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("evidence items must be objects")

        domain = item.get("domain")
        source = item.get("source")
        text = item.get("text")

        if domain not in COGNITIVE_DOMAINS:
            raise ValueError("evidence domain is invalid")
        if source != "dialog":
            raise ValueError("dialogue evidence source must be dialog")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("evidence text is required")

        evidence.append(
            {
                "domain": domain,
                "source": source,
                "text": text.strip(),
            }
        )

    return evidence


def normalize_clock_assessment_payload(payload: Any) -> dict[str, Any]:
    source = payload if isinstance(payload, dict) else {}
    clock_findings = _normalize_clock_findings(source.get("clock_findings"))
    cdt_features = _normalize_cdt_features(source.get("cdt_features"))
    evidence = _normalize_clock_evidence(source.get("evidence"), clock_findings)

    return {
        "domain_scores": _normalize_partial_domain_scores(
            source.get("domain_scores")
        ),
        "evidence": evidence,
        "clock_findings": clock_findings,
        "cdt_features": cdt_features,
        "risk_level": _normalize_risk_level(source.get("risk_level")),
        "explanation": _normalize_text(
            source.get("explanation"),
            "画钟图片分析结果仅作为技术原型风险提示参考。",
        ),
        "disclaimer": _normalize_disclaimer(source.get("disclaimer")),
        "is_mock": False,
    }


def validate_clock_assessment(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("clock assessment must be a JSON object")

    risk_level = payload.get("risk_level")
    if risk_level not in RISK_LEVELS:
        raise ValueError("risk_level is invalid")

    explanation = payload.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise ValueError("explanation is required")

    disclaimer = payload.get("disclaimer")
    if not isinstance(disclaimer, str) or "不构成医学诊断" not in disclaimer:
        raise ValueError("valid disclaimer is required")

    clock_findings = payload.get("clock_findings")
    if not isinstance(clock_findings, dict):
        raise ValueError("clock_findings must be an object")

    return {
        "domain_scores": _validate_domain_scores(payload.get("domain_scores")),
        "evidence": _validate_clock_evidence(payload.get("evidence")),
        "clock_findings": _normalize_clock_findings(clock_findings),
        "cdt_features": _normalize_cdt_features(payload.get("cdt_features")),
        "risk_level": risk_level,
        "explanation": explanation.strip(),
        "disclaimer": DISCLAIMER,
        "is_mock": False,
    }


def calibrate_clock_result(result: dict[str, Any]) -> dict[str, Any]:
    calibrated = deepcopy(result)
    scores = calibrated.get("domain_scores")
    if not isinstance(scores, dict):
        calibrated["calibrated"] = False
        calibrated["calibration_notes"] = []
        return calibrated

    notes: list[str] = []
    cdt_features = _normalize_cdt_features(calibrated.get("cdt_features"))
    calibrated["cdt_features"] = cdt_features
    signal_text = _clock_signal_text(calibrated)
    has_number_issue = _contains_any(
        signal_text,
        (
            "数字明显偏移",
            "数字偏移",
            "整体偏移",
            "向右偏",
            "向左偏",
            "偏右",
            "偏左",
            "集中",
            "挤在",
            "遗漏",
            "缺失",
            "顺序混乱",
            "间距不均",
        ),
    )
    number_distribution = cdt_features["number_distribution"]
    number_spacing = cdt_features["number_spacing"]
    target_time_match = cdt_features["target_time_match"]
    has_number_feature_issue = number_distribution in {
        "right_shifted",
        "left_shifted",
        "clustered",
    } or number_spacing in {"crowded", "shifted", "irregular"}
    has_hand_feature_issue = target_time_match is False
    has_hand_issue = _contains_any(
        signal_text,
        (
            "指针错误",
            "指针方向错误",
            "指针方向不准确",
            "方向不准确",
            "不够准确",
            "目标时间错误",
            "时间错误",
            "长短针混淆",
            "指针不符合",
            "指针不准",
        ),
    )
    has_number_issue = has_number_issue or has_number_feature_issue
    has_hand_issue = has_hand_issue or has_hand_feature_issue
    has_severe_issue = _contains_any(
        signal_text,
        (
            "无法辨认",
            "难以辨认",
            "严重缺失",
            "大面积缺失",
            "基本失败",
            "没有有效画钟",
        ),
    )

    if has_number_issue:
        if _cap_domain_score(scores, "visuospatial", 0.5):
            notes.append(
                "visuospatial capped at 0.5 because CDT number layout issue was detected"
            )
        if _raise_risk_level(calibrated, "medium"):
            notes.append(
                "risk_level raised to medium because CDT number layout issue was detected"
            )

    if has_hand_issue:
        if _cap_domain_score(scores, "executive_function", 0.6):
            notes.append(
                "executive_function capped at 0.6 because CDT hand/time issue was detected"
            )
        if _raise_risk_level(calibrated, "medium"):
            notes.append(
                "risk_level raised to medium because CDT hand/time issue was detected"
            )

    if has_number_issue and has_hand_issue:
        if _raise_risk_level(calibrated, "medium"):
            notes.append(
                "risk_level raised to medium because number layout and hand issues "
                "were both detected"
            )

    if has_severe_issue:
        if _raise_risk_level(calibrated, "high"):
            notes.append("risk_level raised to high because severe clock failure was detected")

    if _should_boost_normal_clock(calibrated, cdt_features, has_number_issue, has_hand_issue, has_severe_issue):
        boosted_domains = []
        if _floor_domain_score(scores, "visuospatial", 0.9):
            boosted_domains.append("visuospatial")
        if _floor_domain_score(scores, "executive_function", 0.9):
            boosted_domains.append("executive_function")
        if boosted_domains:
            notes.append(
                "domain scores raised to >= 0.9 because CDT features were "
                f"normal and risk_level was low: {', '.join(boosted_domains)}"
            )

    calibrated["domain_scores"] = scores
    calibrated["calibrated"] = bool(notes)
    calibrated["calibration_notes"] = notes
    return calibrated


def _normalize_partial_domain_scores(value: Any) -> dict[str, Optional[float]]:
    scores = value if isinstance(value, dict) else {}
    return {domain: _normalize_score(scores.get(domain)) for domain in COGNITIVE_DOMAINS}


def _normalize_score(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if not 0 <= score <= 1:
        return None
    return score


def _normalize_risk_level(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"

    normalized = value.strip().lower()
    risk_map = {
        "low": "low",
        "low risk": "low",
        "低": "low",
        "低风险": "low",
        "medium": "medium",
        "medium risk": "medium",
        "moderate": "medium",
        "中": "medium",
        "中等": "medium",
        "中风险": "medium",
        "中等风险": "medium",
        "high": "high",
        "high risk": "high",
        "高": "high",
        "高风险": "high",
        "unknown": "unknown",
        "未知": "unknown",
        "不确定": "unknown",
        "无法判断": "unknown",
    }
    return risk_map.get(normalized, "unknown")


def _normalize_clock_evidence(
    value: Any,
    clock_findings: dict[str, Any],
) -> list[dict[str, str]]:
    if value is None:
        value = clock_findings.get("visuospatial_evidence", [])

    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []

    evidence: list[dict[str, str]] = []
    for item in items:
        normalized = _normalize_clock_evidence_item(item)
        if normalized is not None:
            evidence.append(normalized)
    return evidence


def _normalize_clock_evidence_item(value: Any) -> Optional[dict[str, str]]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return {
            "domain": _infer_clock_domain(text),
            "source": "clock",
            "text": text,
        }

    if not isinstance(value, dict):
        return None

    raw_text = value.get("text") or value.get("evidence") or value.get("content")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    raw_domain = value.get("domain")
    domain = raw_domain if raw_domain in COGNITIVE_DOMAINS else _infer_clock_domain(raw_text)
    return {
        "domain": domain,
        "source": "clock",
        "text": raw_text.strip(),
    }


def _validate_clock_evidence(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        raise ValueError("evidence must be an array")

    evidence: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            raise ValueError("evidence items must be objects")

        domain = item.get("domain")
        source = item.get("source")
        text = item.get("text")

        if domain not in COGNITIVE_DOMAINS:
            raise ValueError("evidence domain is invalid")
        if source != "clock":
            raise ValueError("clock evidence source must be clock")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("evidence text is required")

        evidence.append(
            {
                "domain": domain,
                "source": source,
                "text": text.strip(),
            }
        )
    return evidence


def _normalize_clock_findings(value: Any) -> dict[str, Any]:
    findings = value if isinstance(value, dict) else {}
    return {
        "number_placement": _normalize_text(
            findings.get("number_placement"),
            "未获得可靠的数字布局观察。",
        ),
        "hand_accuracy": _normalize_text(
            findings.get("hand_accuracy"),
            "未获得可靠的指针方向观察。",
        ),
        "visuospatial_evidence": _normalize_text_list(
            findings.get("visuospatial_evidence")
        ),
    }


def _normalize_cdt_features(value: Any) -> dict[str, Any]:
    features = value if isinstance(value, dict) else {}
    return {
        "numbers_complete": _normalize_optional_bool(features.get("numbers_complete")),
        "number_order_correct": _normalize_optional_bool(
            features.get("number_order_correct")
        ),
        "number_spacing": _normalize_enum(
            features.get("number_spacing"),
            CDT_NUMBER_SPACING_VALUES,
        ),
        "number_distribution": _normalize_enum(
            features.get("number_distribution"),
            CDT_NUMBER_DISTRIBUTION_VALUES,
        ),
        "hands_present": _normalize_optional_bool(features.get("hands_present")),
        "target_time_match": _normalize_optional_bool(
            features.get("target_time_match")
        ),
        "center_anchor_clear": _normalize_optional_bool(
            features.get("center_anchor_clear")
        ),
    }


def _normalize_text(value: Any, default: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return default


def _normalize_text_list(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _normalize_disclaimer(value: Any) -> str:
    if isinstance(value, str) and "不构成医学诊断" in value:
        return value.strip()
    return DISCLAIMER


def _normalize_optional_bool(value: Any) -> Optional[bool]:
    return value if isinstance(value, bool) else None


def _normalize_enum(value: Any, allowed_values: tuple[str, ...]) -> str:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in allowed_values:
            return normalized
    return "unknown"


def _infer_clock_domain(text: str) -> str:
    if any(keyword in text for keyword in ("步骤", "计划", "执行", "指令", "要求")):
        return "executive_function"
    return "visuospatial"


def _score_at_or_below(value: Any, threshold: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value <= threshold
    )


def _key_severe_combo_is_high(domains: list[str]) -> bool:
    return "orientation" in domains or len(domains) >= 3


def _has_severe_dialogue_signal(result: dict[str, Any]) -> bool:
    signal_text = _dialogue_signal_text(result)
    return _contains_any(
        signal_text,
        (
            "完全无法判断日期",
            "完全无法判断时间",
            "完全无法判断地点",
            "无法判断日期地点",
            "日期地点混乱",
            "时间地点混乱",
            "不知道日期",
            "不知道地点",
            "无法回忆核心内容",
            "否认刚刚发生过",
            "明显答非所问",
            "无法完成基本步骤",
            "不会使用电话",
            "不知道按哪里",
        ),
    )


def _should_boost_high_confidence_dialogue(
    result: dict[str, Any],
    scores: dict[str, Any],
) -> bool:
    if result.get("risk_level") != "low":
        return False
    values = [
        value
        for value in scores.values()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not values or min(values) < 0.78:
        return False
    signal_text = _dialogue_signal_text(result)
    return _has_strong_normal_dialogue_signal(signal_text) and not _has_negative_dialogue_signal(signal_text)


def _has_strong_normal_dialogue_signal(text: str) -> bool:
    return _contains_any(
        text,
        (
            "完全正确",
            "回答正确",
            "准确回答",
            "能正确",
            "完整完成",
            "顺利完成",
            "表达清楚",
            "内容连贯",
            "无明显错误",
        ),
    )


def _has_negative_dialogue_signal(text: str) -> bool:
    return _contains_any(
        text,
        (
            "明显错误",
            "不确定",
            "记不清",
            "想不起来",
            "混乱",
            "答非所问",
            "无法",
            "不清楚",
            "遗漏",
            "需要提示",
            "模糊",
            "犹豫明显",
        ),
    )


def _dialogue_signal_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    evidence = result.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)
    explanation = result.get("explanation")
    if isinstance(explanation, str):
        parts.append(explanation)
    return " ".join(parts)


def _raise_risk_level(result: dict[str, Any], minimum: str) -> bool:
    current = result.get("risk_level")
    if current not in RISK_ORDER or minimum not in RISK_ORDER:
        return False
    if RISK_ORDER[current] >= RISK_ORDER[minimum]:
        return False
    result["risk_level"] = minimum
    return True


def _cap_domain_score(
    scores: dict[str, Optional[float]],
    domain: str,
    cap: float,
) -> bool:
    value = scores.get(domain)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if value <= cap:
        return False
    scores[domain] = cap
    return True


def _floor_domain_score(
    scores: dict[str, Optional[float]],
    domain: str,
    floor: float,
) -> bool:
    value = scores.get(domain)
    if value is None:
        scores[domain] = floor
        return True
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    if value >= floor:
        return False
    scores[domain] = floor
    return True


def _should_boost_normal_clock(
    result: dict[str, Any],
    cdt_features: dict[str, Any],
    has_number_issue: bool,
    has_hand_issue: bool,
    has_severe_issue: bool,
) -> bool:
    if result.get("risk_level") != "low":
        return False
    if has_number_issue or has_hand_issue or has_severe_issue:
        return False
    return _cdt_features_all_normal(cdt_features) or _clock_structure_score(cdt_features) >= 9


def _cdt_features_all_normal(cdt_features: dict[str, Any]) -> bool:
    return (
        cdt_features.get("numbers_complete") is True
        and cdt_features.get("number_order_correct") is True
        and cdt_features.get("number_spacing") == "normal"
        and cdt_features.get("number_distribution") == "balanced"
        and cdt_features.get("hands_present") is True
        and cdt_features.get("target_time_match") is True
        and cdt_features.get("center_anchor_clear") is True
    )


def _clock_structure_score(cdt_features: dict[str, Any]) -> float:
    score = 0.0
    score += 1.5 if cdt_features.get("numbers_complete") is True else 0
    score += 1.5 if cdt_features.get("number_order_correct") is True else 0
    score += 1.5 if cdt_features.get("number_spacing") == "normal" else 0
    score += 1.5 if cdt_features.get("number_distribution") == "balanced" else 0
    score += 1.0 if cdt_features.get("hands_present") is True else 0
    score += 2.0 if cdt_features.get("target_time_match") is True else 0
    score += 1.0 if cdt_features.get("center_anchor_clear") is True else 0
    return score


def _clock_signal_text(result: dict[str, Any]) -> str:
    parts: list[str] = []
    findings = result.get("clock_findings")
    if isinstance(findings, dict):
        for key in ("number_placement", "hand_accuracy"):
            value = findings.get(key)
            if isinstance(value, str):
                parts.append(value)
        evidence_items = findings.get("visuospatial_evidence")
        if isinstance(evidence_items, list):
            parts.extend(item for item in evidence_items if isinstance(item, str))

    evidence = result.get("evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)

    explanation = result.get("explanation")
    if isinstance(explanation, str):
        parts.append(explanation)

    return " ".join(parts)


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)
