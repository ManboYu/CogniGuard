from __future__ import annotations

from typing import Any

from core.report import (
    compute_clock_structure_score,
    compute_cogniguard_score,
    compute_dialogue_score,
)
from core.schemas import DOMAIN_LABELS, RISK_ORDER, display_risk_level


CLOCK_TRIGGER_DOMAINS: tuple[str, ...] = ("visuospatial", "executive_function")
CLOCK_TRIGGER_SCORE_THRESHOLD = 0.65
SCORE_POLICY_NOTICE = (
    "评分仅为技术原型提示分，用于解释本次访谈、画钟和综合风险提示，"
    "不等同于 MoCA、MMSE 或 Mini-Cog 等正式医学量表。"
)

FLOW_STEP_CLOCK_TEST = "clock_test"
FLOW_STEP_FINISH = "finish"
FLOW_STEP_BRIEF = "brief"


def recommend_clock_test(dialogue_result: dict[str, Any]) -> dict[str, Any]:
    """Decide whether a dialogue result should lead to a clock drawing test."""
    if not isinstance(dialogue_result, dict):
        return _recommend(
            True,
            ["对话结果不可用，需要补充画钟测试或人工检查。"],
            "建议接着完成画钟拍照，补充一张图像样本，便于工作人员综合整理本轮结果。",
            triggered_domains=[],
        )

    risk_level = str(dialogue_result.get("risk_level", "unknown")).strip()
    reasons: list[str] = []
    triggered_domains: list[str] = []

    if risk_level in {"medium", "high", "unknown"}:
        reasons.append(f"当前对话风险等级为{display_risk_level(risk_level)}。")

    scores = dialogue_result.get("domain_scores")
    if isinstance(scores, dict):
        for domain in CLOCK_TRIGGER_DOMAINS:
            score = scores.get(domain)
            if _score_at_or_below(score, CLOCK_TRIGGER_SCORE_THRESHOLD):
                label = DOMAIN_LABELS.get(domain, domain)
                reasons.append(f"{label}得分为 {float(score):.2f}，低于补充测试阈值。")
                triggered_domains.append(domain)

    if reasons:
        return _recommend(
            True,
            reasons,
            "建议接着完成画钟拍照，补充一张图像样本，便于综合整理本轮结果。",
            triggered_domains=triggered_domains,
        )

    return _recommend(
        False,
        ["对话结果为低风险，当前没有自动进入画钟拍照的必要。"],
        "本次可以先结束；工作人员仍可按演示需要手动补充画钟拍照。",
        triggered_domains=[],
    )


def build_assessment_flow_summary(
    dialogue_result: dict[str, Any],
    clock_result: dict[str, Any] | None = None,
    assessment_record: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build product-facing score cards and the next tablet workflow step."""
    clock_recommendation = recommend_clock_test(dialogue_result)
    has_clock = _has_clock_result(clock_result) or _has_clock_result(assessment_record)
    return {
        "score_policy": {
            "notice": SCORE_POLICY_NOTICE,
            "dialogue_scale": "0-100",
            "clock_scale": "0-10",
            "cogniguard_scale": "0-100",
        },
        "score_cards": _build_score_cards(
            dialogue_result,
            clock_result=clock_result,
            assessment_record=assessment_record,
        ),
        "clock_recommendation": clock_recommendation,
        "next_task": _build_next_task(clock_recommendation, has_clock=has_clock),
    }


def _score_at_or_below(value: Any, threshold: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value <= threshold
    )


def _recommend(
    recommended: bool,
    reasons: list[str],
    message: str,
    triggered_domains: list[str],
) -> dict[str, Any]:
    return {
        "recommended": recommended,
        "reasons": reasons,
        "message": message,
        "triggered_domains": triggered_domains,
        "threshold": CLOCK_TRIGGER_SCORE_THRESHOLD,
        "manual_override_allowed": True,
    }


def _build_score_cards(
    dialogue_result: dict[str, Any],
    clock_result: dict[str, Any] | None,
    assessment_record: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    dialogue_score = compute_dialogue_score(dialogue_result)
    clock_score = (
        compute_clock_structure_score(clock_result)
        if isinstance(clock_result, dict)
        else {"score": None, "explanation": "画钟结构分将在补充画钟后生成。"}
    )
    overall_record = (
        assessment_record
        if isinstance(assessment_record, dict)
        else _combined_score_record(dialogue_result, clock_result)
    )
    cogniguard_score = (
        compute_cogniguard_score(overall_record)
        if isinstance(overall_record, dict)
        else {
            "score": None,
            "band": "待补充",
            "explanation": "综合提示分将在形成本轮综合评估后生成。",
        }
    )

    return [
        _score_card(
            "dialogue",
            "对话评估参考分",
            dialogue_score,
            scale_suffix="/ 100",
        ),
        _score_card(
            "clock",
            "画钟结构分",
            clock_score,
            scale_suffix="/ 10",
            pending_text="待补充",
        ),
        _score_card(
            "cogniguard",
            "CogniGuard 综合提示分",
            cogniguard_score,
            scale_suffix="/ 100",
            pending_text="待形成综合评估",
        ),
    ]


def _score_card(
    card_id: str,
    title: str,
    score_result: dict[str, Any],
    scale_suffix: str,
    pending_text: str = "暂无",
) -> dict[str, Any]:
    score = score_result.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        value = f"{score:g} {scale_suffix}"
    else:
        value = pending_text
    return {
        "id": card_id,
        "title": title,
        "value": value,
        "score": score if isinstance(score, (int, float)) and not isinstance(score, bool) else None,
        "band": score_result.get("band", ""),
        "explanation": score_result.get("explanation", SCORE_POLICY_NOTICE),
    }


def _combined_score_record(
    dialogue_result: dict[str, Any],
    clock_result: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not isinstance(dialogue_result, dict):
        return None
    if not isinstance(clock_result, dict):
        return None
    dialogue_scores = dialogue_result.get("domain_scores")
    scores = dict(dialogue_scores) if isinstance(dialogue_scores, dict) else {}
    clock_scores = clock_result.get("domain_scores")
    if isinstance(clock_scores, dict):
        for domain, value in clock_scores.items():
            if value is not None:
                scores[domain] = value
    return {
        "risk_level": _highest_risk_level(
            dialogue_result.get("risk_level"),
            clock_result.get("risk_level"),
        ),
        "domain_scores": scores,
        "dialogue_result": dialogue_result,
        "clock_result": clock_result,
        "components": ["dialogue", "clock"],
        "cdt_features": clock_result.get("cdt_features", {}),
    }


def _highest_risk_level(*levels: Any) -> str:
    valid_levels = [str(level) for level in levels if str(level) in RISK_ORDER]
    if not valid_levels:
        return "unknown"
    return max(valid_levels, key=lambda level: RISK_ORDER[level])


def _has_clock_result(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    components = value.get("components")
    if isinstance(components, list) and "clock" in components:
        return True
    return isinstance(value.get("clock_result"), dict) or isinstance(value.get("cdt_features"), dict)


def _build_next_task(
    clock_recommendation: dict[str, Any],
    has_clock: bool,
) -> dict[str, Any]:
    if has_clock:
        return {
            "step_id": FLOW_STEP_BRIEF,
            "status": "complete",
            "title": "本次综合评估已完成",
            "primary_action_label": "查看认知简报",
            "elder_message": "今天的对话和画钟都完成了，结果已经整理好，请交给家人查看。",
            "staff_message": "可进入认知简报查看最近报告、趋势和家属端提醒。",
            "manual_override_allowed": True,
        }
    if clock_recommendation.get("recommended"):
        return {
            "step_id": FLOW_STEP_CLOCK_TEST,
            "status": "needs_clock",
            "title": "进入画钟拍照环节",
            "primary_action_label": "继续画钟测试",
            "elder_message": "今天的访谈完成了。我们再做一个小小游戏，好吗？请您在纸上画一个钟，指到 11 点 10 分。画好后拍张照片就可以，不着急，慢慢来。",
            "staff_message": clock_recommendation.get("message", ""),
            "manual_override_allowed": True,
        }
    return {
        "step_id": FLOW_STEP_FINISH,
        "status": "dialogue_complete",
        "title": "本次访谈可先结束",
        "primary_action_label": "完成本次访谈",
        "elder_message": "今天的访谈完成了，谢谢您。结果已经整理好，请交给家人查看。",
        "staff_message": clock_recommendation.get("message", ""),
        "manual_override_allowed": True,
    }
