from __future__ import annotations

from core.assessment_flow import (
    FLOW_STEP_BRIEF,
    FLOW_STEP_CLOCK_TEST,
    FLOW_STEP_FINISH,
    build_assessment_flow_summary,
    recommend_clock_test,
)
from core.schemas import COGNITIVE_DOMAINS


def test_low_risk_stable_dialogue_does_not_force_clock_test() -> None:
    result = {
        "risk_level": "low",
        "domain_scores": {domain: 0.86 for domain in COGNITIVE_DOMAINS},
    }

    recommendation = recommend_clock_test(result)

    assert recommendation["recommended"] is False
    assert recommendation["manual_override_allowed"] is True


def test_medium_risk_dialogue_recommends_clock_test() -> None:
    recommendation = recommend_clock_test(
        {
            "risk_level": "medium",
            "domain_scores": {domain: 0.8 for domain in COGNITIVE_DOMAINS},
        }
    )

    assert recommendation["recommended"] is True
    assert any("中等风险" in reason for reason in recommendation["reasons"])


def test_unknown_dialogue_recommends_clock_test() -> None:
    recommendation = recommend_clock_test(
        {
            "risk_level": "unknown",
            "domain_scores": {domain: None for domain in COGNITIVE_DOMAINS},
        }
    )

    assert recommendation["recommended"] is True
    assert any("无法评估" in reason for reason in recommendation["reasons"])


def test_low_risk_visuospatial_low_score_recommends_clock_test() -> None:
    scores = {domain: 0.9 for domain in COGNITIVE_DOMAINS}
    scores["visuospatial"] = 0.6

    recommendation = recommend_clock_test(
        {
            "risk_level": "low",
            "domain_scores": scores,
        }
    )

    assert recommendation["recommended"] is True
    assert any("视觉空间" in reason for reason in recommendation["reasons"])


def test_low_risk_executive_low_score_recommends_clock_test() -> None:
    scores = {domain: 0.9 for domain in COGNITIVE_DOMAINS}
    scores["executive_function"] = 0.65

    recommendation = recommend_clock_test(
        {
            "risk_level": "low",
            "domain_scores": scores,
        }
    )

    assert recommendation["recommended"] is True
    assert any("执行功能" in reason for reason in recommendation["reasons"])


def test_flow_summary_productizes_low_risk_dialogue_finish_step() -> None:
    summary = build_assessment_flow_summary(
        {
            "risk_level": "low",
            "domain_scores": {domain: 0.86 for domain in COGNITIVE_DOMAINS},
        }
    )

    assert summary["next_task"]["step_id"] == FLOW_STEP_FINISH
    assert summary["next_task"]["primary_action_label"] == "完成本次访谈"
    assert summary["score_cards"][0]["title"] == "对话评估参考分"
    assert summary["score_cards"][0]["value"] == "86 / 100"
    assert summary["score_cards"][1]["value"] == "待补充"
    assert "正式医学量表" in summary["score_policy"]["notice"]


def test_flow_summary_productizes_clock_test_step_for_triggered_dialogue() -> None:
    summary = build_assessment_flow_summary(
        {
            "risk_level": "medium",
            "domain_scores": {domain: 0.8 for domain in COGNITIVE_DOMAINS},
        }
    )

    assert summary["next_task"]["step_id"] == FLOW_STEP_CLOCK_TEST
    assert summary["next_task"]["status"] == "needs_clock"
    assert "小小游戏" in summary["next_task"]["elder_message"]
    assert "请您在纸上画一个钟，指到 11 点 10 分" in summary["next_task"]["elder_message"]
    assert "不着急，慢慢来" in summary["next_task"]["elder_message"]
    assert "不稳定" not in summary["next_task"]["elder_message"]
    assert summary["clock_recommendation"]["recommended"] is True


def test_flow_summary_moves_to_brief_after_clock_result_exists() -> None:
    clock_result = {
        "risk_level": "low",
        "domain_scores": {
            "orientation": None,
            "memory": None,
            "language": None,
            "executive_function": 0.9,
            "attention": None,
            "visuospatial": 0.9,
        },
        "cdt_features": {
            "numbers_complete": True,
            "number_order_correct": True,
            "number_spacing": "normal",
            "number_distribution": "balanced",
            "hands_present": True,
            "target_time_match": True,
            "center_anchor_clear": True,
        },
    }

    summary = build_assessment_flow_summary(
        {
            "risk_level": "low",
            "domain_scores": {domain: 0.9 for domain in COGNITIVE_DOMAINS},
        },
        clock_result=clock_result,
    )

    assert summary["next_task"]["step_id"] == FLOW_STEP_BRIEF
    assert summary["score_cards"][1]["value"] == "10 / 10"
    assert summary["score_cards"][2]["value"] == "90 / 100"
