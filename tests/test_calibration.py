from __future__ import annotations

from typing import Optional

from core.schemas import (
    DISCLAIMER,
    calibrate_clock_result,
    calibrate_dialogue_result,
)


def test_dialogue_low_key_domain_combo_raises_risk_to_high() -> None:
    report = _dialogue_report(
        risk_level="medium",
        domain_scores={
            "orientation": 0.3,
            "memory": 0.2,
            "language": 0.7,
            "executive_function": 0.5,
            "attention": 0.6,
            "visuospatial": None,
        },
    )

    calibrated = calibrate_dialogue_result(report)

    assert calibrated["risk_level"] == "high"
    assert calibrated["calibrated"] is True
    assert calibrated["calibration_notes"]


def test_dialogue_mild_decline_scores_do_not_raise_to_high() -> None:
    report = _dialogue_report(
        risk_level="medium",
        domain_scores={
            "orientation": 0.55,
            "memory": 0.4,
            "language": 0.65,
            "executive_function": 0.4,
            "attention": 0.6,
            "visuospatial": None,
        },
    )

    calibrated = calibrate_dialogue_result(report)

    assert calibrated["risk_level"] == "medium"
    assert calibrated["calibrated"] is False
    assert calibrated["calibration_notes"] == []


def test_dialogue_multiple_severe_low_domains_raise_to_high() -> None:
    report = _dialogue_report(
        risk_level="medium",
        domain_scores={
            "orientation": 0.5,
            "memory": 0.2,
            "language": 0.4,
            "executive_function": 0.3,
            "attention": 0.4,
            "visuospatial": 0.2,
        },
    )

    calibrated = calibrate_dialogue_result(report)

    assert calibrated["risk_level"] == "high"
    assert calibrated["calibrated"] is True


def test_dialogue_normal_scores_are_not_raised() -> None:
    report = _dialogue_report(
        risk_level="low",
        domain_scores={
            "orientation": 0.9,
            "memory": 0.85,
            "language": 0.9,
            "executive_function": 0.88,
            "attention": 0.9,
            "visuospatial": 0.82,
        },
    )

    calibrated = calibrate_dialogue_result(report)

    assert calibrated["risk_level"] == "low"
    assert calibrated["domain_scores"] == report["domain_scores"]
    assert calibrated["calibrated"] is False
    assert calibrated["calibration_notes"] == []


def test_dialogue_fully_correct_low_risk_scores_can_be_raised() -> None:
    report = _dialogue_report(
        risk_level="low",
        domain_scores={
            "orientation": 0.8,
            "memory": 0.82,
            "language": 0.85,
            "executive_function": 0.8,
            "attention": 0.81,
            "visuospatial": 0.8,
        },
    )
    report["evidence"] = [
        {
            "domain": "attention",
            "source": "dialog",
            "text": "老人能正确倒背数字，回答完全正确。",
        }
    ]
    report["explanation"] = "全部任务回答正确，表达清楚，内容连贯。"

    calibrated = calibrate_dialogue_result(report)

    assert calibrated["risk_level"] == "low"
    assert min(calibrated["domain_scores"].values()) >= 0.95
    assert calibrated["calibrated"] is True


def test_clock_hand_error_caps_executive_function() -> None:
    report = _clock_report(
        risk_level="low",
        domain_scores={
            "orientation": None,
            "memory": None,
            "language": None,
            "executive_function": 0.85,
            "attention": None,
            "visuospatial": 0.8,
        },
        hand_accuracy="指针方向错误，目标时间错误。",
    )

    calibrated = calibrate_clock_result(report)

    assert calibrated["domain_scores"]["executive_function"] == 0.6
    assert calibrated["risk_level"] == "medium"
    assert calibrated["calibrated"] is True


def test_clock_target_time_mismatch_feature_caps_executive_function() -> None:
    report = _clock_report(
        risk_level="low",
        domain_scores={
            "orientation": None,
            "memory": None,
            "language": None,
            "executive_function": 0.88,
            "attention": None,
            "visuospatial": 0.8,
        },
        cdt_features={"target_time_match": False},
    )

    calibrated = calibrate_clock_result(report)

    assert calibrated["domain_scores"]["executive_function"] == 0.6
    assert calibrated["risk_level"] == "medium"


def test_clock_number_shift_caps_visuospatial() -> None:
    report = _clock_report(
        risk_level="low",
        domain_scores={
            "orientation": None,
            "memory": None,
            "language": None,
            "executive_function": 0.9,
            "attention": None,
            "visuospatial": 0.9,
        },
        number_placement="数字明显偏移，集中在右侧。",
    )

    calibrated = calibrate_clock_result(report)

    assert calibrated["domain_scores"]["visuospatial"] == 0.5
    assert calibrated["risk_level"] == "medium"
    assert calibrated["calibrated"] is True


def test_clock_number_distribution_feature_caps_visuospatial() -> None:
    report = _clock_report(
        risk_level="low",
        domain_scores={
            "orientation": None,
            "memory": None,
            "language": None,
            "executive_function": 0.9,
            "attention": None,
            "visuospatial": 0.92,
        },
        cdt_features={"number_distribution": "right_shifted"},
    )

    calibrated = calibrate_clock_result(report)

    assert calibrated["domain_scores"]["visuospatial"] == 0.5
    assert calibrated["risk_level"] == "medium"


def test_clock_number_and_hand_feature_issues_raise_risk_to_medium() -> None:
    report = _clock_report(
        risk_level="low",
        domain_scores={
            "orientation": None,
            "memory": None,
            "language": None,
            "executive_function": 0.85,
            "attention": None,
            "visuospatial": 0.9,
        },
        cdt_features={
            "number_distribution": "clustered",
            "target_time_match": False,
        },
    )

    calibrated = calibrate_clock_result(report)

    assert calibrated["domain_scores"]["visuospatial"] == 0.5
    assert calibrated["domain_scores"]["executive_function"] == 0.6
    assert calibrated["risk_level"] == "medium"


def test_clock_normal_findings_are_not_changed() -> None:
    report = _clock_report(
        risk_level="low",
        domain_scores={
            "orientation": None,
            "memory": None,
            "language": None,
            "executive_function": 0.9,
            "attention": None,
            "visuospatial": 0.9,
        },
        number_placement="数字布局均匀完整。",
        hand_accuracy="指针方向基本准确。",
    )

    calibrated = calibrate_clock_result(report)

    assert calibrated["risk_level"] == "low"
    assert calibrated["domain_scores"] == report["domain_scores"]
    assert calibrated["calibrated"] is False
    assert calibrated["calibration_notes"] == []


def test_clock_perfect_cdt_features_raise_low_scores_to_nine_tenths() -> None:
    report = _clock_report(
        risk_level="low",
        domain_scores={
            "orientation": None,
            "memory": None,
            "language": None,
            "executive_function": 0.8,
            "attention": None,
            "visuospatial": 0.8,
        },
        number_placement="数字完整、顺序正确、间距均匀。",
        hand_accuracy="指针符合目标时间。",
        cdt_features={
            "numbers_complete": True,
            "number_order_correct": True,
            "number_spacing": "normal",
            "number_distribution": "balanced",
            "hands_present": True,
            "target_time_match": True,
            "center_anchor_clear": True,
        },
    )

    calibrated = calibrate_clock_result(report)

    assert calibrated["risk_level"] == "low"
    assert calibrated["domain_scores"]["visuospatial"] >= 0.9
    assert calibrated["domain_scores"]["executive_function"] >= 0.9
    assert calibrated["calibrated"] is True


def _dialogue_report(risk_level: str, domain_scores: dict) -> dict:
    return {
        "domain_scores": domain_scores,
        "evidence": [
            {
                "domain": "memory",
                "source": "dialog",
                "text": "模拟证据。",
            }
        ],
        "risk_level": risk_level,
        "explanation": "模拟评估说明。",
        "disclaimer": DISCLAIMER,
        "is_mock": False,
    }


def _clock_report(
    risk_level: str,
    domain_scores: dict,
    number_placement: str = "数字布局观察。",
    hand_accuracy: str = "指针方向观察。",
    cdt_features: Optional[dict] = None,
) -> dict:
    return {
        "domain_scores": domain_scores,
        "evidence": [
            {
                "domain": "visuospatial",
                "source": "clock",
                "text": number_placement,
            },
            {
                "domain": "executive_function",
                "source": "clock",
                "text": hand_accuracy,
            },
        ],
        "clock_findings": {
            "number_placement": number_placement,
            "hand_accuracy": hand_accuracy,
            "visuospatial_evidence": [number_placement],
        },
        "cdt_features": cdt_features or {
            "numbers_complete": None,
            "number_order_correct": None,
            "number_spacing": "unknown",
            "number_distribution": "unknown",
            "hands_present": None,
            "target_time_match": None,
            "center_anchor_clear": None,
        },
        "risk_level": risk_level,
        "explanation": "模拟画钟说明。",
        "disclaimer": DISCLAIMER,
        "is_mock": False,
    }
