from __future__ import annotations

import json
import time
from pathlib import Path

from core.config import AppConfig
from core.schemas import COGNITIVE_DOMAINS, RISK_LEVELS
from scripts import evaluate_qwen_quality as quality_eval
from scripts import generate_deepseek_eval_cases as deepseek_eval_gen
from scripts.evaluate_qwen_quality import build_mismatch_summary


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIALOG_CASES_PATH = PROJECT_ROOT / "demo" / "fixtures" / "eval_cases_dialog.json"
CLOCK_CASES_PATH = PROJECT_ROOT / "demo" / "fixtures" / "eval_cases_clock.json"


def test_dialog_eval_cases_have_required_shape() -> None:
    cases = _load_cases(DIALOG_CASES_PATH)

    assert len(cases) >= 9
    labels = [case["label"] for case in cases]
    assert labels.count("normal") >= 3
    assert labels.count("mild_decline") >= 3
    assert labels.count("obvious_issue") >= 3

    for case in cases:
        _assert_common_case_fields(case)
        assert case["label"] in {"normal", "mild_decline", "obvious_issue"}
        assert isinstance(case["dialogue_turns"], list)
        assert len(case["dialogue_turns"]) >= 3
        for turn in case["dialogue_turns"]:
            assert turn["assistant"].strip()
            assert turn["user"].strip()


def test_clock_eval_cases_have_required_shape_and_images_exist() -> None:
    cases = _load_cases(CLOCK_CASES_PATH)

    assert len(cases) >= 3
    image_names = {Path(case["image_path"]).name for case in cases}
    assert {
        "normal_clock.png",
        "spatial_shift_clock.png",
        "wrong_hands_clock.png",
    } <= image_names

    for case in cases:
        _assert_common_case_fields(case)
        assert case.get("target_time", "11:10") == "11:10"
        image_path = PROJECT_ROOT / case["image_path"]
        assert image_path.exists()
        assert image_path.suffix == ".png"


def test_qwen_quality_mismatch_summary_flags_under_penalty() -> None:
    summary = build_mismatch_summary(
        source="qwen",
        risk_level="medium",
        domain_scores={"memory": 0.72, "attention": 0.55},
        expected_risk_level="medium",
        expected_low_domains=["memory", "attention"],
        expected_source="qwen",
    )

    assert "possible_under_penalty:memory=0.72" in summary
    assert not any(item.startswith("possible_under_penalty:attention") for item in summary)


def test_qwen_quality_mismatch_summary_flags_source_and_risk() -> None:
    summary = build_mismatch_summary(
        source="mock",
        risk_level="low",
        domain_scores={"visuospatial": None},
        expected_risk_level="medium",
        expected_low_domains=["visuospatial"],
        expected_source="qwen-vl",
    )

    assert "non_real_source:mock" in summary
    assert "risk_mismatch:expected=medium,actual=low" in summary
    assert "missing_score:visuospatial" in summary


def test_qwen_quality_mismatch_summary_ok_when_expectations_match() -> None:
    summary = build_mismatch_summary(
        source="qwen-vl",
        risk_level="medium",
        domain_scores={"visuospatial": 0.42, "executive_function": 0.5},
        expected_risk_level="medium",
        expected_low_domains=["visuospatial", "executive_function"],
        expected_source="qwen-vl",
    )

    assert summary == ["ok"]


def test_qwen_quality_result_summary_reports_accuracy_rates() -> None:
    results = [
        _fake_result(
            label="normal",
            risk_level="low",
            expected_risk_level="low",
            mismatch_summary=["ok"],
        ),
        _fake_result(
            label="mild_decline",
            risk_level="low",
            expected_risk_level="medium",
            mismatch_summary=["risk_mismatch:expected=medium,actual=low"],
        ),
        _fake_result(
            label="obvious_issue",
            risk_level="high",
            expected_risk_level="high",
            mismatch_summary=["non_real_source:mock"],
        ),
    ]

    summary = quality_eval.build_result_summary(results)

    assert summary["total"] == 3
    assert summary["risk_match_count"] == 2
    assert summary["exact_match_count"] == 1
    assert summary["source_match_count"] == 2
    assert {
        (row["expected"], row["actual"], row["count"])
        for row in summary["confusion_matrix"]
    } == {("high", "high", 1), ("low", "low", 1), ("medium", "low", 1)}
    assert summary["by_label"]["normal"]["risk_match_rate"] == 1.0
    assert summary["by_label"]["mild_decline"]["risk_match_rate"] == 0.0


def test_qwen_quality_result_summary_reports_screening_metrics() -> None:
    results = [
        _fake_result(
            label="normal",
            risk_level="low",
            expected_risk_level="low",
            mismatch_summary=["ok"],
        ),
        _fake_result(
            label="normal",
            risk_level="medium",
            expected_risk_level="low",
            mismatch_summary=["risk_mismatch:expected=low,actual=medium"],
        ),
        _fake_result(
            label="mild_decline",
            risk_level="medium",
            expected_risk_level="medium",
            mismatch_summary=["ok"],
        ),
        _fake_result(
            label="mild_decline",
            risk_level="low",
            expected_risk_level="medium",
            mismatch_summary=["risk_mismatch:expected=medium,actual=low"],
        ),
        _fake_result(
            label="obvious_issue",
            risk_level="medium",
            expected_risk_level="high",
            mismatch_summary=["risk_mismatch:expected=high,actual=medium"],
        ),
    ]

    summary = quality_eval.build_result_summary(results)
    screening = summary["screening"]

    assert screening["expected_abnormal_total"] == 3
    assert screening["abnormal_detected_count"] == 2
    assert screening["abnormal_missed_count"] == 1
    assert screening["abnormal_detection_rate"] == 2 / 3
    assert screening["expected_normal_total"] == 2
    assert screening["normal_low_count"] == 1
    assert screening["false_alarm_count"] == 1
    assert screening["alert_precision"] == 2 / 3
    assert screening["binary_success_count"] == 3
    assert screening["binary_total"] == 5
    assert screening["binary_success_rate"] == 3 / 5
    assert screening["by_expected_risk"]["medium"]["detected"] == 1
    assert screening["by_expected_risk"]["medium"]["missed"] == 1
    assert screening["by_expected_risk"]["high"]["detected"] == 1


def test_qwen_quality_result_summary_reports_cdt_feature_accuracy() -> None:
    results = [
        _fake_result(
            kind="clock",
            label="normal",
            risk_level="low",
            expected_risk_level="low",
            mismatch_summary=["ok"],
            cdt_features={
                "numbers_complete": True,
                "target_time_match": True,
            },
            expected_cdt_features={
                "numbers_complete": True,
                "target_time_match": True,
            },
        ),
        _fake_result(
            kind="clock",
            label="obvious_issue",
            risk_level="high",
            expected_risk_level="high",
            mismatch_summary=["ok"],
            cdt_features={
                "numbers_complete": True,
                "target_time_match": False,
            },
            expected_cdt_features={
                "numbers_complete": False,
                "target_time_match": False,
            },
        ),
        _fake_result(
            label="normal",
            risk_level="low",
            expected_risk_level="low",
            mismatch_summary=["ok"],
        ),
    ]

    summary = quality_eval.build_result_summary(results)
    cdt_summary = summary["cdt_features"]

    assert cdt_summary["case_total"] == 2
    assert cdt_summary["exact_case_match_count"] == 1
    assert cdt_summary["core_comparison_total"] == 2
    assert cdt_summary["core_comparison_match_count"] == 1
    assert cdt_summary["core_comparison_match_rate"] == 1 / 2
    assert cdt_summary["comparison_total"] == 4
    assert cdt_summary["comparison_match_count"] == 3
    assert cdt_summary["by_feature"]["numbers_complete"]["accuracy"] == 1 / 2
    assert cdt_summary["by_feature"]["target_time_match"]["accuracy"] == 1.0
    assert cdt_summary["mismatches"] == [
        {
            "case_id": "case_obvious_issue",
            "feature": "numbers_complete",
            "expected": False,
            "actual": True,
        }
    ]


def test_qwen_quality_summary_markdown_explains_screening_definition() -> None:
    summary = quality_eval.build_result_summary(
        [
            _fake_result(
                label="normal",
                risk_level="low",
                expected_risk_level="low",
                mismatch_summary=["ok"],
            ),
            _fake_result(
                label="obvious_issue",
                risk_level="medium",
                expected_risk_level="high",
                mismatch_summary=["risk_mismatch:expected=high,actual=medium"],
            ),
        ]
    )

    markdown = "\n".join(quality_eval._format_summary_markdown(summary))

    assert "Screening-Oriented Metrics" in markdown
    assert "abnormal_detection_rate" in markdown
    assert "actual `medium/high` are treated as family alerts" in markdown


def test_qwen_quality_summary_markdown_includes_cdt_feature_metrics() -> None:
    summary = quality_eval.build_result_summary(
        [
            _fake_result(
                kind="clock",
                label="normal",
                risk_level="low",
                expected_risk_level="low",
                mismatch_summary=["ok"],
                cdt_features={"numbers_complete": True},
                expected_cdt_features={"numbers_complete": True},
            )
        ]
    )

    markdown = "\n".join(quality_eval._format_summary_markdown(summary))

    assert "CDT Feature Metrics" in markdown
    assert "cdt_core_structure_field_accuracy" in markdown
    assert "cdt_feature_field_accuracy" in markdown
    assert "`numbers_complete`" in markdown


def test_qwen_quality_parse_args_accepts_generated_dialog_case_path() -> None:
    args = quality_eval._parse_args(
        [
            "--dialog",
            "--dialog-cases",
            "data/eval/deepseek_dialog_cases.json",
            "--report",
            "data/eval/qwen_eval_report.md",
        ]
    )

    assert args.dialog is True
    assert args.dialog_cases == "data/eval/deepseek_dialog_cases.json"
    assert args.report == "data/eval/qwen_eval_report.md"


def test_deepseek_eval_prompt_uses_independent_mock_case_schema() -> None:
    prompt = deepseek_eval_gen.build_generation_prompt(
        label="mild_decline",
        count=4,
        start_index=1,
        seed=123,
    )

    assert "cases" in prompt
    assert 'label 必须固定为 "mild_decline"' in prompt
    assert 'expected_risk_level 必须为 "medium"' in prompt
    assert "模拟数据，非临床数据" in prompt
    assert "orientation, memory, language, executive_function, attention, visuospatial" in prompt


def test_deepseek_generated_cases_are_validated_without_api_calls() -> None:
    raw_cases = [
        {
            "label": "normal",
            "is_mock": True,
            "dialogue_turns": [
                {"assistant": f"问题 {index}", "user": f"回答 {index}"}
                for index in range(1, 7)
            ],
            "expected_risk_level": "low",
            "expected_low_domains": ["memory"],
            "notes": "用于测试。",
        }
    ]

    cases = deepseek_eval_gen.validate_generated_cases(raw_cases, label="normal")

    assert cases[0]["case_id"] == "ds_dialog_normal_001"
    assert cases[0]["label"] == "normal"
    assert cases[0]["is_mock"] is True
    assert cases[0]["expected_risk_level"] == "low"
    assert cases[0]["expected_low_domains"] == []
    assert "模拟数据" in cases[0]["notes"]
    assert "非临床数据" in cases[0]["notes"]


def test_qwen_quality_workers_one_keeps_sequential_behavior(monkeypatch) -> None:
    call_order: list[str] = []

    def fake_evaluate_dialog_case(case: dict, config: AppConfig) -> dict:
        call_order.append(case["case_id"])
        return _fake_report(source="qwen", model=config.llm_model)

    monkeypatch.setattr(
        quality_eval,
        "_evaluate_dialog_case",
        fake_evaluate_dialog_case,
    )
    tasks = [_make_task(index) for index in range(3)]

    results = quality_eval._run_evaluation_tasks(
        tasks,
        AppConfig(llm_model="fake-qwen"),
        workers=1,
    )

    assert call_order == ["case_0", "case_1", "case_2"]
    assert [result["case_id"] for result in results] == call_order


def test_qwen_quality_workers_parallel_keeps_report_order(monkeypatch) -> None:
    def fake_evaluate_dialog_case(case: dict, config: AppConfig) -> dict:
        time.sleep(case["sleep"])
        return _fake_report(source="qwen", model=config.llm_model)

    monkeypatch.setattr(
        quality_eval,
        "_evaluate_dialog_case",
        fake_evaluate_dialog_case,
    )
    tasks = [
        _make_task(0, sleep=0.03),
        _make_task(1, sleep=0.01),
        _make_task(2, sleep=0.0),
    ]

    results = quality_eval._run_evaluation_tasks(
        tasks,
        AppConfig(llm_model="fake-qwen"),
        workers=3,
    )

    assert len(results) == 3
    assert [result["case_id"] for result in results] == ["case_0", "case_1", "case_2"]


def test_qwen_quality_limit_applies_to_selected_cases() -> None:
    tasks = quality_eval._build_evaluation_tasks(
        run_dialog=True,
        run_clock=True,
        limit=3,
    )

    assert len(tasks) == 3
    assert [task["order"] for task in tasks] == [0, 1, 2]
    assert {task["kind"] for task in tasks} == {"dialog"}


def test_qwen_quality_case_error_does_not_stop_other_cases(monkeypatch) -> None:
    def fake_evaluate_dialog_case(case: dict, config: AppConfig) -> dict:
        if case["case_id"] == "case_1":
            raise RuntimeError("temporary failure with sk-secret-example")
        return _fake_report(source="qwen", model=config.llm_model)

    monkeypatch.setattr(
        quality_eval,
        "_evaluate_dialog_case",
        fake_evaluate_dialog_case,
    )
    tasks = [_make_task(index) for index in range(3)]

    results = quality_eval._run_evaluation_tasks(
        tasks,
        AppConfig(llm_model="fake-qwen"),
        workers=2,
    )

    assert [result["case_id"] for result in results] == ["case_0", "case_1", "case_2"]
    assert results[1]["source"] == "case_error"
    assert results[1]["error_type"] == "RuntimeError"
    assert "sk-secret-example" not in results[1]["error_message"]
    assert results[0]["source"] == "qwen"
    assert results[2]["source"] == "qwen"


def _load_cases(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    assert isinstance(cases, list)
    return cases


def _assert_common_case_fields(case: dict) -> None:
    assert case["case_id"].strip()
    assert case["is_mock"] is True
    assert case["expected_risk_level"] in RISK_LEVELS
    assert isinstance(case["expected_low_domains"], list)
    assert set(case["expected_low_domains"]) <= set(COGNITIVE_DOMAINS)
    assert "模拟数据" in case["notes"]
    assert "非临床数据" in case["notes"]


def _make_task(index: int, *, sleep: float = 0.0) -> dict:
    return {
        "order": index,
        "kind": "dialog",
        "expected_source": "qwen",
        "case": {
            "case_id": f"case_{index}",
            "label": "normal",
            "expected_risk_level": "low",
            "expected_low_domains": [],
            "notes": "模拟数据，非临床数据。",
            "sleep": sleep,
        },
    }


def _fake_report(*, source: str, model: str) -> dict:
    return {
        "metadata": {"source": source, "model": model},
        "risk_level": "low",
        "domain_scores": {domain: 0.9 for domain in COGNITIVE_DOMAINS},
        "calibrated": False,
        "calibration_notes": [],
    }


def _fake_result(
    *,
    kind: str = "dialog",
    label: str,
    risk_level: str,
    expected_risk_level: str,
    mismatch_summary: list[str],
    cdt_features: dict | None = None,
    expected_cdt_features: dict | None = None,
) -> dict:
    return {
        "kind": kind,
        "case_id": f"case_{label}",
        "label": label,
        "source": "qwen",
        "model": "fake-qwen",
        "risk_level": risk_level,
        "expected_risk_level": expected_risk_level,
        "domain_scores": {domain: 0.9 for domain in COGNITIVE_DOMAINS},
        "expected_low_domains": [],
        "mismatch_summary": mismatch_summary,
        "calibrated": False,
        "calibration_notes": [],
        "cdt_features": cdt_features or {},
        "expected_cdt_features": expected_cdt_features or {},
        "notes": "模拟数据，非临床数据。",
    }
