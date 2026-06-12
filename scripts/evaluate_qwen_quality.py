from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import AppConfig, load_config  # noqa: E402
from core.llm_client import evaluate_dialogue  # noqa: E402
from core.schemas import COGNITIVE_DOMAINS  # noqa: E402
from core.vlm_client import analyze_clock_image  # noqa: E402


DIALOG_CASES_PATH = PROJECT_ROOT / "demo" / "fixtures" / "eval_cases_dialog.json"
CLOCK_CASES_PATH = PROJECT_ROOT / "demo" / "fixtures" / "eval_cases_clock.json"
REPORT_PATH = PROJECT_ROOT / "docs" / "qwen-eval-report.md"
LOW_SCORE_THRESHOLD = 0.6
ALERT_RISK_LEVELS = {"medium", "high"}
CDT_FEATURE_KEYS = (
    "numbers_complete",
    "number_order_correct",
    "number_spacing",
    "number_distribution",
    "hands_present",
    "target_time_match",
    "center_anchor_clear",
)
CORE_CDT_FEATURE_KEYS = (
    "numbers_complete",
    "number_order_correct",
    "hands_present",
    "center_anchor_clear",
)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    config = load_config()
    dialog_cases_path = _resolve_project_path(args.dialog_cases, DIALOG_CASES_PATH)
    clock_cases_path = _resolve_project_path(args.clock_cases, CLOCK_CASES_PATH)
    report_path = _resolve_project_path(args.report, REPORT_PATH)

    _print_config_status(config)

    tasks = _build_evaluation_tasks(
        run_dialog=args.dialog or args.all,
        run_clock=args.clock or args.all,
        dialog_cases_path=dialog_cases_path,
        clock_cases_path=clock_cases_path,
        limit=args.limit,
    )
    results = _run_evaluation_tasks(tasks, config, workers=args.workers)

    _write_report(results, config, report_path=report_path)
    print(f"Qwen quality report written to {report_path}")
    return 0


def build_mismatch_summary(
    *,
    source: str,
    risk_level: str,
    domain_scores: dict[str, Any],
    expected_risk_level: str,
    expected_low_domains: list[str],
    expected_source: str,
    low_score_threshold: float = LOW_SCORE_THRESHOLD,
) -> list[str]:
    summary: list[str] = []

    if source != expected_source:
        summary.append(f"non_real_source:{source}")

    if risk_level != expected_risk_level:
        summary.append(
            f"risk_mismatch:expected={expected_risk_level},actual={risk_level}"
        )

    for domain in expected_low_domains:
        score = domain_scores.get(domain)
        if score is None:
            summary.append(f"missing_score:{domain}")
            continue
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            summary.append(f"invalid_score:{domain}")
            continue
        if score > low_score_threshold:
            summary.append(f"possible_under_penalty:{domain}={score:.2f}")

    return summary or ["ok"]


def build_result_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    exact_match_count = sum(1 for result in results if result.get("mismatch_summary") == ["ok"])
    risk_match_count = sum(
        1
        for result in results
        if result.get("risk_level") == result.get("expected_risk_level")
    )
    source_match_count = sum(
        1
        for result in results
        if not any(
            str(item).startswith("non_real_source:")
            for item in result.get("mismatch_summary", [])
        )
    )
    case_error_count = sum(1 for result in results if result.get("error_type"))
    confusion_counter = Counter(
        (
            str(result.get("expected_risk_level", "unknown")),
            str(result.get("risk_level", "unknown")),
        )
        for result in results
    )
    label_buckets: dict[str, dict[str, int]] = {}
    for result in results:
        label = str(result.get("label") or "n/a")
        bucket = label_buckets.setdefault(
            label,
            {"total": 0, "risk_match": 0, "exact_match": 0},
        )
        bucket["total"] += 1
        if result.get("risk_level") == result.get("expected_risk_level"):
            bucket["risk_match"] += 1
        if result.get("mismatch_summary") == ["ok"]:
            bucket["exact_match"] += 1

    return {
        "total": total,
        "exact_match_count": exact_match_count,
        "exact_match_rate": _rate(exact_match_count, total),
        "risk_match_count": risk_match_count,
        "risk_match_rate": _rate(risk_match_count, total),
        "source_match_count": source_match_count,
        "source_match_rate": _rate(source_match_count, total),
        "case_error_count": case_error_count,
        "screening": _build_screening_summary(results),
        "cdt_features": _build_cdt_feature_summary(results),
        "confusion_matrix": [
            {
                "expected": expected,
                "actual": actual,
                "count": count,
            }
            for (expected, actual), count in sorted(confusion_counter.items())
        ],
        "by_label": {
            label: {
                **bucket,
                "risk_match_rate": _rate(bucket["risk_match"], bucket["total"]),
                "exact_match_rate": _rate(bucket["exact_match"], bucket["total"]),
            }
            for label, bucket in sorted(label_buckets.items())
        },
    }


def _build_cdt_feature_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    clock_results = [
        result
        for result in results
        if result.get("kind") == "clock" and result.get("expected_cdt_features")
    ]
    feature_buckets: dict[str, dict[str, int]] = {
        key: {"total": 0, "match": 0} for key in CDT_FEATURE_KEYS
    }
    exact_case_match_count = 0
    mismatch_rows: list[dict[str, Any]] = []

    for result in clock_results:
        expected_features = _normalize_expected_cdt_features(
            result.get("expected_cdt_features")
        )
        actual_features = result.get("cdt_features")
        actual = actual_features if isinstance(actual_features, dict) else {}
        case_mismatches = _build_cdt_feature_mismatches(actual, expected_features)
        if not case_mismatches:
            exact_case_match_count += 1
        for mismatch in case_mismatches:
            mismatch_rows.append({"case_id": result.get("case_id"), **mismatch})

        for key, expected_value in expected_features.items():
            if key not in feature_buckets:
                continue
            feature_buckets[key]["total"] += 1
            if actual.get(key) == expected_value:
                feature_buckets[key]["match"] += 1

    total_comparisons = sum(bucket["total"] for bucket in feature_buckets.values())
    matched_comparisons = sum(bucket["match"] for bucket in feature_buckets.values())
    core_total_comparisons = sum(
        feature_buckets[key]["total"] for key in CORE_CDT_FEATURE_KEYS
    )
    core_matched_comparisons = sum(
        feature_buckets[key]["match"] for key in CORE_CDT_FEATURE_KEYS
    )
    return {
        "case_total": len(clock_results),
        "exact_case_match_count": exact_case_match_count,
        "exact_case_match_rate": _rate(exact_case_match_count, len(clock_results)),
        "core_comparison_total": core_total_comparisons,
        "core_comparison_match_count": core_matched_comparisons,
        "core_comparison_match_rate": _rate(
            core_matched_comparisons,
            core_total_comparisons,
        ),
        "comparison_total": total_comparisons,
        "comparison_match_count": matched_comparisons,
        "comparison_match_rate": _rate(matched_comparisons, total_comparisons),
        "by_feature": {
            key: {
                **bucket,
                "accuracy": _rate(bucket["match"], bucket["total"]),
            }
            for key, bucket in feature_buckets.items()
        },
        "mismatches": mismatch_rows,
    }


def _build_screening_summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    expected_abnormal = [
        result
        for result in results
        if _is_alert_risk_level(str(result.get("expected_risk_level", "unknown")))
    ]
    expected_normal = [
        result
        for result in results
        if str(result.get("expected_risk_level", "unknown")) == "low"
    ]
    abnormal_detected_count = sum(
        1
        for result in expected_abnormal
        if _is_alert_risk_level(str(result.get("risk_level", "unknown")))
    )
    normal_low_count = sum(
        1 for result in expected_normal if str(result.get("risk_level", "unknown")) == "low"
    )
    false_alarm_count = sum(
        1
        for result in expected_normal
        if _is_alert_risk_level(str(result.get("risk_level", "unknown")))
    )
    alert_count = sum(
        1
        for result in results
        if _is_alert_risk_level(str(result.get("risk_level", "unknown")))
    )
    expected_alert_by_level: dict[str, dict[str, Any]] = {}
    for expected_level in ("medium", "high"):
        level_results = [
            result
            for result in results
            if str(result.get("expected_risk_level", "unknown")) == expected_level
        ]
        detected_count = sum(
            1
            for result in level_results
            if _is_alert_risk_level(str(result.get("risk_level", "unknown")))
        )
        expected_alert_by_level[expected_level] = {
            "total": len(level_results),
            "detected": detected_count,
            "missed": len(level_results) - detected_count,
            "detection_rate": _rate(detected_count, len(level_results)),
        }

    binary_success_count = abnormal_detected_count + normal_low_count
    binary_total = len(expected_abnormal) + len(expected_normal)
    return {
        "expected_abnormal_total": len(expected_abnormal),
        "abnormal_detected_count": abnormal_detected_count,
        "abnormal_detection_rate": _rate(abnormal_detected_count, len(expected_abnormal)),
        "abnormal_missed_count": len(expected_abnormal) - abnormal_detected_count,
        "expected_normal_total": len(expected_normal),
        "normal_low_count": normal_low_count,
        "normal_low_rate": _rate(normal_low_count, len(expected_normal)),
        "false_alarm_count": false_alarm_count,
        "false_alarm_rate": _rate(false_alarm_count, len(expected_normal)),
        "alert_count": alert_count,
        "alert_precision": _rate(abnormal_detected_count, alert_count),
        "binary_success_count": binary_success_count,
        "binary_total": binary_total,
        "binary_success_rate": _rate(binary_success_count, binary_total),
        "by_expected_risk": expected_alert_by_level,
    }


def _is_alert_risk_level(risk_level: str) -> bool:
    return risk_level in ALERT_RISK_LEVELS


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run manual Qwen quality evaluation cases. This script may call real "
            "Qwen APIs and consume tokens."
        )
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dialog", action="store_true", help="Run dialogue cases only.")
    group.add_argument("--clock", action="store_true", help="Run clock image cases only.")
    group.add_argument("--all", action="store_true", help="Run all evaluation cases.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of concurrent case workers. Defaults to 1 for safer API usage.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected cases for quick manual debugging.",
    )
    parser.add_argument(
        "--dialog-cases",
        default=None,
        help=(
            "Dialogue evaluation cases JSON path. Defaults to "
            "demo/fixtures/eval_cases_dialog.json. Path must stay inside project."
        ),
    )
    parser.add_argument(
        "--clock-cases",
        default=None,
        help=(
            "Clock evaluation cases JSON path. Defaults to "
            "demo/fixtures/eval_cases_clock.json. Path must stay inside project."
        ),
    )
    parser.add_argument(
        "--report",
        default=None,
        help=(
            "Markdown report output path. Defaults to docs/qwen-eval-report.md. "
            "Path must stay inside project."
        ),
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be >= 1")
    return args


def _resolve_project_path(value: Optional[str], default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default
    raw_path = Path(value)
    resolved = raw_path.resolve() if raw_path.is_absolute() else (PROJECT_ROOT / raw_path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"path must stay inside project: {value}") from error
    return resolved


def _evaluate_dialog_cases(
    config: AppConfig,
    *,
    workers: int = 1,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    tasks = _build_evaluation_tasks(run_dialog=True, run_clock=False, limit=limit)
    return _run_evaluation_tasks(tasks, config, workers=workers)


def _evaluate_clock_cases(
    config: AppConfig,
    *,
    workers: int = 1,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    tasks = _build_evaluation_tasks(run_dialog=False, run_clock=True, limit=limit)
    return _run_evaluation_tasks(tasks, config, workers=workers)


def _build_evaluation_tasks(
    *,
    run_dialog: bool,
    run_clock: bool,
    dialog_cases_path: Path = DIALOG_CASES_PATH,
    clock_cases_path: Path = CLOCK_CASES_PATH,
    limit: Optional[int] = None,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    if run_dialog:
        for case in _load_cases(dialog_cases_path):
            tasks.append(
                {
                    "order": len(tasks),
                    "kind": "dialog",
                    "case": case,
                    "expected_source": "qwen",
                }
            )
    if run_clock:
        for case in _load_cases(clock_cases_path):
            tasks.append(
                {
                    "order": len(tasks),
                    "kind": "clock",
                    "case": case,
                    "expected_source": "qwen-vl",
                }
            )
    return tasks[:limit] if limit is not None else tasks


def _run_evaluation_tasks(
    tasks: list[dict[str, Any]],
    config: AppConfig,
    *,
    workers: int = 1,
) -> list[dict[str, Any]]:
    if workers <= 1:
        results = []
        for task in tasks:
            result = _evaluate_task(task, config)
            results.append(result)
            _print_case_result(result)
        return results

    results = _run_evaluation_tasks_concurrently(tasks, config, workers=workers)
    results.sort(key=lambda result: result.get("order", 0))
    for result in results:
        _print_case_result(result)
    return results


def _run_evaluation_tasks_concurrently(
    tasks: list[dict[str, Any]],
    config: AppConfig,
    *,
    workers: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(_evaluate_task, task, config): task for task in tasks
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                results.append(future.result())
            except Exception as error:  # pragma: no cover - defensive safety net.
                results.append(_build_error_result(task, error))
    return results


def _evaluate_task(task: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    try:
        kind = task["kind"]
        case = task["case"]
        expected_source = task["expected_source"]
        if kind == "dialog":
            report = _evaluate_dialog_case(case, config)
        elif kind == "clock":
            report = _evaluate_clock_case(case, config)
        else:
            raise ValueError(f"unsupported evaluation kind: {kind}")

        result = _build_case_result(
            kind=kind,
            case=case,
            report=report,
            expected_source=expected_source,
        )
    except Exception as error:
        result = _build_error_result(task, error)

    result["order"] = task.get("order", 0)
    return result


def _evaluate_dialog_case(case: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    messages = _flatten_dialogue_turns(case["dialogue_turns"])
    return evaluate_dialogue(messages, config=config)


def _evaluate_clock_case(case: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    image_path = PROJECT_ROOT / case["image_path"]
    target_time = case.get("target_time", "11:10")
    if image_path.exists() and image_path.is_file():
        return analyze_clock_image(
            image=image_path.read_bytes(),
            filename=image_path.name,
            config=config,
            target_time=target_time,
        )
    return {
        "metadata": {
            "source": "missing_image",
            "model": config.vlm_model.strip() or "未配置",
            "reason": f"image file not found: {case['image_path']}",
        },
        "risk_level": "unknown",
        "domain_scores": {},
    }


def _build_error_result(task: dict[str, Any], error: Exception) -> dict[str, Any]:
    case = task.get("case", {})
    kind = task.get("kind", "unknown")
    domain_scores = {domain: None for domain in COGNITIVE_DOMAINS}
    source = "case_error"
    risk_level = "unknown"
    expected_low_domains = list(case.get("expected_low_domains", []))
    mismatch_summary = build_mismatch_summary(
        source=source,
        risk_level=risk_level,
        domain_scores=domain_scores,
        expected_risk_level=case.get("expected_risk_level", "unknown"),
        expected_low_domains=expected_low_domains,
        expected_source=task.get("expected_source", "unknown"),
    )
    return {
        "order": task.get("order", 0),
        "kind": kind,
        "case_id": case.get("case_id", "unknown_case"),
        "label": case.get("label", ""),
        "source": source,
        "model": "n/a",
        "risk_level": risk_level,
        "domain_scores": domain_scores,
        "expected_risk_level": case.get("expected_risk_level", "unknown"),
        "expected_low_domains": expected_low_domains,
        "mismatch_summary": mismatch_summary,
        "calibrated": False,
        "calibration_notes": [],
        "cdt_features": {},
        "expected_cdt_features": _normalize_expected_cdt_features(
            case.get("expected_cdt_features")
        ),
        "cdt_feature_mismatches": [],
        "notes": case.get("notes", ""),
        "error_type": type(error).__name__,
        "error_message": _safe_error_message(error),
    }


def _build_case_result(
    *,
    kind: str,
    case: dict[str, Any],
    report: dict[str, Any],
    expected_source: str,
) -> dict[str, Any]:
    metadata = report.get("metadata", {})
    source = metadata.get("source", "unknown")
    risk_level = report.get("risk_level", "unknown")
    domain_scores = _normalize_report_scores(report.get("domain_scores", {}))
    cdt_features = report.get("cdt_features", {})
    expected_cdt_features = _normalize_expected_cdt_features(
        case.get("expected_cdt_features")
    )
    expected_low_domains = list(case.get("expected_low_domains", []))
    mismatch_summary = build_mismatch_summary(
        source=source,
        risk_level=risk_level,
        domain_scores=domain_scores,
        expected_risk_level=case["expected_risk_level"],
        expected_low_domains=expected_low_domains,
        expected_source=expected_source,
    )
    return {
        "kind": kind,
        "case_id": case["case_id"],
        "label": case.get("label", ""),
        "source": source,
        "model": metadata.get("model", "未配置"),
        "risk_level": risk_level,
        "domain_scores": domain_scores,
        "expected_risk_level": case["expected_risk_level"],
        "expected_low_domains": expected_low_domains,
        "mismatch_summary": mismatch_summary,
        "calibrated": bool(report.get("calibrated", False)),
        "calibration_notes": list(report.get("calibration_notes", [])),
        "cdt_features": cdt_features,
        "expected_cdt_features": expected_cdt_features,
        "cdt_feature_mismatches": _build_cdt_feature_mismatches(
            cdt_features if isinstance(cdt_features, dict) else {},
            expected_cdt_features,
        ),
        "notes": case.get("notes", ""),
    }


def _load_cases(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError(f"{path.name} must contain a list")
    return cases


def _flatten_dialogue_turns(turns: list[dict[str, str]]) -> list[str]:
    messages: list[str] = []
    for turn in turns:
        assistant = turn.get("assistant", "").strip()
        user = turn.get("user", "").strip()
        if assistant:
            messages.append(f"AI访谈问题：{assistant}")
        if user:
            messages.append(f"老人回答：{user}")
    return messages


def _normalize_report_scores(value: Any) -> dict[str, Optional[float]]:
    scores = value if isinstance(value, dict) else {}
    normalized: dict[str, Optional[float]] = {}
    for domain in COGNITIVE_DOMAINS:
        score = scores.get(domain)
        normalized[domain] = score if isinstance(score, (int, float)) and not isinstance(score, bool) else None
    return normalized


def _normalize_expected_cdt_features(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {}
    return {key: value[key] for key in CDT_FEATURE_KEYS if key in value}


def _build_cdt_feature_mismatches(
    actual_features: dict[str, Any],
    expected_features: dict[str, Any],
) -> list[dict[str, Any]]:
    mismatches = []
    for key, expected_value in expected_features.items():
        actual_value = actual_features.get(key)
        if actual_value != expected_value:
            mismatches.append(
                {
                    "feature": key,
                    "expected": expected_value,
                    "actual": actual_value,
                }
            )
    return mismatches


def _rate(count: int, total: int) -> float:
    return 0.0 if total <= 0 else count / total


def _safe_error_message(error: Exception) -> str:
    message = str(error).strip().replace("\r", " ").replace("\n", " ")
    if not message:
        return type(error).__name__
    message = re.sub(r"sk-[A-Za-z0-9_\-]{6,}", "sk-***", message)
    message = re.sub(
        r"(?i)(api[_-]?key\s*[:=]\s*)[^\s,;]+",
        r"\1***",
        message,
    )
    message = re.sub(r"(?i)(bearer\s+)[A-Za-z0-9._\-]+", r"\1***", message)
    return message[:300]


def _print_config_status(config: AppConfig) -> None:
    print(f"DEMO_MODE={str(config.demo_mode).lower()}")
    print(f"LLM_BASE_URL configured={bool(config.llm_base_url.strip())}")
    print(f"LLM_MODEL={config.llm_model or '未配置'}")
    print(f"LLM_API_KEY configured={bool(config.llm_api_key.strip())}")
    print(f"VLM_BASE_URL configured={bool(config.vlm_base_url.strip())}")
    print(f"VLM_MODEL={config.vlm_model or '未配置'}")
    print(f"VLM_API_KEY configured={bool(config.vlm_api_key.strip())}")


def _print_case_result(result: dict[str, Any]) -> None:
    print("-" * 72)
    print(f"case_id={result['case_id']} kind={result['kind']}")
    print(f"source={result['source']} model={result['model']}")
    print(
        f"risk_level={result['risk_level']} "
        f"expected_risk_level={result['expected_risk_level']}"
    )
    print(
        "domain_scores="
        + json.dumps(result["domain_scores"], ensure_ascii=False, sort_keys=True)
    )
    print(f"expected_low_domains={result['expected_low_domains']}")
    print(f"calibrated={result['calibrated']}")
    print(f"calibration_notes={result['calibration_notes']}")
    if result["kind"] == "clock":
        print(
            "cdt_features="
            + json.dumps(result["cdt_features"], ensure_ascii=False, sort_keys=True)
        )
    print(f"mismatch_summary={result['mismatch_summary']}")
    if result.get("error_type"):
        print(f"case_error={result['error_type']}: {result.get('error_message', '')}")


def _write_report(
    results: list[dict[str, Any]],
    config: AppConfig,
    *,
    report_path: Path = REPORT_PATH,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    summary = build_result_summary(results)
    lines = [
        "# Qwen Effect Evaluation Report",
        "",
        f"Generated at: {datetime.now(timezone.utc).isoformat()}",
        "",
        "This report is generated by a manual script. It may involve real Qwen API calls and token usage.",
        "No API Key or raw model response is written here.",
        "",
        "## Safe Config Status",
        "",
        f"- DEMO_MODE: {str(config.demo_mode).lower()}",
        f"- LLM_BASE_URL configured: {bool(config.llm_base_url.strip())}",
        f"- LLM_MODEL: {config.llm_model or '未配置'}",
        f"- LLM_API_KEY configured: {bool(config.llm_api_key.strip())}",
        f"- VLM_BASE_URL configured: {bool(config.vlm_base_url.strip())}",
        f"- VLM_MODEL: {config.vlm_model or '未配置'}",
        f"- VLM_API_KEY configured: {bool(config.vlm_api_key.strip())}",
        "",
        "## Summary",
        "",
        *_format_summary_markdown(summary),
        "",
        "## Results",
        "",
    ]

    if not results:
        lines.append("No cases were evaluated.")
    for result in results:
        lines.extend(_format_result_markdown(result))

    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _format_summary_markdown(summary: dict[str, Any]) -> list[str]:
    lines = [
        f"- total_cases: `{summary['total']}`",
        (
            f"- exact_rule_match: `{summary['exact_match_count']} / {summary['total']}` "
            f"({_format_percent(summary['exact_match_rate'])})"
        ),
        (
            f"- risk_level_accuracy: `{summary['risk_match_count']} / {summary['total']}` "
            f"({_format_percent(summary['risk_match_rate'])})"
        ),
        (
            f"- real_source_match: `{summary['source_match_count']} / {summary['total']}` "
            f"({_format_percent(summary['source_match_rate'])})"
        ),
        f"- case_errors: `{summary['case_error_count']}`",
        "",
        "### Screening-Oriented Metrics",
        "",
        (
            "- abnormal_detection_rate: "
            f"`{summary['screening']['abnormal_detected_count']} / "
            f"{summary['screening']['expected_abnormal_total']}` "
            f"({_format_percent(summary['screening']['abnormal_detection_rate'])})"
        ),
        (
            "- abnormal_missed_cases: "
            f"`{summary['screening']['abnormal_missed_count']}`"
        ),
        (
            "- normal_low_rate: "
            f"`{summary['screening']['normal_low_count']} / "
            f"{summary['screening']['expected_normal_total']}` "
            f"({_format_percent(summary['screening']['normal_low_rate'])})"
        ),
        (
            "- false_alarm_rate: "
            f"`{summary['screening']['false_alarm_count']} / "
            f"{summary['screening']['expected_normal_total']}` "
            f"({_format_percent(summary['screening']['false_alarm_rate'])})"
        ),
        (
            "- alert_precision: "
            f"`{summary['screening']['abnormal_detected_count']} / "
            f"{summary['screening']['alert_count']}` "
            f"({_format_percent(summary['screening']['alert_precision'])})"
        ),
        (
            "- binary_screening_success: "
            f"`{summary['screening']['binary_success_count']} / "
            f"{summary['screening']['binary_total']}` "
            f"({_format_percent(summary['screening']['binary_success_rate'])})"
        ),
        "",
        (
            "Definition: expected `medium/high` are treated as abnormal signals; "
            "actual `medium/high` are treated as family alerts. Expected `low` "
            "with actual `low` is counted as normal not falsely alerted."
        ),
        "",
        "### Abnormal Detection By Expected Risk",
        "",
        *_format_screening_by_expected_risk(summary["screening"]),
        "",
        "### CDT Feature Metrics",
        "",
        *_format_cdt_feature_summary_markdown(summary["cdt_features"]),
        "",
        "### Risk Confusion Matrix",
        "",
    ]
    confusion_matrix = summary.get("confusion_matrix", [])
    if not confusion_matrix:
        lines.append("- `none`")
    for row in confusion_matrix:
        lines.append(
            f"- expected `{row['expected']}` -> actual `{row['actual']}`: `{row['count']}`"
        )
    lines.extend(["", "### By Label", ""])
    by_label = summary.get("by_label", {})
    if not by_label:
        lines.append("- `none`")
    for label, bucket in by_label.items():
        lines.append(
            f"- {label}: total `{bucket['total']}`, "
            f"risk_accuracy `{_format_percent(bucket['risk_match_rate'])}`, "
            f"exact_match `{_format_percent(bucket['exact_match_rate'])}`"
        )
    return lines


def _format_screening_by_expected_risk(screening: dict[str, Any]) -> list[str]:
    rows = screening.get("by_expected_risk", {})
    lines: list[str] = []
    for expected_level in ("medium", "high"):
        row = rows.get(expected_level, {})
        lines.append(
            f"- expected `{expected_level}`: detected `{row.get('detected', 0)} / "
            f"{row.get('total', 0)}`, missed `{row.get('missed', 0)}`, "
            f"detection `{_format_percent(row.get('detection_rate', 0.0))}`"
        )
    return lines


def _format_cdt_feature_summary_markdown(cdt_summary: dict[str, Any]) -> list[str]:
    case_total = cdt_summary.get("case_total", 0)
    if not case_total:
        return ["- cdt_feature_cases: `0`"]

    lines = [
        (
            "- cdt_core_structure_field_accuracy: "
            f"`{cdt_summary['core_comparison_match_count']} / "
            f"{cdt_summary['core_comparison_total']}` "
            f"({_format_percent(cdt_summary['core_comparison_match_rate'])})"
        ),
        (
            "- cdt_feature_exact_case_match: "
            f"`{cdt_summary['exact_case_match_count']} / {case_total}` "
            f"({_format_percent(cdt_summary['exact_case_match_rate'])})"
        ),
        (
            "- cdt_feature_field_accuracy: "
            f"`{cdt_summary['comparison_match_count']} / "
            f"{cdt_summary['comparison_total']}` "
            f"({_format_percent(cdt_summary['comparison_match_rate'])})"
        ),
        "",
        "| feature | match / total | accuracy |",
        "|---|---:|---:|",
    ]
    for feature, bucket in cdt_summary.get("by_feature", {}).items():
        if bucket.get("total", 0) <= 0:
            continue
        lines.append(
            f"| `{feature}` | `{bucket['match']} / {bucket['total']}` | "
            f"{_format_percent(bucket['accuracy'])} |"
        )

    mismatches = cdt_summary.get("mismatches", [])
    if mismatches:
        lines.extend(["", "Feature mismatches:"])
        for mismatch in mismatches:
            lines.append(
                "- "
                f"`{mismatch.get('case_id')}` `{mismatch.get('feature')}`: "
                f"expected `{mismatch.get('expected')}`, "
                f"actual `{mismatch.get('actual')}`"
            )
    return lines


def _format_result_markdown(result: dict[str, Any]) -> list[str]:
    return [
        f"### {result['case_id']}",
        "",
        f"- kind: `{result['kind']}`",
        f"- label: `{result['label'] or 'n/a'}`",
        f"- source: `{result['source']}`",
        f"- model: `{result['model']}`",
        f"- risk_level: `{result['risk_level']}`",
        f"- expected_risk_level: `{result['expected_risk_level']}`",
        f"- expected_low_domains: `{', '.join(result['expected_low_domains']) or 'none'}`",
        f"- calibrated: `{str(result['calibrated']).lower()}`",
        f"- calibration_notes: `{', '.join(result['calibration_notes']) or 'none'}`",
        *(_format_cdt_features(result["cdt_features"]) if result["kind"] == "clock" else []),
        *(
            _format_expected_cdt_features(result.get("expected_cdt_features"))
            if result["kind"] == "clock"
            else []
        ),
        *(
            _format_cdt_feature_mismatches(result.get("cdt_feature_mismatches"))
            if result["kind"] == "clock"
            else []
        ),
        "- domain_scores:",
        *[
            f"  - {domain}: {_format_score(score)}"
            for domain, score in result["domain_scores"].items()
        ],
        f"- mismatch_summary: `{', '.join(result['mismatch_summary'])}`",
        *_format_case_error(result),
        f"- notes: {result['notes']}",
        "",
    ]


def _format_score(score: Optional[float]) -> str:
    return "null" if score is None else f"{score:.2f}"


def _format_percent(value: float) -> str:
    return f"{value * 100:.1f}%"


def _format_cdt_features(features: Any) -> list[str]:
    if not isinstance(features, dict) or not features:
        return ["- cdt_features: `none`"]
    lines = ["- cdt_features:"]
    for key, value in features.items():
        lines.append(f"  - {key}: `{value}`")
    return lines


def _format_expected_cdt_features(features: Any) -> list[str]:
    if not isinstance(features, dict) or not features:
        return ["- expected_cdt_features: `none`"]
    lines = ["- expected_cdt_features:"]
    for key, value in features.items():
        lines.append(f"  - {key}: `{value}`")
    return lines


def _format_cdt_feature_mismatches(mismatches: Any) -> list[str]:
    if not isinstance(mismatches, list) or not mismatches:
        return ["- cdt_feature_mismatches: `none`"]
    lines = ["- cdt_feature_mismatches:"]
    for mismatch in mismatches:
        if not isinstance(mismatch, dict):
            continue
        lines.append(
            "  - "
            f"{mismatch.get('feature')}: expected `{mismatch.get('expected')}`, "
            f"actual `{mismatch.get('actual')}`"
        )
    return lines


def _format_case_error(result: dict[str, Any]) -> list[str]:
    if not result.get("error_type"):
        return []
    return [
        f"- error_type: `{result['error_type']}`",
        f"- error_message: {result.get('error_message', '')}",
    ]


if __name__ == "__main__":
    raise SystemExit(main())
