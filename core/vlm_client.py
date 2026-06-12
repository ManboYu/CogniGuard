from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any, Optional

from core.config import AppConfig, load_config
from core.report import generate_mock_clock_report
from core.schemas import (
    DISCLAIMER,
    calibrate_clock_result,
    fallback_result,
    normalize_clock_assessment_payload,
    validate_clock_assessment,
)


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
VLM_TEMPERATURE = 0.1
VLM_MAX_TOKENS = 900
VLM_TIMEOUT_SECONDS = 30.0
DEFAULT_CLOCK_EVAL_PROMPT = (
    "Return only a valid JSON object. Do not output Markdown, code fences, "
    "or extra explanation. The JSON object must include domain_scores, "
    "evidence, risk_level, explanation, and disclaimer."
)


def analyze_clock_image(
    image: Optional[Any] = None,
    filename: Optional[str] = None,
    config: Optional[AppConfig] = None,
    target_time: str = "11:10",
) -> dict[str, Any]:
    active_config = config or load_config()
    inferred_name = filename or getattr(image, "name", None)

    if active_config.demo_mode:
        return _mock_with_metadata(
            inferred_name,
            active_config,
            reason="DEMO_MODE=true",
        )

    if not _should_use_real_vlm(active_config):
        return _mock_with_metadata(
            inferred_name,
            active_config,
            reason="VLM 配置不完整",
        )

    try:
        data_url = _image_to_data_url(image, filename=inferred_name)
    except ValueError:
        return _mock_with_metadata(
            inferred_name,
            active_config,
            reason="未提供图片，使用 mock 画钟结果",
        )

    try:
        content = _request_clock_evaluation(data_url, active_config, target_time)
    except Exception:
        return _fallback_with_metadata(
            active_config,
            reason="api_error: VLM 调用失败，已回退到安全结果",
        )

    try:
        payload = _parse_json_object(content)
    except json.JSONDecodeError:
        return _fallback_with_metadata(
            active_config,
            reason="json_error: 模型返回内容不是有效 JSON",
        )
    except ValueError:
        return _fallback_with_metadata(
            active_config,
            reason="json_error: 模型返回 JSON 不是对象",
        )

    try:
        normalized = normalize_clock_assessment_payload(payload)
        return _with_metadata(
            calibrate_clock_result(validate_clock_assessment(normalized)),
            source="qwen-vl",
            model=active_config.vlm_model,
        )
    except ValueError as error:
        return _fallback_with_metadata(
            active_config,
            reason="schema_error: 模型返回 JSON 未通过 schema 校验",
            validation_errors=[_safe_error_text(error, active_config)],
        )


def _should_use_real_vlm(config: AppConfig) -> bool:
    return (
        not config.demo_mode
        and bool(config.vlm_base_url.strip())
        and bool(config.vlm_api_key.strip())
        and bool(config.vlm_model.strip())
    )


def _request_clock_evaluation(
    data_url: str,
    config: AppConfig,
    target_time: str,
) -> str:
    client = _create_openai_client(config)
    response = client.chat.completions.create(
        model=config.vlm_model,
        messages=_build_clock_messages(data_url, target_time),
        temperature=VLM_TEMPERATURE,
        max_tokens=VLM_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return _extract_message_content(response)


def _create_openai_client(config: AppConfig) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=config.vlm_api_key,
        base_url=config.vlm_base_url,
        timeout=VLM_TIMEOUT_SECONDS,
    )


def _build_clock_messages(data_url: str, target_time: str = "11:10") -> list[dict[str, Any]]:
    prompt = _read_prompt("clock_eval.md")
    clean_target_time = target_time.strip() or "11:10"
    return [
        {
            "role": "system",
            "content": prompt,
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "Analyze this clock drawing image. Return only a valid "
                        "JSON object following the schema in the system prompt. "
                        f"The target_time is {clean_target_time}."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                },
            ],
        },
    ]


def _read_prompt(filename: str) -> str:
    try:
        prompt = (PROMPT_DIR / filename).read_text(encoding="utf-8").strip()
    except OSError:
        return DEFAULT_CLOCK_EVAL_PROMPT
    return prompt or DEFAULT_CLOCK_EVAL_PROMPT


def _image_to_data_url(image: Optional[Any], filename: Optional[str] = None) -> str:
    image_bytes = _read_image_bytes(image)
    if not image_bytes:
        raise ValueError("image bytes are required")

    mime_type = _detect_mime_type(image_bytes, image=image, filename=filename)
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _read_image_bytes(image: Optional[Any]) -> bytes:
    if image is None:
        return b""
    if isinstance(image, bytes):
        return image
    if isinstance(image, bytearray):
        return bytes(image)
    if hasattr(image, "getvalue"):
        value = image.getvalue()
        return bytes(value) if isinstance(value, (bytes, bytearray)) else b""
    if hasattr(image, "read"):
        value = image.read()
        return bytes(value) if isinstance(value, (bytes, bytearray)) else b""
    return b""


def _detect_mime_type(
    image_bytes: bytes,
    image: Optional[Any] = None,
    filename: Optional[str] = None,
) -> str:
    uploaded_type = getattr(image, "type", None)
    if uploaded_type in {"image/png", "image/jpeg"}:
        return uploaded_type

    lower_name = (filename or getattr(image, "name", "") or "").lower()
    if lower_name.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower_name.endswith(".png"):
        return "image/png"

    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    return "image/png"


def _parse_json_object(content: str) -> dict[str, Any]:
    candidate = _strip_json_fence(content)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(candidate[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("VLM response JSON must be an object")
    return parsed


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _extract_message_content(response: Any) -> str:
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("VLM response content is empty")
    return content.strip()


def _mock_with_metadata(
    filename: Optional[str],
    config: AppConfig,
    reason: str,
) -> dict[str, Any]:
    report = generate_mock_clock_report(filename)
    report.setdefault("domain_scores", _mock_clock_scores())
    report.setdefault("evidence", _clock_findings_to_evidence(report))
    report.setdefault("explanation", "本次使用 mock 画钟分析结果，仅用于演示。")
    return _with_metadata(
        report,
        source="mock",
        model=config.vlm_model,
        reason=reason,
    )


def _fallback_with_metadata(
    config: AppConfig,
    reason: str,
    validation_errors: Optional[list[str]] = None,
) -> dict[str, Any]:
    report = fallback_result()
    report["clock_findings"] = {
        "number_placement": "未生成可靠分析。",
        "hand_accuracy": "未生成可靠分析。",
        "visuospatial_evidence": [],
    }
    report["cdt_features"] = {
        "numbers_complete": None,
        "number_order_correct": None,
        "number_spacing": "unknown",
        "number_distribution": "unknown",
        "hands_present": None,
        "target_time_match": None,
        "center_anchor_clear": None,
    }
    return _with_metadata(
        report,
        source="fallback",
        model=config.vlm_model,
        reason=reason,
        validation_errors=validation_errors or [],
    )


def _with_metadata(
    report: dict[str, Any],
    source: str,
    model: str = "",
    reason: str = "",
    validation_errors: Optional[list[str]] = None,
) -> dict[str, Any]:
    metadata = {
        "source": source,
        "model": model.strip() or "未配置",
        "validation_errors": validation_errors or [],
    }
    if reason:
        metadata["reason"] = reason
    report["metadata"] = metadata
    return report


def _mock_clock_scores() -> dict[str, Optional[float]]:
    return {
        "orientation": None,
        "memory": None,
        "language": None,
        "executive_function": 0.68,
        "attention": None,
        "visuospatial": 0.62,
    }


def _clock_findings_to_evidence(report: dict[str, Any]) -> list[dict[str, str]]:
    findings = report.get("clock_findings") or {}
    evidence = findings.get("visuospatial_evidence") or []
    return [
        {
            "domain": "visuospatial",
            "source": "clock",
            "text": item,
        }
        for item in evidence
        if isinstance(item, str) and item.strip()
    ]


def _safe_error_text(error: Exception, config: AppConfig) -> str:
    message = str(error)
    api_key = config.vlm_api_key.strip()
    if api_key:
        message = message.replace(api_key, "[redacted-api-key]")
    return message
