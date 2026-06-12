from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Union

from core.db import get_recent_sessions
from core.mock_data import load_fixture_sessions
from core.schemas import COGNITIVE_DOMAINS, RISK_ORDER, empty_domain_scores


CURRENT_USER_ID = "zhang-nainai"
CURRENT_USER_DISPLAY_NAME = "张奶奶"
CURRENT_USER_USERNAME = "zhang"
CURRENT_USER_PROFILE_TYPE = "elder_demo"
SESSION_USER_KEY = "current_demo_user"
HISTORY_FOCUS_THRESHOLD = 0.78
HISTORY_FOCUS_PRIORITY = (
    "memory",
    "visuospatial",
    "executive_function",
    "attention",
    "language",
    "orientation",
)

HISTORY_FOCUS_QUESTIONS = {
    "orientation": "欢迎回来，{display_name}。小顾今天继续陪您慢慢聊。您知道今天是星期几吗？",
    "memory": "欢迎回来，{display_name}。小顾今天继续陪您慢慢聊。我们先从一件日常小事开始：您还记得今天早上吃了什么吗？",
    "language": "欢迎回来，{display_name}。小顾今天继续陪您慢慢聊。您能用一句话说说身边看到的一样东西吗？",
    "executive_function": "欢迎回来，{display_name}。小顾今天继续陪您慢慢聊。如果一会儿要出门散步，您会先准备什么？",
    "attention": "欢迎回来，{display_name}。小顾今天继续陪您慢慢聊。我们做个很短的小练习，您可以从 20 往回数三个数吗？",
    "visuospatial": "欢迎回来，{display_name}。小顾今天继续陪您慢慢聊。从客厅走到厨房，通常会经过哪里？",
}

DEMO_TRAJECTORY_USERS = {
    "normal": "demo-person-normal",
    "mild_decline": "demo-person-mild-decline",
    "fluctuating": "demo-person-fluctuating",
}


def default_user_profile() -> dict[str, Any]:
    return {
        "user_id": CURRENT_USER_ID,
        "username": CURRENT_USER_USERNAME,
        "display_name": CURRENT_USER_DISPLAY_NAME,
        "profile_type": CURRENT_USER_PROFILE_TYPE,
        "is_authenticated": False,
    }


def normalize_user_profile(
    value: Optional[Mapping[str, Any]],
    *,
    is_authenticated: Optional[bool] = None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        return default_user_profile()

    fallback = default_user_profile()
    user_id = _clean_text(value.get("user_id")) or fallback["user_id"]
    display_name = _clean_text(value.get("display_name")) or fallback["display_name"]
    username = _clean_text(value.get("username")) or user_id
    profile_type = _clean_text(value.get("profile_type")) or fallback["profile_type"]
    authenticated = (
        bool(value.get("is_authenticated"))
        if is_authenticated is None
        else bool(is_authenticated)
    )
    return {
        "user_id": user_id,
        "username": username,
        "display_name": display_name,
        "profile_type": profile_type,
        "is_authenticated": authenticated,
    }


def get_current_user_profile(
    session_state: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    if isinstance(session_state, Mapping):
        profile = session_state.get(SESSION_USER_KEY)
        if isinstance(profile, Mapping):
            return normalize_user_profile(profile)
    return default_user_profile()


def store_current_user_profile(
    session_state: MutableMapping[str, Any],
    user: Mapping[str, Any],
) -> dict[str, Any]:
    profile = normalize_user_profile(user, is_authenticated=True)
    session_state[SESSION_USER_KEY] = profile
    return profile


def clear_current_user_profile(session_state: MutableMapping[str, Any]) -> None:
    session_state.pop(SESSION_USER_KEY, None)


def load_current_user_sessions(
    db_path: Optional[Union[str, Path]] = None,
    limit: int = 10,
    user_profile: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    profile = normalize_user_profile(user_profile)
    try:
        sessions = get_recent_sessions(
            user_id=profile["user_id"],
            limit=limit,
            db_path=db_path,
        )
    except Exception:
        sessions = []

    return {
        "sessions": sessions,
        "source": "sqlite_current_user" if sessions else "empty_current_user",
        "user_id": profile["user_id"],
        "display_name": profile["display_name"],
        "profile_type": profile["profile_type"],
    }


def infer_history_focus_domain(
    sessions: list[dict[str, Any]],
    threshold: float = HISTORY_FOCUS_THRESHOLD,
) -> Optional[str]:
    domain_values: dict[str, list[float]] = {domain: [] for domain in COGNITIVE_DOMAINS}
    for session in sessions:
        if not isinstance(session, dict):
            continue
        scores = session.get("domain_scores")
        if not isinstance(scores, dict):
            continue
        for domain in COGNITIVE_DOMAINS:
            value = scores.get(domain)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            domain_values[domain].append(float(value))

    candidates: list[tuple[float, int, str]] = []
    for domain, values in domain_values.items():
        if not values:
            continue
        average_score = sum(values) / len(values)
        if average_score <= threshold:
            priority = HISTORY_FOCUS_PRIORITY.index(domain)
            candidates.append((average_score, priority, domain))

    if not candidates:
        return None
    return min(candidates)[2]


def build_history_personalized_start(
    sessions: list[dict[str, Any]],
    *,
    display_name: str,
    fallback_question: str,
    fallback_domain: str,
) -> dict[str, Any]:
    safe_display_name = _clean_text(display_name) or CURRENT_USER_DISPLAY_NAME
    if not sessions:
        return {
            "has_history": False,
            "target_domain": fallback_domain,
            "question": fallback_question,
            "elder_hint": "您好，我是小顾，今天我陪您轻松聊一会儿。",
            "reason": "暂无历史记录，使用第一轮通用定向问题。",
        }

    focus_domain = infer_history_focus_domain(sessions) or fallback_domain
    template = HISTORY_FOCUS_QUESTIONS.get(
        focus_domain,
        HISTORY_FOCUS_QUESTIONS["orientation"],
    )
    return {
        "has_history": True,
        "target_domain": focus_domain,
        "question": template.format(display_name=safe_display_name),
        "elder_hint": f"欢迎回来，{safe_display_name}。小顾今天继续陪您慢慢聊。",
        "reason": (
            f"根据最近记录优先覆盖 {focus_domain}。"
            if focus_domain != fallback_domain
            else "有历史记录，但未发现明显偏低认知域，使用通用定向问题。"
        ),
    }


def find_assessment_record(
    assessment_id: Optional[str],
    user_id: str = CURRENT_USER_ID,
    db_path: Optional[Union[str, Path]] = None,
    limit: int = 20,
) -> Optional[dict[str, Any]]:
    if not isinstance(assessment_id, str) or not assessment_id.strip():
        return None
    try:
        sessions = get_recent_sessions(user_id=user_id, limit=limit, db_path=db_path)
    except Exception:
        return None
    for session in sessions:
        if session.get("assessment_id") == assessment_id or session.get("session_id") == assessment_id:
            return session
    return None


def load_sessions_for_brief(
    trajectory: str,
    db_path: Optional[Union[str, Path]] = None,
    limit: int = 3,
) -> dict[str, Any]:
    user_id = DEMO_TRAJECTORY_USERS.get(trajectory)
    if user_id is None:
        raise ValueError(f"Unknown demo trajectory: {trajectory}")

    try:
        sessions = get_recent_sessions(user_id=user_id, limit=limit, db_path=db_path)
    except Exception:
        sessions = []

    if sessions:
        return {
            "sessions": sessions,
            "source": "sqlite",
            "user_id": user_id,
        }

    return {
        "sessions": load_fixture_sessions(trajectory),
        "source": "fixtures",
        "user_id": user_id,
    }


def build_dialog_session_record(
    report: dict[str, Any],
    user_id: str = CURRENT_USER_ID,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    timestamp = created_at or _now_iso()
    normalized_user_id = user_id.strip() or CURRENT_USER_ID
    record = deepcopy(report)

    record["session_id"] = _dialog_session_id(timestamp)
    record["participant_id"] = normalized_user_id
    record["user_id"] = normalized_user_id
    record["created_at"] = timestamp
    record["is_mock"] = True
    record.setdefault("trajectory", "normal")
    return record


def build_dialog_assessment_record(
    report: dict[str, Any],
    user_id: str = CURRENT_USER_ID,
    assessment_id: Optional[str] = None,
    existing_record: Optional[dict[str, Any]] = None,
    created_at: Optional[str] = None,
) -> dict[str, Any]:
    timestamp = created_at or _now_iso()
    normalized_user_id = user_id.strip() or CURRENT_USER_ID
    resolved_assessment_id = _resolve_assessment_id(
        timestamp,
        assessment_id=assessment_id,
        existing_record=existing_record,
    )
    return _build_assessment_record(
        user_id=normalized_user_id,
        assessment_id=resolved_assessment_id,
        created_at=timestamp,
        dialogue_result=report,
        clock_result=None,
        existing_record=existing_record,
    )


def build_clock_session_record(
    report: dict[str, Any],
    user_id: str = CURRENT_USER_ID,
    created_at: Optional[str] = None,
    target_time: str = "11:10",
) -> dict[str, Any]:
    timestamp = created_at or _now_iso()
    normalized_user_id = user_id.strip() or CURRENT_USER_ID
    record = deepcopy(report)
    metadata = report.get("metadata", {}) if isinstance(report.get("metadata"), dict) else {}

    record["session_id"] = _clock_session_id(timestamp)
    record["participant_id"] = normalized_user_id
    record["user_id"] = normalized_user_id
    record["created_at"] = timestamp
    record.setdefault("trajectory", "normal")
    record["clock_result"] = {
        "target_time": target_time.strip() or "11:10",
        "source": metadata.get("source", "unknown"),
        "model": metadata.get("model", "未配置"),
        "clock_findings": deepcopy(report.get("clock_findings", {})),
        "cdt_features": deepcopy(report.get("cdt_features", {})),
        "metadata": deepcopy(metadata),
    }
    return record


def build_clock_assessment_record(
    report: dict[str, Any],
    user_id: str = CURRENT_USER_ID,
    assessment_id: Optional[str] = None,
    existing_record: Optional[dict[str, Any]] = None,
    created_at: Optional[str] = None,
    target_time: str = "11:10",
) -> dict[str, Any]:
    timestamp = created_at or _now_iso()
    normalized_user_id = user_id.strip() or CURRENT_USER_ID
    resolved_assessment_id = _resolve_assessment_id(
        timestamp,
        assessment_id=assessment_id,
        existing_record=existing_record,
    )
    return _build_assessment_record(
        user_id=normalized_user_id,
        assessment_id=resolved_assessment_id,
        created_at=timestamp,
        dialogue_result=None,
        clock_result=_clock_result_payload(report, target_time=target_time),
        existing_record=existing_record,
    )


def _now_iso() -> str:
    china_tz = timezone(timedelta(hours=8))
    return datetime.now(china_tz).isoformat(timespec="milliseconds")


def _dialog_session_id(created_at: str) -> str:
    digits = "".join(character for character in created_at if character.isdigit())
    return f"dialog-{digits or 'session'}"


def _clock_session_id(created_at: str) -> str:
    digits = "".join(character for character in created_at if character.isdigit())
    return f"clock-{digits or 'session'}"


def _assessment_session_id(created_at: str) -> str:
    digits = "".join(character for character in created_at if character.isdigit())
    return f"assessment-{digits or 'session'}"


def _resolve_assessment_id(
    created_at: str,
    assessment_id: Optional[str],
    existing_record: Optional[dict[str, Any]],
) -> str:
    if isinstance(assessment_id, str) and assessment_id.strip():
        return assessment_id.strip()
    if isinstance(existing_record, dict):
        existing_id = existing_record.get("assessment_id") or existing_record.get("session_id")
        if isinstance(existing_id, str) and existing_id.strip():
            return existing_id.strip()
    return _assessment_session_id(created_at)


def _build_assessment_record(
    *,
    user_id: str,
    assessment_id: str,
    created_at: str,
    dialogue_result: Optional[dict[str, Any]],
    clock_result: Optional[dict[str, Any]],
    existing_record: Optional[dict[str, Any]],
) -> dict[str, Any]:
    existing = deepcopy(existing_record) if isinstance(existing_record, dict) else {}
    existing_dialogue = existing.get("dialogue_result")
    existing_clock = existing.get("clock_result")
    final_dialogue = deepcopy(dialogue_result) if dialogue_result is not None else deepcopy(existing_dialogue)
    final_clock = deepcopy(clock_result) if clock_result is not None else deepcopy(existing_clock)

    components: list[str] = []
    if isinstance(final_dialogue, dict):
        components.append("dialogue")
    if isinstance(final_clock, dict):
        components.append("clock")

    domain_scores = _merge_domain_scores(final_dialogue, final_clock)
    evidence = _merge_evidence(final_dialogue, final_clock)
    risk_level = _merge_risk_level(final_dialogue, final_clock)
    explanation = _merge_explanation(final_dialogue, final_clock)

    record = {
        "session_id": assessment_id,
        "assessment_id": assessment_id,
        "participant_id": user_id,
        "user_id": user_id,
        "created_at": created_at,
        "trajectory": existing.get("trajectory", "normal"),
        "components": components,
        "domain_scores": domain_scores,
        "evidence": evidence,
        "risk_level": risk_level,
        "explanation": explanation,
        "disclaimer": _first_text(
            _nested_get(final_dialogue, "disclaimer"),
            _nested_get(final_clock, "disclaimer"),
            existing.get("disclaimer"),
            "",
        ),
        "is_mock": bool(
            _nested_get(final_dialogue, "is_mock") or _nested_get(final_clock, "is_mock")
        ),
    }
    if isinstance(final_dialogue, dict):
        record["dialogue_result"] = final_dialogue
    if isinstance(final_clock, dict):
        record["clock_result"] = final_clock
        record["cdt_features"] = deepcopy(final_clock.get("cdt_features", {}))
    return record


def _clock_result_payload(report: dict[str, Any], target_time: str) -> dict[str, Any]:
    metadata = report.get("metadata", {}) if isinstance(report.get("metadata"), dict) else {}
    payload = deepcopy(report)
    payload.update(
        {
            "target_time": target_time.strip() or "11:10",
            "source": metadata.get("source", "unknown"),
            "model": metadata.get("model", "未配置"),
            "clock_findings": deepcopy(report.get("clock_findings", {})),
            "cdt_features": deepcopy(report.get("cdt_features", {})),
            "metadata": deepcopy(metadata),
        }
    )
    return payload


def _merge_domain_scores(
    dialogue_result: Optional[dict[str, Any]],
    clock_result: Optional[dict[str, Any]],
) -> dict[str, Optional[float]]:
    merged = empty_domain_scores()
    for result in (dialogue_result, clock_result):
        if not isinstance(result, dict):
            continue
        scores = result.get("domain_scores")
        if not isinstance(scores, dict):
            continue
        for domain in COGNITIVE_DOMAINS:
            value = scores.get(domain)
            if value is None:
                continue
            try:
                numeric_value = float(value)
            except (TypeError, ValueError):
                continue
            if merged[domain] is None:
                merged[domain] = numeric_value
            else:
                merged[domain] = min(float(merged[domain]), numeric_value)
    return merged


def _merge_evidence(
    dialogue_result: Optional[dict[str, Any]],
    clock_result: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for result in (dialogue_result, clock_result):
        if not isinstance(result, dict):
            continue
        items = result.get("evidence")
        if isinstance(items, list):
            evidence.extend(deepcopy(items))
    return evidence


def _merge_risk_level(
    dialogue_result: Optional[dict[str, Any]],
    clock_result: Optional[dict[str, Any]],
) -> str:
    levels = []
    for result in (dialogue_result, clock_result):
        if isinstance(result, dict) and result.get("risk_level") in RISK_ORDER:
            levels.append(result["risk_level"])
    if not levels:
        return "unknown"
    return max(levels, key=lambda level: RISK_ORDER[level])


def _merge_explanation(
    dialogue_result: Optional[dict[str, Any]],
    clock_result: Optional[dict[str, Any]],
) -> str:
    parts: list[str] = []
    if isinstance(dialogue_result, dict):
        text = dialogue_result.get("explanation")
        if isinstance(text, str) and text.strip():
            parts.append(f"对话评估：{text.strip()}")
    if isinstance(clock_result, dict):
        text = clock_result.get("explanation")
        if isinstance(text, str) and text.strip():
            parts.append(f"画钟测试：{text.strip()}")
    if parts:
        prefix = "本次综合评估包含对话评估和画钟测试。" if len(parts) == 2 else ""
        return prefix + " ".join(parts)
    return "本次记录暂无可用解释。"


def _nested_get(value: Any, key: str) -> Any:
    return value.get(key) if isinstance(value, dict) else None


def _first_text(*values: Any) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    from core.schemas import DISCLAIMER

    return DISCLAIMER


def _clean_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) and value.strip() else ""
