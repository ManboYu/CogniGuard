import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from html import escape
import json
from pathlib import Path
import time
from typing import Any, Optional

import streamlit as st
import streamlit.components.v1 as components

from core.config import load_config
from core.db import save_session
from core.llm_client import evaluate_dialogue
from core.memory import save_session_memory
from core.mock_data import (
    CLASSROOM_DEMO_LEVEL_LABELS,
    build_classroom_clock_report,
    build_classroom_demo_interview,
)
from core.report import (
    compute_clock_structure_score,
    compute_cogniguard_score,
    compute_dialogue_score,
    format_session_time,
)
from core.schemas import (
    DISCLAIMER,
    DOMAIN_LABELS,
    display_cdt_feature_value,
    display_risk_level,
    display_source,
)
from core.session_history import (
    CURRENT_USER_DISPLAY_NAME,
    CURRENT_USER_ID,
    build_clock_assessment_record,
    build_dialog_assessment_record,
)
from core.staff_gate import hide_sidebar_nav, render_staff_gate
from core.tts_client import synthesize_speech
from core.ui import (
    callout_html,
    chip_html,
    evidence_card_html,
    inject_staff_theme,
    metric_card_html,
    page_brand_header_html,
    risk_badge_html,
    section_header_html,
    status_strip_html,
    timeline_item_html,
)


DEFAULT_ASSISTANT_VOICE = "Cherry"
DEFAULT_PATIENT_DEMO_MODEL = "cosyvoice-v3-flash"
DEFAULT_PATIENT_DEMO_VOICE = "longlaoyi_v3"
DEMO_TTS_MAX_ATTEMPTS = 2
DEMO_TTS_RETRY_DELAY_SECONDS = 0.75
DEMO_TTS_MAX_WORKERS = 7
DEMO_REPORT_CACHE_TTL_SECONDS = 24 * 60 * 60
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASSROOM_CLOCK_IMAGE_DIR = PROJECT_ROOT / "assets" / "classroom_clock_samples"
CLASSROOM_DEMO_AUDIO_DIR = PROJECT_ROOT / "assets" / "classroom_demo_audio"
CLASSROOM_DEMO_AUDIO_EXTENSIONS = (".wav", ".mp3", ".m4a", ".ogg")
CLASSROOM_CLOCK_IMAGE_COLUMN_RATIOS = (0.22, 0.56, 0.22)
CLASSROOM_CLOCK_IMAGES = {
    "正常表现": {
        "filename": "classroom_clock_normal.png",
        "caption": "演示示意：回答稳定时的正常画钟照片。",
    },
    "轻度下降": {
        "filename": "classroom_clock_mild_decline.png",
        "caption": "演示示意：轻度下降场景中，画钟数字位置和指针略有不稳定。",
    },
    "明显异常": {
        "filename": "classroom_clock_obvious_issue.png",
        "caption": "演示示意：明显异常场景中，画钟布局和指针出现较多问题。",
    },
}
CLASSROOM_SCENARIO_COPY = {
    "正常表现": {
        "summary": "回答清晰、能按生活经验组织信息，演示中不会自动触发画钟。",
        "trigger": "访谈后判断",
        "tone": "green",
    },
    "轻度下降": {
        "summary": "保留生活化回答，但出现记忆模糊和路线不确定，用来展示对话后触发画钟。",
        "trigger": "访谈后判断",
        "tone": "amber",
    },
    "明显异常": {
        "summary": "多轮回答出现明显不确定和步骤困难，用来展示访谈结束后进入画钟。",
        "trigger": "访谈后判断",
        "tone": "terracotta",
    },
}


st.set_page_config(page_title="演示模式", layout="wide")
config = load_config()
hide_sidebar_nav()
inject_staff_theme()
render_staff_gate(config)

st.markdown(
    page_brand_header_html(
        "演示模式",
        eyebrow="CogniGuard Demo Stage",
        body="用于快速演示和展示，快速展示小顾访谈、动态下一步、语音播放和报告生成。",
        meta="模拟演示数据，不是真实老人输入。",
    ),
    unsafe_allow_html=True,
)
st.warning("模拟演示数据，不是真实老人输入。本页仅用于展示，不代表真实患者数据。")
st.caption("本页用于演示，老人回答由系统预设生成；音频为 TTS 合成，不代表真实老人输入。")
st.caption("小顾声音和老人声音使用不同音色，便于演示区分。模拟回答文本会直接进入评估流程，不会伪装成真实 ASR 结果。")
st.markdown(
    section_header_html(
        "路演模式舞台",
        eyebrow="Classroom Demo",
        body="选择三类预设输入后，按生成流程、生成语音、播放演示、运行评估的顺序展示完整闭环。",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    status_strip_html(
        [
            {"label": "数据性质", "value": "模拟演示数据", "tone": "amber"},
            {"label": "输入来源", "value": "预设文本 + TTS", "tone": "blue"},
            {"label": "评估路径", "value": "对话 → 动态画钟 → 报告", "tone": "green"},
        ]
    ),
    unsafe_allow_html=True,
)


def _init_state() -> None:
    defaults = {
        "classroom_demo_level": CLASSROOM_DEMO_LEVEL_LABELS[0],
        "classroom_demo_turns": [],
        "classroom_tts_results": {},
        "classroom_tts_status": "",
        "classroom_report": None,
        "classroom_report_cache_key": None,
        "classroom_report_status": "",
        "classroom_show_report": False,
        "classroom_save_status": "",
        "classroom_show_player": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _generate_demo_flow(level: str) -> None:
    st.session_state.classroom_demo_level = level
    st.session_state.classroom_demo_turns = build_classroom_demo_interview(
        level,
        display_name=CURRENT_USER_DISPLAY_NAME,
    )
    st.session_state.classroom_tts_results = {}
    st.session_state.classroom_tts_status = "已生成完整演示流程，可继续生成全部演示语音。"
    st.session_state.classroom_report = None
    st.session_state.classroom_report_cache_key = None
    st.session_state.classroom_report_status = ""
    st.session_state.classroom_show_report = False
    st.session_state.classroom_save_status = ""
    st.session_state.classroom_show_player = False


def _flatten_demo_turns(turns: list[dict[str, Any]]) -> list[str]:
    messages = []
    for turn in turns:
        question = str(turn.get("system_question", "")).strip()
        answer = str(turn.get("patient_answer", "")).strip()
        if question:
            messages.append(f"AI访谈问题：{question}")
        if answer:
            messages.append(f"老人回答：{answer}")
    return messages


@st.cache_data(show_spinner=False, ttl=DEMO_REPORT_CACHE_TTL_SECONDS)
def _cached_classroom_dialogue_report(
    level: str,
    messages: tuple[str, ...],
    demo_mode: bool,
    llm_base_url: str,
    llm_api_key_configured: bool,
    llm_model: str,
) -> dict[str, Any]:
    _ = (level, demo_mode, llm_base_url, llm_api_key_configured, llm_model)
    return evaluate_dialogue(list(messages), config=load_config())


def _report_cache_key(level: str, turns: list[dict[str, Any]]) -> tuple[str, tuple[str, ...]]:
    return level, tuple(_flatten_demo_turns(turns))


def _ensure_demo_report_cache(level: str, turns: list[dict[str, Any]]) -> None:
    if not turns:
        st.session_state.classroom_report_status = "请先生成完整演示流程。"
        return

    cache_key = _report_cache_key(level, turns)
    if (
        st.session_state.classroom_report_cache_key == cache_key
        and isinstance(st.session_state.classroom_report, dict)
    ):
        return

    dialogue_report = deepcopy(
        _cached_classroom_dialogue_report(
            level,
            cache_key[1],
            bool(config.demo_mode),
            config.llm_base_url.strip(),
            bool(config.llm_api_key.strip()),
            config.llm_model.strip(),
        )
    )
    dialogue_report["is_mock"] = True
    dialogue_report["is_simulated"] = True
    dialogue_report["classroom_demo_level"] = level
    dialogue_report["demo_note"] = "模拟演示数据，不是真实老人输入。"

    clock_report = build_classroom_clock_report(level, model=config.vlm_model)
    report = _build_complete_demo_report(level, dialogue_report, clock_report)

    st.session_state.classroom_report = report
    st.session_state.classroom_report_cache_key = cache_key
    st.session_state.classroom_report_status = (
        "完整报告缓存已准备好：包含对话评估和画钟分析；"
        f"对话来源：{display_source(dialogue_report.get('metadata', {}).get('source', 'unknown'))}。"
    )


def _build_complete_demo_report(
    level: str,
    dialogue_report: dict[str, Any],
    clock_report: dict[str, Any],
) -> dict[str, Any]:
    dialogue_record = build_dialog_assessment_record(
        dialogue_report,
        user_id=CURRENT_USER_ID,
    )
    complete_record = build_clock_assessment_record(
        clock_report,
        user_id=CURRENT_USER_ID,
        assessment_id=dialogue_record["assessment_id"],
        existing_record=dialogue_record,
    )
    complete_record["is_mock"] = True
    complete_record["is_simulated"] = True
    complete_record["trajectory"] = _trajectory_for_level(level)
    complete_record["classroom_demo_level"] = level
    complete_record["demo_note"] = "模拟演示数据，不是真实老人输入。"
    return complete_record


def _synthesize_demo_audio(text: str, model: str, voice: str) -> dict[str, Any]:
    return synthesize_speech(
        text,
        model=model or None,
        voice=voice or None,
        config=config,
        prefer_remote_url=True,
    )


def _failed_demo_audio_result(model: str, voice: str, reason: str) -> dict[str, Any]:
    return {
        "audio_bytes": None,
        "mime_type": "audio/mpeg",
        "metadata": {
            "source": "fallback",
            "model": model or "未配置",
            "voice": voice,
            "reason": reason,
            "cached": False,
        },
    }


def _static_demo_audio_result(segment: dict[str, Any], level: str) -> dict[str, Any]:
    result_key = str(segment.get("result_key", "")).strip()
    if not result_key:
        return {}

    level_slug = _trajectory_for_level(level)
    for extension in CLASSROOM_DEMO_AUDIO_EXTENSIONS:
        audio_path = CLASSROOM_DEMO_AUDIO_DIR / level_slug / f"{result_key}{extension}"
        if not audio_path.is_file():
            continue
        try:
            audio_bytes = audio_path.read_bytes()
        except OSError:
            return _failed_demo_audio_result(
                str(segment.get("model", "")),
                str(segment.get("voice", "")),
                reason="static_audio_read_error",
            )
        return {
            "audio_bytes": audio_bytes,
            "mime_type": _mime_type_for_static_audio(audio_path),
            "metadata": {
                "source": "static_audio",
                "model": str(segment.get("model", "")) or "项目内固定音频",
                "voice": str(segment.get("voice", "")),
                "cached": True,
                "cache_path": str(audio_path.relative_to(PROJECT_ROOT)),
            },
        }

    return {}


def _mime_type_for_static_audio(audio_path: Path) -> str:
    suffix = audio_path.suffix.lower()
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".m4a":
        return "audio/mp4"
    if suffix == ".ogg":
        return "audio/ogg"
    return "audio/mpeg"


def _is_audio_success(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    metadata = result.get("metadata", {})
    source = str(metadata.get("source", ""))
    return bool(_audio_source(result)) and source in {"static_audio", "tts", "tts_url", "tts_cache"}


def _is_audio_failure(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    metadata = result.get("metadata", {})
    return str(metadata.get("source", "")) == "fallback" and not _is_audio_success(result)


def _clock_prompt_text(turn: dict[str, Any]) -> str:
    return str(turn.get("clock_trigger_elder_message", "")).strip()


def _clock_prompt_segment_key(turn_index: int) -> str:
    return f"clock_prompt_{turn_index}"


def _classroom_speech_html(role: str, text: Any) -> str:
    role_text = str(role).strip() or "角色"
    role_key = "assistant" if "小顾" in role_text else "elder" if "老人" in role_text else "neutral"
    safe_role = escape(role_text, quote=True)
    safe_text = escape(str(text), quote=True)
    return f"""
<div class="cg-chat-turn cg-classroom-speech cg-classroom-speech-{role_key}">
  <div class="cg-chat-role">{safe_role}</div>
  <div class="cg-chat-text">{safe_text}</div>
</div>
""".strip()


def _expected_demo_audio_segment_count(turns: list[dict[str, Any]]) -> int:
    return len(turns) * 2 + sum(1 for turn in turns if _clock_prompt_text(turn))


def _demo_audio_segments(
    turns: list[dict[str, Any]],
    assistant_model: str,
    assistant_voice: str,
    patient_model: str,
    patient_voice: str,
) -> list[dict[str, Any]]:
    segments = []
    segment_index = 0
    for turn_index, turn in enumerate(turns, start=1):
        for result_key, role_label, text, model, voice in [
            (
                f"system_{turn_index}",
                "小顾",
                str(turn.get("system_question", "")),
                assistant_model,
                assistant_voice,
            ),
            (
                f"patient_{turn_index}",
                "老人",
                str(turn.get("patient_answer", "")),
                patient_model,
                patient_voice,
            ),
        ]:
            segment_index += 1
            segments.append(
                {
                    "result_key": result_key,
                    "role_label": role_label,
                    "text": text,
                    "model": model,
                    "voice": voice,
                    "segment_index": segment_index,
                }
            )
        clock_prompt = _clock_prompt_text(turn)
        if clock_prompt:
            segment_index += 1
            segments.append(
                {
                    "result_key": _clock_prompt_segment_key(turn_index),
                    "role_label": "小顾画钟提示",
                    "text": clock_prompt,
                    "model": assistant_model,
                    "voice": assistant_voice,
                    "segment_index": segment_index,
                }
            )
    return segments


def _demo_audio_stats(segments: list[dict[str, Any]]) -> dict[str, int]:
    success_count = 0
    failure_count = 0
    mock_count = 0
    for segment in segments:
        result = st.session_state.classroom_tts_results.get(segment["result_key"]) or {}
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        if _is_audio_success(result):
            success_count += 1
        elif str(metadata.get("source", "")) == "mock":
            mock_count += 1
        elif _is_audio_failure(result):
            failure_count += 1
    return {
        "success": success_count,
        "failure": failure_count,
        "mock": mock_count,
        "total": len(segments),
    }


def _segment_range_label(segments: list[dict[str, Any]], total_segments: int) -> str:
    indexes = sorted(
        {
            int(segment.get("segment_index", 0))
            for segment in segments
            if int(segment.get("segment_index", 0)) > 0
        }
    )
    if not indexes:
        return f"0/{total_segments}"

    ranges = []
    start = indexes[0]
    previous = indexes[0]
    for index in indexes[1:]:
        if index == previous + 1:
            previous = index
            continue
        ranges.append(_format_index_range(start, previous))
        start = index
        previous = index
    ranges.append(_format_index_range(start, previous))
    return f"{'、'.join(ranges)}/{total_segments}"


def _format_index_range(start: int, end: int) -> str:
    if start == end:
        return str(start)
    return f"{start}-{end}"


def _failed_demo_result_keys(turns: list[dict[str, Any]]) -> list[str]:
    failed_keys = []
    for index, turn in enumerate(turns, start=1):
        result_keys = [f"system_{index}", f"patient_{index}"]
        if _clock_prompt_text(turn):
            result_keys.append(_clock_prompt_segment_key(index))
        for result_key in result_keys:
            if _is_audio_failure(st.session_state.classroom_tts_results.get(result_key)):
                failed_keys.append(result_key)
    return failed_keys


def _synthesize_demo_audio_with_retry(
    segment: dict[str, Any],
    level: str,
    max_attempts: int = DEMO_TTS_MAX_ATTEMPTS,
) -> dict[str, Any]:
    static_result = _static_demo_audio_result(segment, level)
    if _is_audio_success(static_result):
        return static_result

    result = {}
    for attempt in range(1, max_attempts + 1):
        try:
            result = _synthesize_demo_audio(
                str(segment.get("text", "")),
                str(segment.get("model", "")),
                str(segment.get("voice", "")),
            )
        except Exception as error:
            result = _failed_demo_audio_result(
                str(segment.get("model", "")),
                str(segment.get("voice", "")),
                reason=f"page_error: {type(error).__name__}",
            )

        if _is_audio_success(result):
            return result

        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        if str(metadata.get("source", "")) != "fallback" or attempt >= max_attempts:
            return result

        time.sleep(DEMO_TTS_RETRY_DELAY_SECONDS)

    return result


def _generate_all_demo_audio(
    level: str,
    turns: list[dict[str, Any]],
    assistant_model: str,
    assistant_voice: str,
    patient_model: str,
    patient_voice: str,
    retry_failed_only: bool = False,
    progress_bar: Any = None,
    status_slot: Any = None,
) -> None:
    segments = _demo_audio_segments(
        turns,
        assistant_model,
        assistant_voice,
        patient_model,
        patient_voice,
    )
    jobs = []
    skipped_success = 0
    for segment in segments:
        existing = st.session_state.classroom_tts_results.get(segment["result_key"])
        if _is_audio_success(existing):
            skipped_success += 1
            continue
        if retry_failed_only and not _is_audio_failure(existing):
            continue
        jobs.append(segment)

    if not jobs:
        stats = _demo_audio_stats(segments)
        st.session_state.classroom_tts_status = (
            f"无需生成新的语音：成功 {stats['success']}/{stats['total']} 段，"
            f"失败 {stats['failure']} 段，mock {stats['mock']} 段。"
        )
        return

    job_range_label = _segment_range_label(jobs, len(segments))
    max_workers = min(DEMO_TTS_MAX_WORKERS, len(jobs))
    if status_slot is not None:
        status_slot.info(
            f"正在并行生成第 {job_range_label} 段语音；"
            f"本次最多 {max_workers} 路并行，固定音频或缓存命中会快速跳过，真实生成可能较慢。"
        )
    if progress_bar is not None:
        progress_bar.progress(0)

    completed_jobs = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_segment = {
            executor.submit(_synthesize_demo_audio_with_retry, segment, level): segment
            for segment in jobs
        }
        for future in as_completed(future_to_segment):
            segment = future_to_segment[future]
            try:
                result = future.result()
            except Exception as error:
                result = _failed_demo_audio_result(
                    str(segment.get("model", "")),
                    str(segment.get("voice", "")),
                    reason=f"parallel_error: {type(error).__name__}",
                )
            st.session_state.classroom_tts_results[segment["result_key"]] = result
            completed_jobs += 1
            if progress_bar is not None:
                progress_bar.progress(completed_jobs / len(jobs))
            if status_slot is not None:
                status_slot.info(
                    f"正在并行生成第 {job_range_label} 段语音；"
                    f"已完成 {completed_jobs}/{len(jobs)} 段，"
                    f"刚完成：第 {segment['segment_index']} / {len(segments)} 段，"
                    f"{segment['role_label']}。"
                )

    stats = _demo_audio_stats(segments)
    action_label = "失败语音重试完成" if retry_failed_only else "全部演示语音生成完成"
    if stats["success"] == stats["total"]:
        st.session_state.classroom_tts_status = (
            f"{action_label}：成功 {stats['success']}/{stats['total']} 段，"
            "可点击播放完整演示。"
        )
    else:
        st.session_state.classroom_tts_status = (
            f"{action_label}：成功 {stats['success']}/{stats['total']} 段，"
            f"失败 {stats['failure']} 段，mock {stats['mock']} 段，"
            f"已跳过成功音频 {skipped_success} 段。可继续用文本演示，或点击“重试失败语音”。"
        )


def _audio_status_label(result_key: str) -> str:
    result = st.session_state.classroom_tts_results.get(result_key) or {}
    if not result:
        return "未生成"
    metadata = result.get("metadata", {})
    if metadata.get("source") == "static_audio":
        return "项目固定音频"
    audio_bytes = result.get("audio_bytes")
    if isinstance(audio_bytes, (bytes, bytearray)):
        return "使用缓存" if metadata.get("cached") else "已生成"
    if str(result.get("audio_url", "") or "").strip().lower().startswith(("http://", "https://")):
        return "已生成"
    return "生成失败"


def _display_audio_result(result_key: str, role_label: str = "音频", include_player: bool = True) -> None:
    result = st.session_state.classroom_tts_results.get(result_key) or {}
    if not result:
        st.caption(f"角色：{role_label}；音频状态：未生成")
        return

    metadata = result.get("metadata", {})
    audio_src = _audio_source(result)
    status = _audio_status_label(result_key)
    cached = str(bool(metadata.get("cached"))).lower()
    st.caption(
        f"角色：{role_label}；"
        f"音频状态：{status}；"
        f"模型：{metadata.get('model', '未配置')}；"
        f"音色：{metadata.get('voice', '未配置')}；"
        f"来源：{metadata.get('source', 'unknown')}；"
        f"cached：{cached}"
    )
    if audio_src and include_player:
        st.audio(audio_src, format=str(result.get("mime_type", "audio/mpeg")))
        return
    if audio_src:
        return

    st.warning("该轮语音生成失败，可继续用文本演示。")
    if metadata.get("reason"):
        st.caption(f"原因：{metadata.get('reason')}")
    if result_key.startswith("patient_") and (
        "CosyVoice" in str(metadata.get("reason", ""))
        or "longlaoyi" in str(metadata.get("voice", ""))
    ):
        st.caption("提示：单条诊断可用但批量失败时，通常是接口限流、超时或并发过高；可点击重试失败语音。")


def _audio_source(result: Any) -> Any:
    if not isinstance(result, dict):
        return ""
    audio_bytes = result.get("audio_bytes")
    if isinstance(audio_bytes, (bytes, bytearray)) and audio_bytes:
        return bytes(audio_bytes)
    audio_url = str(result.get("audio_url", "") or "").strip()
    if audio_url.lower().startswith(("http://", "https://")):
        return audio_url
    return ""


def _audio_data_url(result: Any) -> str:
    audio_source = _audio_source(result)
    if isinstance(audio_source, str) and audio_source.lower().startswith(("http://", "https://")):
        return audio_source
    if not isinstance(audio_source, (bytes, bytearray)):
        return ""
    mime_type = str(result.get("mime_type", "audio/mpeg"))
    encoded_audio = base64.b64encode(bytes(audio_source)).decode("ascii")
    return f"data:{mime_type};base64,{encoded_audio}"


def _build_demo_playlist(turns: list[dict[str, Any]]) -> list[dict[str, str]]:
    playlist = []
    for index, turn in enumerate(turns, start=1):
        playlist_items = [
            ("小顾", f"system_{index}", "system_question"),
            ("老人", f"patient_{index}", "patient_answer"),
        ]
        if _clock_prompt_text(turn):
            playlist_items.append(
                ("小顾画钟提示", _clock_prompt_segment_key(index), "clock_trigger_elder_message")
            )
        for role, result_key, text_key in playlist_items:
            data_url = _audio_data_url(st.session_state.classroom_tts_results.get(result_key))
            if not data_url:
                continue
            domain_label = DOMAIN_LABELS.get(
                turn.get("target_domain"),
                str(turn.get("target_domain", "")),
            )
            playlist.append(
                {
                    "label": f"第 {index} 轮｜{domain_label}｜{role}",
                    "role": role,
                    "text": str(turn.get(text_key, "")),
                    "src": data_url,
                }
            )
    return playlist


def _safe_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _estimate_full_demo_player_height(playlist: list[dict[str, str]]) -> int:
    base_height = 238
    row_height = 0
    for item in playlist:
        label_length = len(str(item.get("label", "")))
        text_length = len(str(item.get("text", "")))
        line_count = max(1, (label_length + text_length + 31) // 32)
        row_height += 34 + line_count * 23
    return max(420, min(1800, base_height + row_height))


def _render_full_demo_player(playlist: list[dict[str, str]]) -> None:
    player_html = """
<style>
  :root {
    color-scheme: light;
  }
  * {
    box-sizing: border-box;
  }
  body {
    margin: 0;
    background: transparent;
  }
  .demo-player {
    width: 100%;
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    border: 1px solid #bdd5ca;
    border-radius: 8px;
    padding: 18px;
    background: linear-gradient(180deg, #fbfffc 0%, #f2faf6 100%);
    color: #1d3148;
    box-shadow: 0 14px 34px rgba(29, 49, 72, 0.09);
  }
  .demo-primary {
    width: 100%;
    min-height: 54px;
    border-radius: 8px;
    border: 0;
    background: #2f7d62;
    color: #ffffff;
    font-size: 18px;
    font-weight: 800;
    cursor: pointer;
    box-shadow: 0 8px 18px rgba(47, 125, 98, 0.22);
  }
  .demo-primary:hover {
    background: #286e56;
  }
  .demo-controls {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 10px;
    margin-top: 10px;
  }
  .demo-control {
    min-height: 44px;
    border-radius: 8px;
    border: 1px solid #c8dcd2;
    background: rgba(255, 255, 255, 0.84);
    color: #1d3148;
    font-size: 15px;
    font-weight: 760;
    cursor: pointer;
  }
  .demo-control:hover {
    border-color: #7fb09a;
    background: #ffffff;
  }
  .demo-status {
    margin-top: 14px;
    border: 1px solid #cfe4d9;
    border-left: 4px solid #2f7d62;
    border-radius: 8px;
    padding: 10px 12px;
    background: #ffffff;
    color: #526579;
    font-size: 15px;
    line-height: 1.55;
  }
  .demo-list {
    display: grid;
    gap: 9px;
    list-style: none;
    margin: 14px 0 0;
    padding: 0;
    color: #1d3148;
  }
  .demo-play-row {
    display: grid;
    grid-template-columns: 34px minmax(0, 1fr);
    gap: 10px;
    align-items: start;
    margin: 0;
    border: 1px solid #d8e7df;
    border-radius: 8px;
    padding: 10px 12px;
    background: rgba(255, 255, 255, 0.82);
    line-height: 1.55;
    transition: border-color 150ms ease, background 150ms ease, box-shadow 150ms ease;
  }
  .demo-play-row.is-active {
    border-color: #2f7d62;
    background: #edf8f3;
    box-shadow: inset 0 0 0 1px rgba(47, 125, 98, 0.18);
  }
  .demo-index {
    display: inline-flex;
    width: 28px;
    height: 28px;
    align-items: center;
    justify-content: center;
    border-radius: 999px;
    background: #e4f2eb;
    color: #2f7d62;
    font-size: 14px;
    font-weight: 800;
  }
  .demo-play-row.is-active .demo-index {
    background: #2f7d62;
    color: #ffffff;
  }
  .demo-label {
    margin-bottom: 4px;
    color: #276a53;
    font-size: 14px;
    font-weight: 800;
  }
  .demo-text {
    color: #1d3148;
    font-size: 15px;
    overflow-wrap: anywhere;
  }
  @media (max-width: 560px) {
    .demo-player {
      padding: 14px;
    }
    .demo-controls {
      grid-template-columns: 1fr;
    }
    .demo-play-row {
      grid-template-columns: 30px minmax(0, 1fr);
      padding: 10px;
    }
  }
</style>
<div class="demo-player">
  <button id="playAllDemo" class="demo-primary" type="button">
    播放完整演示
  </button>
  <div class="demo-controls">
    <button id="pauseDemo" class="demo-control" type="button">暂停播放</button>
    <button id="resumeDemo" class="demo-control" type="button">继续播放</button>
    <button id="stopDemo" class="demo-control" type="button">停止播放</button>
  </div>
  <div id="demoStatus" class="demo-status">请点击按钮开始顺序播放。播放过程中可随时暂停、继续或停止。浏览器限制自动播放时，可使用页面下方备用音频控件。</div>
  <ol id="demoList" class="demo-list"></ol>
</div>
<script>
const playlist = __PLAYLIST_JSON__;
const button = document.getElementById("playAllDemo");
const pauseButton = document.getElementById("pauseDemo");
const resumeButton = document.getElementById("resumeDemo");
const stopButton = document.getElementById("stopDemo");
const status = document.getElementById("demoStatus");
const list = document.getElementById("demoList");
let currentAudio = null;
let currentIndex = 0;
let isStopped = true;
let isPaused = false;

function requestFrameResize() {
  const height = Math.ceil(document.documentElement.scrollHeight + 12);
  window.parent.postMessage({
    isStreamlitMessage: true,
    type: "streamlit:setFrameHeight",
    height: height
  }, "*");
}

function renderList(activeIndex) {
  list.innerHTML = "";
  playlist.forEach(function (item, index) {
    const row = document.createElement("li");
    row.className = "demo-play-row" + (index === activeIndex ? " is-active" : "");
    const number = document.createElement("span");
    number.className = "demo-index";
    number.textContent = String(index + 1);
    const content = document.createElement("div");
    const label = document.createElement("div");
    label.className = "demo-label";
    label.textContent = item.label;
    const text = document.createElement("div");
    text.className = "demo-text";
    text.textContent = item.text;
    content.appendChild(label);
    content.appendChild(text);
    row.appendChild(number);
    row.appendChild(content);
    list.appendChild(row);
  });
  window.requestAnimationFrame(requestFrameResize);
}

function playAt(index) {
  if (index >= playlist.length) {
    status.textContent = "完整演示播放完成。";
    renderList(-1);
    currentAudio = null;
    isStopped = true;
    isPaused = false;
    return;
  }
  isStopped = false;
  isPaused = false;
  currentIndex = index;
  const item = playlist[index];
  renderList(index);
  status.textContent = "正在播放：" + item.label;
  currentAudio = new Audio(item.src);
  currentAudio.onended = function () {
    if (!isStopped) {
      playAt(index + 1);
    }
  };
  currentAudio.onerror = function () {
    if (isStopped) {
      return;
    }
    status.textContent = item.label + " 播放失败，已尝试继续下一段。";
    playAt(index + 1);
  };
  currentAudio.play().catch(function () {
    status.textContent = "浏览器阻止了连续播放，请使用页面下方备用音频控件按顺序播放。";
  });
}

renderList(-1);
button.addEventListener("click", function () {
  if (!playlist.length) {
    status.textContent = "还没有可播放的演示语音。";
    return;
  }
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
  }
  isStopped = false;
  isPaused = false;
  currentIndex = 0;
  playAt(0);
});

pauseButton.addEventListener("click", function () {
  if (!currentAudio || isStopped) {
    status.textContent = "当前没有正在播放的音频。";
    return;
  }
  currentAudio.pause();
  isPaused = true;
  status.textContent = "已暂停播放。可以点击继续播放，或点击停止播放重置。";
});

resumeButton.addEventListener("click", function () {
  if (!currentAudio || isStopped) {
    if (playlist.length) {
      isStopped = false;
      playAt(currentIndex);
      return;
    }
    status.textContent = "还没有可继续播放的演示语音。";
    return;
  }
  currentAudio.play().then(function () {
    isPaused = false;
    status.textContent = "继续播放：" + playlist[currentIndex].label;
  }).catch(function () {
    status.textContent = "浏览器阻止了继续播放，请使用页面下方备用音频控件。";
  });
});

stopButton.addEventListener("click", function () {
  isStopped = true;
  isPaused = false;
  currentIndex = 0;
  if (currentAudio) {
    currentAudio.pause();
    currentAudio.currentTime = 0;
    currentAudio = null;
  }
  renderList(-1);
  status.textContent = "已停止播放。如需重新播放，请点击播放完整演示。";
});

if (window.ResizeObserver) {
  const resizeObserver = new ResizeObserver(requestFrameResize);
  resizeObserver.observe(document.body);
}
window.addEventListener("load", requestFrameResize);
</script>
""".replace("__PLAYLIST_JSON__", _safe_json(playlist))
    components.html(
        player_html,
        height=_estimate_full_demo_player_height(playlist),
    )


def _render_ordered_audio_controls(turns: list[dict[str, Any]]) -> None:
    for index, turn in enumerate(turns, start=1):
        st.markdown(f"**第 {index} 轮｜{DOMAIN_LABELS.get(turn.get('target_domain'), turn.get('target_domain', ''))}**")
        st.caption(f"小顾：{turn.get('system_question')}")
        _display_audio_result(f"system_{index}", role_label="小顾声音")
        st.caption(f"老人：{turn.get('patient_answer')}")
        _display_audio_result(f"patient_{index}", role_label="老人声音")
        clock_prompt = _clock_prompt_text(turn)
        if clock_prompt:
            st.caption(f"小顾画钟提示：{clock_prompt}")
            _display_audio_result(
                _clock_prompt_segment_key(index),
                role_label="小顾画钟提示",
            )


def _run_demo_evaluation() -> None:
    turns = st.session_state.classroom_demo_turns
    if not turns:
        st.session_state.classroom_save_status = "请先生成完整演示流程。"
        return
    _ensure_demo_report_cache(st.session_state.classroom_demo_level, turns)
    st.session_state.classroom_show_report = True
    st.session_state.classroom_save_status = ""


def _save_demo_report() -> None:
    report = st.session_state.classroom_report
    if not isinstance(report, dict):
        st.session_state.classroom_save_status = "请先查看完整报告。"
        return
    try:
        record = deepcopy(report)
        record["user_id"] = CURRENT_USER_ID
        record["participant_id"] = CURRENT_USER_ID
        record["is_mock"] = True
        record["is_simulated"] = True
        record["trajectory"] = _trajectory_for_level(st.session_state.classroom_demo_level)
        record["classroom_demo_level"] = st.session_state.classroom_demo_level
        record["demo_note"] = "模拟演示数据，不是真实老人输入。"
        save_session(record)
        save_session_memory(record)
        st.session_state.classroom_save_status = (
            f"已保存为{CURRENT_USER_DISPLAY_NAME}的一次综合演示评估，保存时间："
            f"{format_session_time(record['created_at'])}。这是模拟演示记录。"
        )
    except Exception as error:
        st.session_state.classroom_save_status = f"保存失败：{error}"


def _trajectory_for_level(level: str) -> str:
    if level == "轻度下降":
        return "mild_decline"
    if level == "明显异常":
        return "obvious_issue"
    return "normal"


def _expected_risk_for_turns(turns: list[dict[str, Any]]) -> str:
    if not turns:
        return "unknown"
    return str(turns[0].get("expected_risk", "unknown"))


def _clock_trigger_turn(turns: list[dict[str, Any]]) -> Optional[tuple[int, dict[str, Any]]]:
    for index, turn in enumerate(turns, start=1):
        if turn.get("clock_triggered"):
            return index, turn
    return None


def _classroom_clock_image_info(level: str) -> tuple[Path, str]:
    image_info = CLASSROOM_CLOCK_IMAGES.get(level, CLASSROOM_CLOCK_IMAGES["正常表现"])
    return CLASSROOM_CLOCK_IMAGE_DIR / image_info["filename"], image_info["caption"]


def _render_classroom_clock_photo(level: str, title: str, tone: str = "green") -> None:
    image_path, image_caption = _classroom_clock_image_info(level)
    if not image_path.exists():
        st.markdown(
            callout_html(
                title,
                "演示画钟图片资产暂未找到，可继续用文字时间线完成演示。",
                tone="amber",
            ),
            unsafe_allow_html=True,
        )
        return

    st.markdown(
        callout_html(
            title,
            "演示示意图，仅展示本次画钟结果。",
            tone=tone,
        ),
        unsafe_allow_html=True,
    )
    _left_column, image_column, _right_column = st.columns(CLASSROOM_CLOCK_IMAGE_COLUMN_RATIOS)
    with image_column:
        st.image(str(image_path), caption=image_caption, width="stretch")


def _render_classroom_flow_board(turns: list[dict[str, Any]]) -> None:
    risk_level = _expected_risk_for_turns(turns)
    trigger_tone = "terracotta" if risk_level == "high" else "amber" if risk_level == "medium" else "green"

    st.markdown(
        section_header_html(
            "展示链路",
            eyebrow="Demo Flow",
            body="先展示小顾和老人对话，再在回答下方给出动态下一步判断，最后运行评估报告。",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        status_strip_html(
            [
                {
                    "label": "当前场景",
                    "value": st.session_state.classroom_demo_level,
                    "tone": trigger_tone,
                },
                {
                    "label": "预期风险倾向",
                    "value": display_risk_level(risk_level),
                    "tone": trigger_tone,
                },
                {"label": "动态画钟", "value": "访谈结束后进入", "tone": "green"},
                {"label": "演示下一步", "value": "先看对话时间线", "tone": "green"},
            ]
        ),
        unsafe_allow_html=True,
    )
    st.caption("系统会记录对话中的风险信号，但演示统一在访谈结束后进入画钟。")


def _display_report_evidence(report: dict[str, Any]) -> None:
    evidence = report.get("evidence", [])
    st.markdown("**主要证据**")
    if not evidence:
        st.caption("暂无可展示证据。")
        return
    cards = []
    for item in evidence:
        if isinstance(item, dict):
            domain = DOMAIN_LABELS.get(item.get("domain"), item.get("domain", ""))
            text = item.get("text", "")
            source = item.get("source")
            meta = "画钟观察" if source == "clock" else "对话观察" if source == "dialog" else "综合证据"
            tone = "blue" if source == "clock" else "green"
            cards.append(evidence_card_html(domain, text, meta=meta, tone=tone))
        else:
            cards.append(evidence_card_html("证据", item, tone="green"))
    st.markdown(
        '<div class="cg-evidence-grid">' + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def _score_text(score_result: dict[str, Any], scale_suffix: str) -> str:
    score = score_result.get("score")
    if isinstance(score, (int, float)) and not isinstance(score, bool):
        return f"{score:g} {scale_suffix}"
    return "暂无"


def _display_report_clock_summary(report: dict[str, Any]) -> None:
    clock_result = report.get("clock_result")
    if not isinstance(clock_result, dict):
        st.caption("本次完整报告中暂无画钟分析。")
        return

    findings = clock_result.get("clock_findings", {})
    cdt_features = clock_result.get("cdt_features", {})
    st.markdown("**画钟分析**")
    chip_items = [
        chip_html("来源", display_source(clock_result.get("source", "mock")), "blue"),
        chip_html("模型", clock_result.get("model", "演示示意"), "neutral"),
    ]
    if isinstance(cdt_features, dict) and cdt_features:
        chip_items.extend(
            [
                chip_html(
                    "数字分布",
                    display_cdt_feature_value(
                        "number_distribution",
                        cdt_features.get("number_distribution"),
                    ),
                    "green",
                ),
                chip_html(
                    "数字间距",
                    display_cdt_feature_value(
                        "number_spacing",
                        cdt_features.get("number_spacing"),
                    ),
                    "amber",
                ),
                chip_html(
                    "目标时间",
                    display_cdt_feature_value(
                        "target_time_match",
                        cdt_features.get("target_time_match"),
                    ),
                    "terracotta",
                ),
            ]
        )
    st.markdown("".join(chip_items), unsafe_allow_html=True)

    if isinstance(findings, dict):
        st.markdown(
            '<div class="cg-evidence-grid">'
            + evidence_card_html(
                "数字布局",
                findings.get("number_placement", "暂无"),
                meta="画钟观察",
                tone="blue",
            )
            + evidence_card_html(
                "指针准确性",
                findings.get("hand_accuracy", "暂无"),
                meta="画钟观察",
                tone="terracotta",
            )
            + "</div>",
            unsafe_allow_html=True,
        )


_init_state()

st.markdown(
    section_header_html(
        "认知水平选择",
        eyebrow="Scenario",
        body="三类轨迹均为模拟数据，用来展示系统对稳定、轻度下降和明显异常线索的响应。",
    ),
    unsafe_allow_html=True,
)
selected_level = st.selectbox(
    "选择模拟认知水平",
    options=list(CLASSROOM_DEMO_LEVEL_LABELS),
    index=list(CLASSROOM_DEMO_LEVEL_LABELS).index(st.session_state.classroom_demo_level),
)
scenario_copy = CLASSROOM_SCENARIO_COPY.get(selected_level, CLASSROOM_SCENARIO_COPY["正常表现"])
st.markdown(
    status_strip_html(
        [
            {"label": "预设输入", "value": selected_level, "tone": scenario_copy["tone"]},
            {"label": "动态画钟", "value": scenario_copy["trigger"], "tone": scenario_copy["tone"]},
            {"label": "演示用途", "value": "模拟自然访谈", "tone": "green"},
        ]
    ),
    unsafe_allow_html=True,
)
st.caption(scenario_copy["summary"])

if st.button("生成完整演示流程", type="primary", width="stretch"):
    _generate_demo_flow(selected_level)

turns = st.session_state.classroom_demo_turns
if not turns:
    st.info("请选择模拟认知水平，然后点击“生成完整演示流程”。")
    st.caption(DISCLAIMER)
    st.stop()

_render_classroom_flow_board(turns)
with st.spinner("正在预生成完整报告缓存，稍后页面底部可直接查看……"):
    _ensure_demo_report_cache(st.session_state.classroom_demo_level, turns)
if st.session_state.classroom_report_status:
    st.caption(st.session_state.classroom_report_status)

assistant_model = (
    config.tts_model_assistant.strip()
    or config.tts_model.strip()
    or "qwen-tts"
)
assistant_voice = config.tts_voice_assistant.strip() or DEFAULT_ASSISTANT_VOICE
patient_model = config.tts_model_patient_demo.strip() or DEFAULT_PATIENT_DEMO_MODEL
patient_voice = config.tts_voice_patient_demo.strip() or DEFAULT_PATIENT_DEMO_VOICE

st.markdown(
    section_header_html(
        "语音生成配置",
        eyebrow="Voice Cast",
        body="小顾声音和老人声音使用不同模型或音色，便于演示时区分角色。",
    ),
    unsafe_allow_html=True,
)
voice_columns = st.columns(2)
with voice_columns[0]:
    st.caption(
        f"小顾声音：模型 {assistant_model}，音色 {assistant_voice}。"
        f"TTS_MODEL_ASSISTANT = {assistant_model}，TTS_VOICE_ASSISTANT = {assistant_voice}。"
    )
with voice_columns[1]:
    st.caption(
        f"老人声音：模型 {patient_model}，音色 {patient_voice}。"
        f"TTS_MODEL_PATIENT_DEMO = {patient_model}，TTS_VOICE_PATIENT_DEMO = {patient_voice}。"
    )
st.markdown(
    status_strip_html(
        [
            {"label": "小顾声音", "value": f"{assistant_model} / {assistant_voice}", "tone": "blue"},
            {"label": "老人声音", "value": f"{patient_model} / {patient_voice}", "tone": "amber"},
            {"label": "生成策略", "value": "固定音频优先，最多 7 路并行", "tone": "green"},
        ]
    ),
    unsafe_allow_html=True,
)

if assistant_model == patient_model and assistant_voice == patient_voice:
    st.warning("建议为老人声音配置不同音色，便于演示区分。")
elif assistant_voice == patient_voice:
    st.warning("小顾声音和老人声音音色相同；建议为老人声音配置不同音色，便于演示区分。")
else:
    st.success("小顾声音和老人声音使用不同模型或音色，便于演示区分。")
st.caption("老人声音来自 TTS_MODEL_PATIENT_DEMO / TTS_VOICE_PATIENT_DEMO；修改本地 .env 后请重启 Streamlit。若模型、音色或账号权限不兼容，会 fallback，不影响文本演示。")

st.markdown("### 演示主流程")
control_columns = st.columns(2)
with control_columns[0]:
    if st.button("生成全部演示语音", type="primary", width="stretch"):
        with st.spinner(
            "正在并行生成演示语音，首次生成可能需要较长时间，请稍候……"
        ):
            progress_bar = st.progress(0)
            status_slot = st.empty()
            _generate_all_demo_audio(
                st.session_state.classroom_demo_level,
                turns,
                assistant_model,
                assistant_voice,
                patient_model,
                patient_voice,
                progress_bar=progress_bar,
                status_slot=status_slot,
            )
            status_slot.empty()
with control_columns[1]:
    if st.button("播放完整演示", width="stretch"):
        st.session_state.classroom_show_player = True

if st.session_state.classroom_tts_status:
    st.info(st.session_state.classroom_tts_status)

failed_audio_keys = _failed_demo_result_keys(turns)
if failed_audio_keys:
    st.warning("单条诊断可用但批量失败时，通常是接口限流、超时或并发过高；可点击重试失败语音。")
    if st.button("重试失败语音", width="stretch"):
        with st.spinner("正在并行重试失败语音，请稍候……"):
            retry_progress_bar = st.progress(0)
            retry_status_slot = st.empty()
            _generate_all_demo_audio(
                st.session_state.classroom_demo_level,
                turns,
                assistant_model,
                assistant_voice,
                patient_model,
                patient_voice,
                retry_failed_only=True,
                progress_bar=retry_progress_bar,
                status_slot=retry_status_slot,
            )
            retry_status_slot.empty()

if st.session_state.classroom_show_player:
    st.markdown("### 完整演示播放区")
    playlist = _build_demo_playlist(turns)
    expected_segments = _expected_demo_audio_segment_count(turns)
    if len(playlist) == expected_segments:
        st.success("完整演示语音已准备好。请点击播放区按钮顺序播放：小顾 → 老人；触发画钟时会追加小顾画钟提示。")
        _render_full_demo_player(playlist)
        with st.expander("浏览器限制自动播放时的备用顺序音频控件"):
            _render_ordered_audio_controls(turns)
    else:
        st.warning("完整顺序播放需要先生成全部小顾/老人/画钟提示音频。若部分 TTS 失败，可继续用下方文本和音频控件演示。")
        _render_ordered_audio_controls(turns)

st.markdown("### 演示时间线")
st.markdown(
    section_header_html(
        "演示时间线",
        eyebrow="Run of Show",
        body="每轮展示目标认知域、小顾提问和老人回答；触发画钟时会追加小顾说明和音频状态。",
    ),
    unsafe_allow_html=True,
)
st.caption(f"当前选择：{st.session_state.classroom_demo_level}；预期风险倾向：{display_risk_level(_expected_risk_for_turns(turns))}。")
for index, turn in enumerate(turns, start=1):
    domain_label = DOMAIN_LABELS.get(turn.get("target_domain"), turn.get("target_domain", ""))
    is_clock_trigger = bool(turn.get("clock_triggered"))
    turn_tone = "amber" if is_clock_trigger else "green"
    with st.expander(f"第 {index} 轮｜{domain_label}", expanded=index == 1 or is_clock_trigger):
        st.caption(f"认知水平：{turn.get('cognitive_level')}；预期风险倾向：{display_risk_level(turn.get('expected_risk', 'unknown'))}")
        st.markdown(
            timeline_item_html(
                index,
                f"{domain_label} · {turn.get('cognitive_level')}",
                f"小顾：{turn.get('system_question')}",
                meta=f"老人：{turn.get('patient_answer')}",
                tone=turn_tone,
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            _classroom_speech_html("小顾", turn.get("system_question")),
            unsafe_allow_html=True,
        )
        st.caption("小顾声音")
        _display_audio_result(f"system_{index}", role_label="小顾声音", include_player=False)
        st.markdown(
            _classroom_speech_html("老人", turn.get("patient_answer")),
            unsafe_allow_html=True,
        )
        st.caption("老人声音")
        _display_audio_result(f"patient_{index}", role_label="老人声音", include_player=False)
        if is_clock_trigger:
            clock_prompt = _clock_prompt_text(turn)
            st.markdown(
                chip_html("动态下一步", "进入画钟拍照", "amber"),
                unsafe_allow_html=True,
            )
            if clock_prompt:
                st.markdown(
                    _classroom_speech_html("小顾", clock_prompt),
                    unsafe_allow_html=True,
                )
                st.caption("小顾画钟提示声音")
                _display_audio_result(
                    _clock_prompt_segment_key(index),
                    role_label="小顾画钟提示",
                    include_player=False,
                )
            st.markdown(
                callout_html(
                    str(turn.get("clock_trigger_title") or "建议补充画钟拍照"),
                    (
                        f"{turn.get('clock_trigger_staff_note')} "
                        "展示页只表明需要拍照，不要求现场打开摄像头；真实产品流程会进入画钟拍照页。"
                    ),
                    tone="amber",
                ),
                unsafe_allow_html=True,
            )
            _render_classroom_clock_photo(
                st.session_state.classroom_demo_level,
                "画钟照片",
                tone="amber",
            )

if _clock_trigger_turn(turns) is None:
    st.markdown(
        section_header_html(
            "演示画钟照片",
            eyebrow="Clock Photo",
            body="当前场景没有自动触发画钟；这里展示一张正常画钟示意图。",
        ),
        unsafe_allow_html=True,
    )
    _render_classroom_clock_photo(
        st.session_state.classroom_demo_level,
        "画钟照片",
        tone="green",
    )

st.markdown("### 完整评估报告")
st.markdown(
    section_header_html(
        "完整综合报告",
        eyebrow="Dialogue + Clock Report",
        body="报告已提前缓存，包含对话评估和画钟分析，可在此展开完整综合结果。",
    ),
    unsafe_allow_html=True,
)
if st.session_state.classroom_report_status:
    st.caption(st.session_state.classroom_report_status)
st.button(
    "运行评估并查看报告",
    type="primary",
    on_click=_run_demo_evaluation,
    width="stretch",
)

report = (
    st.session_state.classroom_report
    if st.session_state.classroom_show_report
    else None
)
if isinstance(report, dict):
    st.caption(
        f"预期风险倾向：{display_risk_level(_expected_risk_for_turns(turns))}。"
        "该提示只用于演示，不会篡改模型输出。"
    )
    dialogue_score = compute_dialogue_score(report)
    clock_score = compute_clock_structure_score(report)
    cogniguard_score = compute_cogniguard_score(report)
    st.markdown(
        '<div class="cg-metric-grid">'
        + metric_card_html(
            "CogniGuard 综合提示分",
            _score_text(cogniguard_score, "/ 100"),
            cogniguard_score["band"],
            "green",
        )
        + metric_card_html(
            "风险等级",
            display_risk_level(report.get("risk_level", "unknown")),
            "对话和画钟合并后的非诊断提示",
            "terracotta",
        )
        + metric_card_html(
            "对话评估参考分",
            _score_text(dialogue_score, "/ 100"),
            dialogue_score["band"],
            "blue",
        )
        + metric_card_html(
            "画钟结构分",
            _score_text(clock_score, "/ 10"),
            "演示 CDT 结构分",
            "amber",
        )
        + metric_card_html(
            "报告组成",
            "对话 + 画钟",
            "完整综合报告",
            "neutral",
        )
        + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown(risk_badge_html(report.get("risk_level", "unknown")), unsafe_allow_html=True)
    st.markdown(callout_html("报告解释", report.get("explanation", ""), tone="green"), unsafe_allow_html=True)
    _display_report_clock_summary(report)
    _display_report_evidence(report)
    st.caption(report.get("disclaimer", DISCLAIMER))
    st.caption("本评估基于模拟回答和演示画钟示意结果生成，模拟演示数据，不是真实老人输入。老人音频不会再走 ASR。")
    st.button(
        "保存为张奶奶的一次综合演示评估",
        on_click=_save_demo_report,
        width="stretch",
    )
else:
    st.info("报告已提前缓存。点击上方按钮即可展开对话 + 画钟综合报告。")

if st.session_state.classroom_save_status:
    if st.session_state.classroom_save_status.startswith("已保存"):
        st.success(st.session_state.classroom_save_status)
    else:
        st.warning(st.session_state.classroom_save_status)

st.caption(DISCLAIMER)
