import base64
import hashlib
import html
import json
from typing import Any, Optional

import streamlit as st
import streamlit.components.v1 as components

from core.asr_client import transcribe_audio
from core.assessment_flow import (
    FLOW_STEP_CLOCK_TEST,
    FLOW_STEP_FINISH,
    build_assessment_flow_summary,
)
from core.config import load_config
from core.db import save_session
from core.elder_voice_component import elder_voice_recorder
from core.llm_client import evaluate_dialogue, generate_next_question
from core.memory import save_session_memory
from core.mock_data import (
    INTERVIEW_COMPLETED_MESSAGE,
    PRESET_INTERVIEW_QUESTIONS,
    all_dialog_domains_covered,
    get_covered_dialog_domains,
    get_next_preset_interview_question,
    infer_dialog_question_type,
    normalize_dialog_domains,
)
from core.report import format_session_time
from core.schemas import COGNITIVE_DOMAINS, DISCLAIMER
from core.session_history import (
    build_history_personalized_start,
    build_dialog_assessment_record,
    find_assessment_record,
    get_current_user_profile,
    load_current_user_sessions,
)
from core.staff_gate import hide_sidebar_nav
from core.tts_client import synthesize_speech
from core.ui import inject_elder_theme, status_pill_html


ELDER_SILENCE_MS = 2000
ELDER_MAX_RECORDING_MS = 45000
ELDER_NO_SPEECH_TIMEOUT_MS = 15000
ELDER_MIN_RECORDING_MS = 1200
ELDER_SILENCE_THRESHOLD = 0.018
ELDER_FINISH_MESSAGE = "结果已经整理好，请交给家人查看。"


st.set_page_config(page_title="和小顾聊天", layout="wide")
hide_sidebar_nav()
inject_elder_theme()

st.markdown(
    """
<style>
.st-key-elder_manual_start_button div.stButton > button {
    min-height: min(34vh, 18rem);
    width: 100%;
    border-radius: 22px;
    font-size: clamp(2rem, 6vw, 4.3rem);
    font-weight: 900;
}
.st-key-elder_clock_next_button div.stButton > button {
    min-height: 4.8rem;
    width: 100%;
    border-radius: 18px;
    font-size: clamp(1.25rem, 2.4vw, 1.95rem);
    font-weight: 700;
}
.elder-complete-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.1fr) minmax(300px, 0.72fr);
    gap: clamp(1rem, 2.2vw, 1.45rem);
    align-items: start;
    margin-top: 0.65rem;
}
.elder-complete-primary .elder-done {
    font-size: clamp(1.9rem, 4.3vw, 3.35rem);
    padding: clamp(1rem, 2.1vw, 1.45rem);
}
.elder-complete-primary .elder-next-main {
    font-size: clamp(1.75rem, 3.8vw, 3.05rem);
    margin: 0.65rem 0;
    padding: clamp(1rem, 2.1vw, 1.45rem);
}
.elder-complete-action-panel {
    border: 1px solid #cbdccb;
    border-radius: 22px;
    background: linear-gradient(180deg, #fffdf8 0%, #eef5ef 100%);
    padding: clamp(1rem, 2vw, 1.3rem);
    box-shadow: 0 18px 44px rgba(31, 36, 33, 0.08);
    position: sticky;
    top: 0.85rem;
}
.elder-action-title {
    font-family: var(--cg-serif);
    color: var(--cg-green-dark);
    font-size: clamp(1.45rem, 2.8vw, 2rem);
    font-weight: 700;
    line-height: 1.2;
    margin-bottom: 0.55rem;
}
.elder-action-copy {
    color: var(--cg-muted);
    font-size: clamp(1rem, 1.8vw, 1.2rem);
    line-height: 1.55;
    margin-bottom: 0.75rem;
}
.elder-transition-audio-note {
    color: var(--cg-soft);
    font-size: 0.95rem;
    line-height: 1.5;
    margin: 0.45rem 0 0.7rem;
}
@media (max-width: 760px) {
    .st-key-elder_manual_start_button div.stButton > button {
        min-height: 13rem;
    }
    .elder-complete-grid {
        grid-template-columns: 1fr;
    }
    .elder-complete-action-panel {
        position: static;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

config = load_config()
current_user = get_current_user_profile(st.session_state)
current_user_id = current_user["user_id"]
current_display_name = current_user["display_name"]


def _history_hint() -> str:
    return _initial_context()["elder_hint"]


def _initial_context() -> dict[str, Any]:
    base_question = PRESET_INTERVIEW_QUESTIONS[0]["question"]
    base_domain = PRESET_INTERVIEW_QUESTIONS[0]["domain"]
    current_history = load_current_user_sessions(limit=3, user_profile=current_user)
    return build_history_personalized_start(
        current_history["sessions"],
        display_name=current_display_name,
        fallback_question=base_question,
        fallback_domain=base_domain,
    )


def _initial_question() -> str:
    return str(_initial_context()["question"])


def _initial_domain() -> str:
    return str(_initial_context()["target_domain"])


def _init_state() -> None:
    defaults = {
        "elder_started": False,
        "elder_turns": [],
        "elder_current_question": _initial_question(),
        "elder_current_domain": _initial_domain(),
        "elder_covered_domains": [],
        "elder_status": "请点一下屏幕开始。",
        "elder_tts_result": None,
        "elder_tts_status": "",
        "elder_recorder_auto_start": False,
        "elder_last_audio_signature": "",
        "elder_last_processed_recording_key": "",
        "elder_last_transcript": "",
        "elder_last_transcript_status": "",
        "elder_voice_attempt": 0,
        "elder_report": None,
        "elder_save_status": "",
        "elder_current_assessment_id": None,
        "elder_transition_tts_text": "",
        "elder_transition_tts_result": None,
        "elder_transition_tts_status": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - compatibility for older Streamlit.
        st.experimental_rerun()


def _start_interview() -> None:
    st.session_state.elder_started = True
    st.session_state.elder_turns = []
    initial_context = _initial_context()
    st.session_state.elder_current_question = str(initial_context["question"])
    st.session_state.elder_current_domain = str(initial_context["target_domain"])
    st.session_state.elder_covered_domains = []
    st.session_state.elder_status = "请听问题，小顾会慢慢说。"
    st.session_state.elder_tts_result = None
    st.session_state.elder_tts_status = ""
    st.session_state.elder_recorder_auto_start = False
    st.session_state.elder_last_audio_signature = ""
    st.session_state.elder_last_processed_recording_key = ""
    st.session_state.elder_last_transcript = ""
    st.session_state.elder_last_transcript_status = ""
    st.session_state.elder_voice_attempt = 0
    st.session_state.elder_report = None
    st.session_state.elder_save_status = ""
    st.session_state.elder_current_assessment_id = None
    st.session_state.elder_transition_tts_text = ""
    st.session_state.elder_transition_tts_result = None
    st.session_state.elder_transition_tts_status = ""
    st.session_state.current_assessment_user_id = current_user_id
    _synthesize_question()


def _covered_domains() -> list[str]:
    stored = normalize_dialog_domains(st.session_state.elder_covered_domains)
    inferred = get_covered_dialog_domains(st.session_state.elder_turns)
    return normalize_dialog_domains([*stored, *inferred])


def _interview_is_complete() -> bool:
    return (
        all_dialog_domains_covered(_covered_domains())
        or st.session_state.elder_current_question == INTERVIEW_COMPLETED_MESSAGE
    )


def _flatten_turns(turns: list[dict[str, str]]) -> list[str]:
    messages = []
    for turn in turns:
        assistant = turn.get("assistant", "").strip()
        user = turn.get("user", "").strip()
        if assistant:
            messages.append(f"AI访谈问题：{assistant}")
        if user:
            messages.append(f"老人回答：{user}")
    return messages


def _mark_current_question_covered() -> None:
    domain = st.session_state.elder_current_domain or infer_dialog_question_type(
        st.session_state.elder_current_question
    )
    if domain in COGNITIVE_DOMAINS:
        covered = normalize_dialog_domains(st.session_state.elder_covered_domains)
        if domain not in covered:
            covered.append(domain)
        st.session_state.elder_covered_domains = covered


def _set_interview_completed() -> None:
    st.session_state.elder_current_question = INTERVIEW_COMPLETED_MESSAGE
    st.session_state.elder_current_domain = ""
    st.session_state.elder_status = "访谈完成了，谢谢您。"
    st.session_state.elder_tts_result = None
    st.session_state.elder_tts_status = ""
    st.session_state.elder_recorder_auto_start = False


def _set_next_question() -> None:
    if all_dialog_domains_covered(_covered_domains()):
        _set_interview_completed()
        return

    fallback_reason = ""
    try:
        result = generate_next_question(
            _flatten_turns(st.session_state.elder_turns),
            config=config,
            covered_domains=_covered_domains(),
        )
        question = result.get("question") if isinstance(result, dict) else ""
        target_domain = result.get("target_domain", "") if isinstance(result, dict) else ""
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        fallback_reason = str(metadata.get("reason", "")).strip()
        source = metadata.get("source", "")
        if (
            question
            and source in {"qwen", "mock"}
            and target_domain in COGNITIVE_DOMAINS
            and target_domain not in _covered_domains()
        ):
            st.session_state.elder_current_question = question
            st.session_state.elder_current_domain = target_domain
            st.session_state.elder_status = "请听下一题。"
            _synthesize_question()
            return
    except Exception as error:
        fallback_reason = f"下一题准备失败：{type(error).__name__}"

    preset = get_next_preset_interview_question(_covered_domains())
    if preset is None:
        _set_interview_completed()
        return
    st.session_state.elder_current_question = preset["question"]
    st.session_state.elder_current_domain = preset["domain"]
    st.session_state.elder_status = "请听下一题。" if not fallback_reason else "请听下一题。"
    _synthesize_question()


def _submit_answer(answer_text: str) -> None:
    answer = str(answer_text or "").strip()
    if not answer:
        st.session_state.elder_status = "没有听清，请重新录音。"
        return
    if _interview_is_complete():
        st.session_state.elder_status = "访谈完成了，谢谢您。"
        return

    st.session_state.elder_turns.append(
        {
            "assistant": st.session_state.elder_current_question.strip(),
            "user": answer,
            "target_domain": st.session_state.elder_current_domain,
        }
    )
    _mark_current_question_covered()
    st.session_state.elder_report = None
    st.session_state.elder_voice_attempt = 0
    st.session_state.elder_status = "已经记录，正在准备下一题。"
    _set_next_question()


def _read_audio_bytes(audio_file: Any) -> bytes:
    if audio_file is None:
        return b""
    if isinstance(audio_file, bytes):
        return audio_file
    if isinstance(audio_file, bytearray):
        return bytes(audio_file)
    if hasattr(audio_file, "getvalue"):
        value = audio_file.getvalue()
        return bytes(value) if isinstance(value, (bytes, bytearray)) else b""
    if hasattr(audio_file, "read"):
        value = audio_file.read()
        return bytes(value) if isinstance(value, (bytes, bytearray)) else b""
    return b""


def _audio_signature(audio_bytes: bytes) -> str:
    if not audio_bytes:
        return ""
    return hashlib.sha256(audio_bytes).hexdigest()


def _process_recorded_audio(audio_file: Any, filename: str = "elder_recording.wav") -> bool:
    audio_bytes = _read_audio_bytes(audio_file)
    signature = _audio_signature(audio_bytes)
    if not signature or signature == st.session_state.elder_last_audio_signature:
        return False

    st.session_state.elder_last_audio_signature = signature
    st.session_state.elder_status = "正在识别，请稍等。"
    result = transcribe_audio(
        audio=audio_bytes,
        filename=getattr(audio_file, "name", filename),
        config=config,
    )
    text = str(result.get("text", "")).strip()
    if text:
        st.session_state.elder_last_transcript = text
        st.session_state.elder_last_transcript_status = "上一题已经识别并记录。"
        _submit_answer(text)
    else:
        metadata = result.get("metadata", {})
        reason = metadata.get("reason") or "没有听清，请重新录音。"
        st.session_state.elder_status = reason
        st.session_state.elder_last_transcript = ""
        st.session_state.elder_last_transcript_status = reason
        st.session_state.elder_recorder_auto_start = False
        st.session_state.elder_voice_attempt += 1
    return True


def _decode_audio_base64(value: Any) -> bytes:
    if not isinstance(value, str) or not value.strip():
        return b""
    candidate = value.strip()
    if candidate.startswith("data:") and ";base64," in candidate:
        candidate = candidate.split(";base64,", 1)[1]
    try:
        return base64.b64decode(candidate, validate=True)
    except Exception:
        return b""


def _filename_for_mime_type(mime_type: str) -> str:
    normalized = str(mime_type or "").split(";", 1)[0].strip().lower()
    extension_by_mime = {
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/wav": "wav",
        "audio/mpeg": "mp3",
        "audio/mp4": "m4a",
    }
    extension = extension_by_mime.get(normalized, "webm")
    return f"elder_auto_recording.{extension}"


def _process_auto_recorder_result(result: Any) -> bool:
    if not isinstance(result, dict):
        return False

    status = str(result.get("status", "")).strip()
    if status == "error":
        message = str(result.get("message", "")).strip()
        st.session_state.elder_status = message or "没有听清，请再说一遍。"
        st.session_state.elder_last_transcript = ""
        st.session_state.elder_last_transcript_status = st.session_state.elder_status
        st.session_state.elder_recorder_auto_start = False
        return False
    if status != "recorded":
        return False

    recording_key = str(result.get("recording_key", "")).strip()
    if recording_key:
        if recording_key != _current_recording_key():
            return False
        if recording_key == st.session_state.get("elder_last_processed_recording_key", ""):
            return False
        st.session_state.elder_last_processed_recording_key = recording_key

    audio_bytes = _decode_audio_base64(result.get("audio_base64"))
    if not audio_bytes:
        st.session_state.elder_status = "没有听清，请再说一遍。"
        st.session_state.elder_last_transcript = ""
        st.session_state.elder_last_transcript_status = st.session_state.elder_status
        st.session_state.elder_recorder_auto_start = False
        st.session_state.elder_voice_attempt += 1
        return True

    mime_type = str(result.get("mime_type", "audio/webm")).strip()
    filename = _filename_for_mime_type(mime_type)
    return _process_recorded_audio(audio_bytes, filename=filename)


def _synthesize_question() -> None:
    question = st.session_state.elder_current_question.strip()
    if not question or question == INTERVIEW_COMPLETED_MESSAGE:
        st.session_state.elder_tts_result = None
        st.session_state.elder_tts_status = ""
        st.session_state.elder_recorder_auto_start = False
        return

    st.session_state.elder_tts_status = "正在准备声音，请稍等。"
    result = synthesize_speech(question, config=config, prefer_remote_url=True)
    st.session_state.elder_tts_result = result
    metadata = result.get("metadata", {})
    if _result_has_audio(result):
        if metadata.get("cached"):
            st.session_state.elder_tts_status = "请听问题。"
        else:
            st.session_state.elder_tts_status = "请听问题。"
    else:
        st.session_state.elder_tts_status = "请看小顾的问题，然后录音回答。"
    st.session_state.elder_recorder_auto_start = True


def _ensure_transition_voice(message: str) -> None:
    transition_text = str(message or "").strip()
    if not transition_text:
        st.session_state.elder_transition_tts_text = ""
        st.session_state.elder_transition_tts_result = None
        st.session_state.elder_transition_tts_status = ""
        return
    if (
        st.session_state.get("elder_transition_tts_text") == transition_text
        and isinstance(st.session_state.get("elder_transition_tts_result"), dict)
    ):
        return

    st.session_state.elder_transition_tts_text = transition_text
    result = synthesize_speech(transition_text, config=config, prefer_remote_url=True)
    st.session_state.elder_transition_tts_result = result
    metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
    if _result_has_audio(result):
        source = "缓存语音" if metadata.get("cached") else "小顾语音"
        st.session_state.elder_transition_tts_status = f"{source}已准备好。"
    else:
        st.session_state.elder_transition_tts_status = "小顾语音暂未生成，可继续按文字提示操作。"


def _audio_data_url(audio_bytes: bytes, mime_type: str) -> str:
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded_audio}"


def _result_has_audio(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    audio_bytes = result.get("audio_bytes")
    if isinstance(audio_bytes, (bytes, bytearray)) and audio_bytes:
        return True
    audio_url = str(result.get("audio_url", "") or "").strip().lower()
    return audio_url.startswith(("http://", "https://"))


def _tts_audio_source(result: Any) -> tuple[str, str]:
    if not isinstance(result, dict):
        return "", "audio/mpeg"
    mime_type = str(result.get("mime_type", "audio/mpeg"))
    audio_bytes = result.get("audio_bytes")
    if isinstance(audio_bytes, (bytes, bytearray)) and audio_bytes:
        return _audio_data_url(bytes(audio_bytes), mime_type), mime_type
    audio_url = str(result.get("audio_url", "") or "").strip()
    if audio_url.lower().startswith(("http://", "https://")):
        return audio_url, mime_type
    return "", mime_type


def _safe_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _render_transition_audio_player(
    audio_src: str,
    mime_type: str,
    message: str,
) -> None:
    payload = {
        "src": audio_src,
        "message": str(message or ""),
    }
    player_html = """
<div style="font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; border: 1px solid #cbdccb; border-radius: 14px; padding: 12px; background: #fffdf8;">
  <div id="transitionStatus" style="color: #526579; font-size: 15px; line-height: 1.5; margin-bottom: 8px;">正在准备自动播放小顾提示。</div>
  <button id="transitionPlayButton" type="button" style="display: none; width: 100%; min-height: 46px; border-radius: 10px; border: 0; background: #5c7a6b; color: white; font-size: 16px; font-weight: 800; cursor: pointer; margin-bottom: 8px;">播放小顾提示</button>
  <audio id="transitionAudio" controls preload="auto" style="width: 100%; height: 42px;"></audio>
</div>
<script>
const payload = __PAYLOAD_JSON__;
const transitionAudio = document.getElementById("transitionAudio");
const status = document.getElementById("transitionStatus");
const playButton = document.getElementById("transitionPlayButton");

transitionAudio.src = payload.src;

function playTransitionAudio() {
  status.textContent = "正在播放小顾提示。";
  return transitionAudio.play().then(function () {
    playButton.style.display = "none";
    status.textContent = "小顾正在读下一步提示。";
  }).catch(function () {
    playButton.style.display = "block";
    status.textContent = "浏览器阻止了自动播放，请点一下播放小顾提示。";
  });
}

transitionAudio.addEventListener("ended", function () {
  playButton.textContent = "再听一遍";
  playButton.style.display = "block";
  status.textContent = "小顾提示播放完成。";
});

transitionAudio.addEventListener("error", function () {
  playButton.style.display = "none";
  status.textContent = "小顾提示语音播放失败，可继续按文字提示操作。";
});

playButton.addEventListener("click", function () {
  transitionAudio.currentTime = 0;
  playTransitionAudio();
});

window.setTimeout(playTransitionAudio, 220);
</script>
""".replace("__PAYLOAD_JSON__", _safe_json(payload))
    components.html(player_html, height=142)


def _current_recording_key() -> str:
    question = st.session_state.elder_current_question.strip()
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:10]
    turn_index = len(st.session_state.elder_turns)
    attempt = int(st.session_state.get("elder_voice_attempt", 0))
    return f"elder_auto_{turn_index}_{attempt}_{digest}"


def _render_auto_voice_recorder() -> Any:
    audio_url, mime_type = _tts_audio_source(st.session_state.get("elder_tts_result"))
    auto_start = bool(st.session_state.get("elder_recorder_auto_start", False))
    st.session_state.elder_recorder_auto_start = False
    return elder_voice_recorder(
        question=st.session_state.elder_current_question,
        question_audio_url=audio_url,
        question_audio_mime_type=mime_type,
        recording_key=_current_recording_key(),
        auto_start=auto_start,
        silence_ms=ELDER_SILENCE_MS,
        max_recording_ms=ELDER_MAX_RECORDING_MS,
        no_speech_timeout_ms=ELDER_NO_SPEECH_TIMEOUT_MS,
        min_recording_ms=ELDER_MIN_RECORDING_MS,
        silence_threshold=ELDER_SILENCE_THRESHOLD,
        default=None,
        key=f"elder_voice_recorder_{_current_recording_key()}",
    )


def _generate_and_save_report() -> None:
    if not st.session_state.elder_turns:
        st.session_state.elder_save_status = "还没有回答记录，暂时不能生成评估。"
        return
    report = evaluate_dialogue(_flatten_turns(st.session_state.elder_turns), config=config)
    st.session_state.elder_report = report
    try:
        existing_record = find_assessment_record(
            st.session_state.get("elder_current_assessment_id"),
            user_id=current_user_id,
        )
        record = build_dialog_assessment_record(
            report,
            user_id=current_user_id,
            assessment_id=st.session_state.get("elder_current_assessment_id"),
            existing_record=existing_record,
        )
        save_session(record)
        save_session_memory(record)
        st.session_state.elder_current_assessment_id = record["assessment_id"]
        st.session_state.current_assessment_user_id = current_user_id
        st.session_state.elder_save_status = (
            f"已保存到{current_display_name}的本次综合评估，保存时间："
            f"{format_session_time(record['created_at'])}"
        )
    except Exception as error:
        st.session_state.elder_save_status = f"保存失败：{error}"


def _sync_elder_assessment_context_for_staff() -> None:
    assessment_id = st.session_state.get("elder_current_assessment_id")
    if isinstance(assessment_id, str) and assessment_id.strip():
        st.session_state.current_assessment_id = assessment_id.strip()
        st.session_state.current_assessment_user_id = current_user_id


def _prepare_clock_entry_from_elder() -> None:
    assessment_id = st.session_state.get("elder_current_assessment_id")
    if isinstance(assessment_id, str) and assessment_id.strip():
        st.session_state.current_assessment_id = assessment_id.strip()
    st.session_state.current_assessment_user_id = current_user_id
    st.session_state.clock_entry_from_elder = True
    st.session_state.clock_capture_mode = "camera"
    st.session_state.clock_report = None
    st.session_state.clock_save_status = ""
    st.session_state.clock_sample_path = None
    st.session_state.clock_last_analyzed_hash = ""
    st.session_state.clock_auto_saved_hash = ""
    st.session_state.clock_auto_save_attempted_hash = ""
    st.session_state.clock_last_invalid_hash = ""


def _next_step_view(next_task: dict[str, Any]) -> dict[str, str]:
    step_id = str(next_task.get("step_id", "")).strip()
    if step_id == FLOW_STEP_CLOCK_TEST:
        return {
            "elder_message": "我们再做一个小小游戏，好吗？请您在纸上画一个钟，指到 11 点 10 分。画好后拍张照片就可以，不着急，慢慢来。",
            "staff_page": "pages/2_画钟测试.py",
            "staff_action_label": "继续画钟测试",
            "staff_hint": "进入画钟拍照页；保存画钟结果时会优先合并到本轮长者访谈记录。",
        }
    if step_id == FLOW_STEP_FINISH:
        return {
            "elder_message": "本次访谈可以先结束。",
            "staff_page": "pages/3_认知简报.py",
            "staff_action_label": "查看认知简报",
            "staff_hint": "工作人员可查看最近报告、趋势和家属端提醒。",
        }
    return {
        "elder_message": str(next_task.get("elder_message", "结果已经整理好，请交给家人查看。")),
        "staff_page": "pages/3_认知简报.py",
        "staff_action_label": "查看认知简报",
        "staff_hint": str(next_task.get("staff_message", "可进入认知简报查看结果。")),
    }


_init_state()
if st.session_state.get("current_assessment_user_id") not in (None, current_user_id):
    st.session_state.elder_current_assessment_id = None
    st.session_state.current_assessment_id = None
st.session_state.current_assessment_user_id = current_user_id

if st.session_state.get("elder_autostart_requested"):
    st.session_state.elder_autostart_requested = False
    with st.spinner("正在准备问题，请稍等……"):
        _start_interview()

if not st.session_state.elder_started:
    history_hint = _history_hint()
    safe_history_hint = html.escape(history_hint)
    st.markdown(
        f"""
<div class="elder-start-screen">
  <div class="elder-title">和小顾聊天</div>
  <div class="elder-note">{safe_history_hint}</div>
  <div class="elder-soft">听问题，说回答。</div>
  <div class="elder-soft">只保存文字，不保存录音。</div>
  <div class="elder-device-top">
    {status_pill_html("麦克风会在开始后请求授权", "blue")}
    {status_pill_html("不是考试，慢慢说", "green")}
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    if st.button(
        "我准备好了，开始聊天",
        type="primary",
        width="stretch",
        key="elder_manual_start_button",
    ):
        with st.spinner("正在准备问题，请稍等……"):
            _start_interview()
        _rerun()
    st.stop()

if _interview_is_complete():
    if st.session_state.elder_report is None and st.session_state.elder_turns:
        with st.spinner("正在整理结果，请稍等……"):
            _generate_and_save_report()
    flow_summary = build_assessment_flow_summary(st.session_state.elder_report or {})
    next_task = flow_summary["next_task"]
    _sync_elder_assessment_context_for_staff()
    next_step_view = _next_step_view(next_task)
    safe_next_title = html.escape(next_task["title"])
    safe_next_message = html.escape(next_step_view["elder_message"])
    safe_staff_hint = html.escape(next_step_view["staff_hint"])
    with st.spinner("正在准备小顾的下一步提示语音……"):
        _ensure_transition_voice(next_step_view["elder_message"])

    complete_columns = st.columns([1.15, 0.85], gap="large", vertical_alignment="top")
    with complete_columns[0]:
        st.markdown(
            f"""
<div class="elder-complete-screen elder-complete-primary">
  <div class="elder-device-top">
    {status_pill_html("访谈已完成", "green")}
    {status_pill_html("结果已保存为技术原型记录", "blue")}
  </div>
  <div class="elder-done">今天的访谈完成了，谢谢您。</div>
  <div class="elder-soft">{safe_next_title}</div>
  <div class="elder-next-main">{safe_next_message}</div>
</div>
""",
            unsafe_allow_html=True,
        )
    with complete_columns[1]:
        st.markdown(
            f"""
<div class="elder-complete-action-panel">
  <div class="elder-action-title">工作人员下一步</div>
  <div class="elder-action-copy">{safe_staff_hint}</div>
  <div class="elder-transition-audio-note">小顾会把左侧提示读出来，方便老人知道接下来要做什么。</div>
</div>
""",
            unsafe_allow_html=True,
        )
        transition_audio_src, transition_mime_type = _tts_audio_source(
            st.session_state.get("elder_transition_tts_result")
        )
        if transition_audio_src:
            _render_transition_audio_player(
                transition_audio_src,
                transition_mime_type,
                next_step_view["elder_message"],
            )
        elif st.session_state.elder_transition_tts_status:
            st.caption(st.session_state.elder_transition_tts_status)
        if next_task.get("step_id") == FLOW_STEP_CLOCK_TEST:
            if st.button(
                "继续画钟拍照",
                type="primary",
                width="stretch",
                key="elder_clock_next_button",
            ):
                _prepare_clock_entry_from_elder()
                st.switch_page("pages/2_画钟测试.py")
            st.page_link("pages/3_认知简报.py", label="稍后查看认知简报")
        else:
            st.page_link(
                next_step_view["staff_page"],
                label=next_step_view["staff_action_label"],
            )
    if st.session_state.elder_save_status:
        if str(st.session_state.elder_save_status).startswith("已保存"):
            st.success(st.session_state.elder_save_status)
        else:
            st.warning(st.session_state.elder_save_status)
    st.caption(DISCLAIMER)
    st.stop()

completed_count = len(_covered_domains())
progress_text = f"第 {completed_count + 1} 题，共 {len(COGNITIVE_DOMAINS)} 题"
st.progress(completed_count / len(COGNITIVE_DOMAINS), text=progress_text)

safe_status = html.escape(st.session_state.elder_tts_status or st.session_state.elder_status)
safe_question = html.escape(st.session_state.elder_current_question)
last_transcript = str(st.session_state.get("elder_last_transcript", "")).strip()
last_transcript_status = str(
    st.session_state.get("elder_last_transcript_status", "")
).strip()
safe_last_transcript = html.escape(last_transcript)
safe_last_transcript_status = html.escape(last_transcript_status)
st.markdown(
    f"""
<div class="elder-live-header">
  <div class="elder-device-top">
    {status_pill_html(progress_text, "green")}
    {status_pill_html("正在进行长者语音访谈", "blue")}
    {status_pill_html("只保存文字", "neutral")}
  </div>
</div>
""",
    unsafe_allow_html=True,
)

live_columns = st.columns([1.05, 0.95], gap="large", vertical_alignment="top")
with live_columns[0]:
    st.markdown(
        f"""
<div class="elder-live-question-panel">
  <div class="elder-live-kicker">小顾正在问</div>
  <div class="elder-big">{safe_status}</div>
  <div class="elder-question">{safe_question}</div>
  <div class="elder-live-brief">听完以后，直接说出您的回答。</div>
  <div class="elder-live-privacy">只保存文字，不保存录音。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    if last_transcript or last_transcript_status:
        transcript_body = (
            safe_last_transcript
            if last_transcript
            else "暂时没有识别到清晰文字，可以点右侧按钮再试一次。"
        )
        transcript_status = safe_last_transcript_status or "上一题已经识别并记录。"
        st.markdown(
            f"""
<div class="elder-transcript-card">
  <div class="elder-transcript-label">上一题识别文字</div>
  <div class="elder-transcript-text">{transcript_body}</div>
  <div class="elder-transcript-status">{transcript_status}</div>
</div>
""",
            unsafe_allow_html=True,
        )

with live_columns[1]:
    st.markdown(
        """
<div class="elder-voice-station">
  <div class="elder-live-kicker">语音感应</div>
  <div class="elder-voice-title">录音状态</div>
  <div class="elder-voice-copy">页面会显示“正在听您回答”，听到停顿后会自动停止录音。</div>
</div>
""",
        unsafe_allow_html=True,
    )
    recorder_result = _render_auto_voice_recorder()
    if recorder_result is not None:
        with st.spinner("正在识别，请稍等……"):
            processed = _process_auto_recorder_result(recorder_result)
        if processed:
            _rerun()

if st.session_state.elder_status:
    safe_elder_status = html.escape(st.session_state.elder_status)
    st.markdown(
        f"""<div class="elder-soft">{safe_elder_status}</div>""",
        unsafe_allow_html=True,
    )
