import json
from html import escape

import streamlit as st
import streamlit.components.v1 as components
from typing import Optional

from core.asr_client import transcribe_audio
from core.assessment_flow import FLOW_STEP_CLOCK_TEST, build_assessment_flow_summary
from core.config import is_tts_config_complete, load_config
from core.db import save_session
from core.llm_client import evaluate_dialogue, generate_next_question
from core.memory import save_session_memory
from core.mock_data import (
    INTERVIEW_COMPLETED_MESSAGE,
    PRESET_INTERVIEW_QUESTIONS,
    all_dialog_domains_covered,
    get_covered_dialog_domains,
    get_dialog_example_answers,
    get_next_preset_interview_question,
    infer_dialog_question_type,
    normalize_dialog_domains,
)
from core.report import format_session_time
from core.schemas import (
    COGNITIVE_DOMAINS,
    DISCLAIMER,
    DOMAIN_LABELS,
    display_risk_level,
    display_source,
)
from core.session_history import (
    build_dialog_assessment_record,
    find_assessment_record,
    get_current_user_profile,
)
from core.staff_gate import hide_sidebar_nav, render_staff_gate
from core.tts_client import prepare_text_for_tts, synthesize_speech
from core.ui import (
    callout_html,
    chip_html,
    display_model_name,
    evidence_card_html,
    inject_staff_theme,
    metric_card_html,
    risk_badge_html,
    section_header_html,
    status_strip_html,
)


SYSTEM_VOICE_MODE_HIGH_QUALITY = "高质量 TTS"
SYSTEM_VOICE_MODE_BROWSER_FAST = "快速朗读（浏览器）"
SYSTEM_VOICE_MODES = [
    SYSTEM_VOICE_MODE_HIGH_QUALITY,
    SYSTEM_VOICE_MODE_BROWSER_FAST,
]
DEFAULT_SYSTEM_VOICE_MODE_INDEX = 0
EARLY_CLOCK_TRIGGER_MIN_VAGUE_TURNS = 2
EARLY_CLOCK_TRIGGER_DOMAINS = {"executive_function", "visuospatial"}
CLOCK_TRANSITION_MESSAGE = (
    "我们再做一个小小游戏，好吗？请您在纸上画一个钟，指到 11 点 10 分。"
    "画好后拍张照片就可以，不着急，慢慢来。"
)


st.set_page_config(page_title="快捷访谈评估", layout="wide")
config = load_config()
hide_sidebar_nav()
inject_staff_theme()
render_staff_gate(config)

st.title("快捷访谈评估")
st.write("工作人员可在这里用预选回答、文字或语音快速跑通“小顾”访谈，并同步看到画钟或简报下一步。")

current_user = get_current_user_profile(st.session_state)
current_user_id = current_user["user_id"]
current_display_name = current_user["display_name"]
llm_config_complete = all(
    [
        config.llm_base_url.strip(),
        config.llm_api_key.strip(),
        config.llm_model.strip(),
    ]
)
asr_config_complete = all(
    [
        config.asr_base_url.strip(),
        config.asr_api_key.strip(),
        config.asr_model.strip(),
    ]
)
tts_config_complete = is_tts_config_complete(config)

st.markdown(
    section_header_html(
        "工作人员控制台",
        eyebrow="Interview Console",
        body="用于快速复刻老人端访谈主链路：预选回答或文字输入可跳过语音，评估后同步给出画钟拍照或简报入口。",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    status_strip_html(
        [
            {"label": "当前对象", "value": current_display_name, "tone": "green"},
            {"label": "LLM 配置", "value": "完整" if llm_config_complete else "不完整", "tone": "blue"},
            {"label": "ASR 配置", "value": "完整" if asr_config_complete else "不完整", "tone": "amber"},
            {"label": "TTS 配置", "value": "完整" if tts_config_complete else "不完整", "tone": "neutral"},
        ]
    ),
    unsafe_allow_html=True,
)

with st.expander("系统配置状态", expanded=False):
    status_columns = st.columns(4)
    status_columns[0].metric("DEMO_MODE", str(config.demo_mode).lower())
    status_columns[1].metric("LLM_MODEL", display_model_name(config.llm_model))
    status_columns[2].metric("LLM 配置", "完整" if llm_config_complete else "不完整")
    status_columns[3].metric(
        "API Key",
        "已配置" if config.llm_api_key.strip() else "未配置",
    )

if "dialog_turns" not in st.session_state:
    st.session_state.dialog_turns = []

if "current_question" not in st.session_state:
    st.session_state.current_question = PRESET_INTERVIEW_QUESTIONS[0]["question"]

if "current_question_domain" not in st.session_state:
    st.session_state.current_question_domain = PRESET_INTERVIEW_QUESTIONS[0]["domain"]

if "covered_domains" not in st.session_state:
    st.session_state.covered_domains = []

if "next_question_status" not in st.session_state:
    st.session_state.next_question_status = ""

if "dialog_report" not in st.session_state:
    st.session_state.dialog_report = None

if "dialog_save_status" not in st.session_state:
    st.session_state.dialog_save_status = ""

if "dialog_saved_report_signature" not in st.session_state:
    st.session_state.dialog_saved_report_signature = ""

if "answer_draft" not in st.session_state:
    st.session_state.answer_draft = ""

if "asr_result" not in st.session_state:
    st.session_state.asr_result = None

if "asr_status" not in st.session_state:
    st.session_state.asr_status = ""

if "tts_result" not in st.session_state:
    st.session_state.tts_result = None

if "tts_status" not in st.session_state:
    st.session_state.tts_status = ""

if "current_question_source" not in st.session_state:
    st.session_state.current_question_source = "preset"

if "current_question_source_reason" not in st.session_state:
    st.session_state.current_question_source_reason = ""

if "current_sample_answers" not in st.session_state:
    st.session_state.current_sample_answers = get_dialog_example_answers(
        st.session_state.current_question,
        target_domain=st.session_state.current_question_domain,
    )

if "current_assessment_id" not in st.session_state:
    st.session_state.current_assessment_id = None

if st.session_state.get("current_assessment_user_id") not in (None, current_user_id):
    st.session_state.current_assessment_id = None
st.session_state.current_assessment_user_id = current_user_id


def _clear_dialog() -> None:
    st.session_state.dialog_turns = []
    st.session_state.covered_domains = []
    _set_current_question_from_preset()
    st.session_state.dialog_report = None
    st.session_state.dialog_save_status = ""
    st.session_state.dialog_saved_report_signature = ""
    st.session_state.next_question_status = ""
    st.session_state.answer_draft = ""
    st.session_state.asr_result = None
    st.session_state.asr_status = ""
    _reset_tts_playback_state()
    st.session_state.current_question_source = "preset"
    st.session_state.current_question_source_reason = ""
    st.session_state.current_sample_answers = get_dialog_example_answers(
        st.session_state.current_question,
        target_domain=st.session_state.current_question_domain,
    )


def _prepare_next_question() -> None:
    if _interview_is_complete():
        _set_interview_completed()
        return

    _mark_current_question_covered()
    _set_next_interview_question()
    st.session_state.dialog_report = None


def _submit_answer(answer_text: str) -> None:
    answer = answer_text.strip()
    if not answer:
        return

    if _interview_is_complete():
        st.session_state.next_question_status = "主要认知域已覆盖，可以生成评估并判断下一步。"
        return

    question = st.session_state.current_question.strip()
    if not question:
        _set_current_question_from_preset()
        question = st.session_state.current_question.strip()

    st.session_state.dialog_turns.append(
        {
            "assistant": question,
            "user": answer,
            "target_domain": st.session_state.current_question_domain,
        }
    )
    _mark_current_question_covered()
    _set_next_interview_question()
    st.session_state.dialog_report = None
    st.session_state.dialog_save_status = ""
    st.session_state.dialog_saved_report_signature = ""
    st.session_state.answer_draft = ""


def _submit_manual_answer() -> None:
    _submit_answer(st.session_state.get("answer_draft", ""))


def _transcribe_answer_audio(audio_file) -> None:
    if audio_file is None:
        st.session_state.asr_result = None
        st.session_state.asr_status = "请先录音或上传音频，再进行 ASR 转写。"
        return

    result = transcribe_audio(
        audio=audio_file,
        filename=getattr(audio_file, "name", None),
        config=config,
    )
    st.session_state.asr_result = result
    text = result.get("text", "").strip()
    metadata = result.get("metadata", {})
    source = metadata.get("source", "unknown")
    reason = metadata.get("reason", "")
    if text:
        st.session_state.asr_status = f"ASR 转写完成，结果来源：{_display_asr_source(source)}。"
    else:
        st.session_state.asr_status = reason or "ASR 未生成有效文本，请重试或手动输入。"


def _fill_answer_from_asr() -> None:
    text = _current_asr_text()
    if not text:
        st.session_state.asr_status = "当前没有可填入的 ASR 转写文本。"
        return
    st.session_state.answer_draft = text
    st.session_state.asr_status = "已将 ASR 转写文本填入老人回答框，尚未提交。"


def _submit_answer_from_asr() -> None:
    text = _current_asr_text()
    if not text:
        st.session_state.asr_status = "当前没有可提交的 ASR 转写文本。"
        return
    _submit_answer(text)
    st.session_state.asr_status = "已将 ASR 转写文本提交为当前老人回答。"


def _current_asr_text() -> str:
    result = st.session_state.get("asr_result") or {}
    return str(result.get("text", "")).strip()


def _display_asr_source(source: str) -> str:
    labels = {
        "asr-api": "真实 ASR",
        "mock": "模拟 ASR",
        "fallback": "ASR 兜底",
    }
    return labels.get(source, source or "未知")


def _reset_tts_playback_state() -> None:
    st.session_state.tts_result = None
    st.session_state.tts_status = ""


def _synthesize_current_question() -> None:
    question = st.session_state.current_question.strip()
    if not question:
        st.session_state.tts_result = None
        st.session_state.tts_status = "当前没有可播放的问题。"
        return

    result = synthesize_speech(question, config=config, prefer_remote_url=True)
    st.session_state.tts_result = result
    metadata = result.get("metadata", {})
    source = metadata.get("source", "unknown")
    if _result_has_audio(result):
        if metadata.get("cached"):
            st.session_state.tts_status = "已使用缓存音频，可直接播放。"
        elif metadata.get("cache_path"):
            st.session_state.tts_status = "系统语音已生成。当前采用非实时高质量 TTS，首次生成会稍慢。"
        else:
            st.session_state.tts_status = f"当前问题语音已生成，来源：{_display_tts_source(source)}。"
    else:
        st.session_state.tts_status = "当前未生成真实音频，可继续使用文本问题和语音回答流程。"


def _result_has_audio(result: object) -> bool:
    if not isinstance(result, dict):
        return False
    audio_bytes = result.get("audio_bytes")
    if isinstance(audio_bytes, (bytes, bytearray)) and audio_bytes:
        return True
    audio_url = str(result.get("audio_url", "") or "").strip().lower()
    return audio_url.startswith(("http://", "https://"))


def _current_tts_audio() -> tuple[Optional[object], str]:
    result = st.session_state.get("tts_result") or {}
    audio_bytes = result.get("audio_bytes")
    mime_type = str(result.get("mime_type", "audio/mpeg"))
    if isinstance(audio_bytes, (bytes, bytearray)) and audio_bytes:
        return bytes(audio_bytes), mime_type
    audio_url = str(result.get("audio_url", "") or "").strip()
    if audio_url.lower().startswith(("http://", "https://")):
        return audio_url, mime_type
    return None, mime_type


def _display_tts_source(source: str) -> str:
    labels = {
        "tts": "真实 TTS",
        "tts_url": "真实 TTS URL",
        "tts_cache": "TTS 缓存",
        "mock": "模拟 TTS",
        "fallback": "TTS 兜底",
    }
    return labels.get(source, source or "未知")


def _quick_clock_signal(turns: list[dict[str, str]]) -> dict[str, object]:
    vague_count = 0
    triggered_domains: list[str] = []
    for turn in turns:
        answer = str(turn.get("user", "")).strip()
        if not _answer_suggests_clock_check(answer):
            continue
        vague_count += 1
        domain = str(turn.get("target_domain", "")).strip()
        if domain not in COGNITIVE_DOMAINS:
            domain = infer_dialog_question_type(str(turn.get("assistant", "")))
        if domain in COGNITIVE_DOMAINS and domain not in triggered_domains:
            triggered_domains.append(domain)

    critical_domains = [
        domain for domain in triggered_domains if domain in EARLY_CLOCK_TRIGGER_DOMAINS
    ]
    recommended = (
        vague_count >= EARLY_CLOCK_TRIGGER_MIN_VAGUE_TURNS
        or bool(critical_domains)
    )
    reasons: list[str] = []
    if vague_count >= EARLY_CLOCK_TRIGGER_MIN_VAGUE_TURNS:
        reasons.append(f"已出现 {vague_count} 轮模糊、不确定或跑题回答。")
    for domain in critical_domains:
        reasons.append(f"{DOMAIN_LABELS.get(domain, domain)}相关回答不够稳定。")
    return {
        "recommended": recommended,
        "vague_count": vague_count,
        "triggered_domains": triggered_domains,
        "reasons": reasons,
    }


def _answer_suggests_clock_check(answer: str) -> bool:
    vague_keywords = (
        "不太清楚",
        "不清楚",
        "不知道",
        "说不好",
        "说不上",
        "想不起来",
        "拿不准",
        "不确定",
        "记不住",
        "弄混",
        "混了",
        "分不清",
        "不太想算",
        "不会",
        "差不多",
        "看情况",
    )
    return any(keyword in answer for keyword in vague_keywords)


def _render_example_answer_buttons(
    example_answers: dict[str, str],
    *,
    key_prefix: str,
) -> None:
    st.caption("点击任一预选回答会直接提交当前问题，并自动进入下一问。")
    example_columns = st.columns(3)
    with example_columns[0]:
        st.button(
            "正常回答",
            key=f"{key_prefix}_normal_example_answer",
            on_click=_submit_answer,
            args=(example_answers["normal"],),
            width="stretch",
        )
        st.caption(example_answers["normal"])
    with example_columns[1]:
        st.button(
            "轻度异常回答",
            key=f"{key_prefix}_mild_example_answer",
            on_click=_submit_answer,
            args=(example_answers["mild_decline"],),
            width="stretch",
        )
        st.caption(example_answers["mild_decline"])
    with example_columns[2]:
        st.button(
            "模糊/跑题回答",
            key=f"{key_prefix}_vague_example_answer",
            on_click=_submit_answer,
            args=(example_answers["vague"],),
            width="stretch",
        )
        st.caption(example_answers["vague"])


def _render_browser_speech_button(text: str) -> None:
    speech_text = prepare_text_for_tts(str(text or "").strip())
    if not speech_text:
        st.warning("当前没有可朗读的问题。")
        return

    speech_text_json = (
        json.dumps(speech_text, ensure_ascii=False)
        .replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    speech_html = """
<button
  id="cogniguardBrowserSpeechButton"
  type="button"
  style="
    width: 100%;
    min-height: 42px;
    border: 1px solid #d1d5db;
    border-radius: 6px;
    background: #ffffff;
    color: #111827;
    font-size: 15px;
    cursor: pointer;
  "
>
  快速朗读当前问题
</button>
<div
  id="cogniguardBrowserSpeechStatus"
  style="
    margin-top: 8px;
    color: #4b5563;
    font-size: 13px;
    line-height: 1.5;
  "
>
  请点击按钮开始浏览器快速朗读。
</div>
<script>
const speechText = __SPEECH_TEXT__;
const speakButton = document.getElementById("cogniguardBrowserSpeechButton");
const speechStatus = document.getElementById("cogniguardBrowserSpeechStatus");

speakButton.addEventListener("click", function () {
  if (!("speechSynthesis" in window) || typeof SpeechSynthesisUtterance === "undefined") {
    speechStatus.textContent = "当前浏览器不支持快速朗读，请继续使用文本问题或高质量 TTS。";
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(speechText);
  utterance.lang = "zh-CN";
  utterance.rate = 0.9;
  utterance.volume = 1.0;

  const voices = window.speechSynthesis.getVoices();
  const chineseVoice = voices.find(function (voice) {
    return voice.lang && voice.lang.toLowerCase().startsWith("zh");
  });
  if (chineseVoice) {
    utterance.voice = chineseVoice;
  }

  utterance.onend = function () {
    speechStatus.textContent = "浏览器快速朗读已结束。";
  };
  utterance.onerror = function () {
    speechStatus.textContent = "浏览器快速朗读失败，请继续使用文本问题或高质量 TTS。";
  };

  window.speechSynthesis.speak(utterance);
  speechStatus.textContent = "正在使用浏览器本地语音朗读当前问题。";
});
</script>
""".replace("__SPEECH_TEXT__", speech_text_json)
    components.html(speech_html, height=110)


def _ensure_current_question() -> None:
    if not st.session_state.current_question:
        _set_current_question_from_preset()


def _set_next_interview_question() -> None:
    question, domain, source, reason, sample_answers = _generate_next_interview_question()
    st.session_state.current_question = question
    st.session_state.current_question_domain = domain
    st.session_state.current_question_source = source
    st.session_state.current_question_source_reason = reason
    st.session_state.current_sample_answers = sample_answers
    _reset_tts_playback_state()
    if question == INTERVIEW_COMPLETED_MESSAGE:
        st.session_state.next_question_status = "主要认知域已覆盖，可以生成评估并判断下一步。"
    elif source == "qwen":
        st.session_state.next_question_status = "下一问由小顾根据当前对话生成。"
    elif source == "preset":
        st.session_state.next_question_status = "下一问使用小顾风格的预设生活化问题。"
    elif source == "fallback":
        st.session_state.next_question_status = "下一问生成失败，已使用预设问题兜底。"
    else:
        st.session_state.next_question_status = ""


def _generate_next_interview_question() -> tuple[str, str, str, str, dict[str, str]]:
    covered = _covered_domains()
    if all_dialog_domains_covered(covered):
        return INTERVIEW_COMPLETED_MESSAGE, "", "preset", "主要认知域已覆盖", {}

    fallback_reason = ""
    try:
        result = generate_next_question(
            _flatten_dialog_turns(st.session_state.dialog_turns),
            config=config,
            covered_domains=covered,
        )
        question = result.get("question") if isinstance(result, dict) else ""
        metadata = result.get("metadata", {}) if isinstance(result, dict) else {}
        source = metadata.get("source", "")
        fallback_reason = metadata.get("reason", "")
        target_domain = result.get("target_domain", "") if isinstance(result, dict) else ""
        if (
            question
            and source == "qwen"
            and target_domain in COGNITIVE_DOMAINS
            and target_domain not in covered
        ):
            sample_answers = get_dialog_example_answers(
                question,
                target_domain=target_domain,
                sample_answers=result.get("sample_answers"),
            )
            return question, target_domain, "qwen", fallback_reason, sample_answers
        if (
            question
            and source == "mock"
            and target_domain in COGNITIVE_DOMAINS
            and target_domain not in covered
        ):
            sample_answers = get_dialog_example_answers(
                question,
                target_domain=target_domain,
                sample_answers=result.get("sample_answers"),
            )
            return question, target_domain, "preset", fallback_reason, sample_answers
    except Exception as error:
        fallback_reason = f"下一问生成异常：{type(error).__name__}"
    preset = get_next_preset_interview_question(covered)
    if preset is None:
        return INTERVIEW_COMPLETED_MESSAGE, "", "fallback", fallback_reason, {}
    sample_answers = get_dialog_example_answers(
        preset["question"],
        target_domain=preset["domain"],
    )
    return (
        preset["question"],
        preset["domain"],
        "preset",
        fallback_reason or "使用预设问题",
        sample_answers,
    )


def _set_current_question_from_preset() -> None:
    preset = get_next_preset_interview_question(_covered_domains())
    if preset is None:
        _set_interview_completed()
        return
    st.session_state.current_question = preset["question"]
    st.session_state.current_question_domain = preset["domain"]
    st.session_state.current_question_source = "preset"
    st.session_state.current_question_source_reason = "预设问题"
    st.session_state.current_sample_answers = get_dialog_example_answers(
        preset["question"],
        target_domain=preset["domain"],
    )
    _reset_tts_playback_state()


def _set_interview_completed() -> None:
    st.session_state.current_question = INTERVIEW_COMPLETED_MESSAGE
    st.session_state.current_question_domain = ""
    st.session_state.current_question_source = "preset"
    st.session_state.current_question_source_reason = "主要认知域已覆盖"
    st.session_state.current_sample_answers = {}
    st.session_state.next_question_status = "主要认知域已覆盖，可以生成评估并判断下一步。"
    _reset_tts_playback_state()


def _mark_current_question_covered() -> None:
    domain = st.session_state.current_question_domain or infer_dialog_question_type(
        st.session_state.current_question
    )
    if domain in COGNITIVE_DOMAINS:
        covered = normalize_dialog_domains(st.session_state.covered_domains)
        if domain not in covered:
            covered.append(domain)
        st.session_state.covered_domains = covered


def _covered_domains() -> list[str]:
    stored = normalize_dialog_domains(st.session_state.covered_domains)
    inferred = get_covered_dialog_domains(st.session_state.dialog_turns)
    return normalize_dialog_domains([*stored, *inferred])


def _interview_is_complete() -> bool:
    return (
        all_dialog_domains_covered(_covered_domains())
        or st.session_state.current_question == INTERVIEW_COMPLETED_MESSAGE
    )


def _flatten_dialog_turns(turns: list[dict[str, str]]) -> list[str]:
    messages = []
    for turn in turns:
        assistant = turn.get("assistant", "").strip()
        user = turn.get("user", "").strip()
        if assistant:
            messages.append(f"AI访谈问题：{assistant}")
        if user:
            messages.append(f"老人回答：{user}")
    return messages


def _report_signature(report: dict) -> str:
    try:
        return json.dumps(report, ensure_ascii=False, sort_keys=True, default=str)
    except TypeError:
        return str(report)


def _save_dialog_report(report: dict, *, force: bool = False) -> bool:
    signature = _report_signature(report)
    if (
        not force
        and signature
        and st.session_state.dialog_saved_report_signature == signature
        and isinstance(st.session_state.get("current_assessment_id"), str)
    ):
        return True

    try:
        existing_record = find_assessment_record(
            st.session_state.get("current_assessment_id"),
            user_id=current_user_id,
        )
        record = build_dialog_assessment_record(
            report,
            user_id=current_user_id,
            assessment_id=st.session_state.get("current_assessment_id"),
            existing_record=existing_record,
        )
        save_session(record)
        save_session_memory(record)
        st.session_state.current_assessment_id = record["assessment_id"]
        st.session_state.current_assessment_user_id = current_user_id
        st.session_state.dialog_saved_report_signature = signature
        st.session_state.dialog_save_status = (
            f"已保存到{current_display_name}的本次综合评估，保存时间："
            f"{format_session_time(record['created_at'])}"
        )
        return True
    except Exception as error:
        st.session_state.dialog_save_status = f"保存失败：{error}"
        return False


def _generate_and_save_dialog_report() -> Optional[dict]:
    if not st.session_state.dialog_turns:
        st.session_state.dialog_save_status = "还没有回答记录，暂时不能生成评估。"
        return None
    report = evaluate_dialogue(
        _flatten_dialog_turns(st.session_state.dialog_turns),
        config=config,
    )
    st.session_state.dialog_report = report
    st.session_state.dialog_saved_report_signature = ""
    _save_dialog_report(report)
    return report


def _prepare_clock_entry_from_quick_interview(report: dict) -> bool:
    if not _save_dialog_report(report):
        return False
    st.session_state.clock_entry_from_elder = True
    st.session_state.clock_capture_mode = "camera"
    st.session_state.clock_report = None
    st.session_state.clock_save_status = ""
    st.session_state.clock_sample_path = None
    st.session_state.clock_last_analyzed_hash = ""
    st.session_state.clock_auto_saved_hash = ""
    st.session_state.clock_auto_save_attempted_hash = ""
    st.session_state.clock_last_invalid_hash = ""
    st.session_state.clock_current_input_hash = ""
    return True


st.markdown("### 快捷访谈流程")
st.markdown(
    section_header_html(
        "流程控制",
        eyebrow="Control",
        body="预选回答和文字输入是本页主流程，可快速跳过语音；语音录入保留为可选验证。",
    ),
    unsafe_allow_html=True,
)
control_columns = st.columns(2)
with control_columns[0]:
    st.button("生成下一问", on_click=_prepare_next_question)

with control_columns[1]:
    if st.button("生成评估并判断下一步", type="primary"):
        _generate_and_save_dialog_report()

_ensure_current_question()

st.markdown("### 小顾 · AI 访谈问题")
if _interview_is_complete():
    st.success(INTERVIEW_COMPLETED_MESSAGE)
else:
    st.markdown(
        callout_html("当前问题", st.session_state.current_question, tone="blue"),
        unsafe_allow_html=True,
    )

covered_labels = [
    DOMAIN_LABELS[domain] for domain in _covered_domains() if domain in DOMAIN_LABELS
]
st.caption(
    "已覆盖认知域："
    + ("、".join(covered_labels) if covered_labels else "暂无")
)
if st.session_state.next_question_status:
    st.caption(st.session_state.next_question_status)

quick_clock_signal = _quick_clock_signal(st.session_state.dialog_turns)
if (
    quick_clock_signal["recommended"]
    and _interview_is_complete()
    and not isinstance(st.session_state.get("dialog_report"), dict)
):
    st.markdown("### 画钟建议记录")
    st.markdown(
        callout_html(
            "已记录到可能需要补充画钟",
            "本页不会在访谈中途跳转。请先继续完成当前访谈，生成评估后再按后续任务进入画钟拍照。",
            tone="amber",
        ),
        unsafe_allow_html=True,
    )
    for reason in quick_clock_signal["reasons"]:
        st.write(f"- {reason}")
    st.caption("演示和主流程演示时，画钟统一放在访谈之后，避免中途打断老人回答。")

if not _interview_is_complete():
    current_domain = st.session_state.current_question_domain or infer_dialog_question_type(
        st.session_state.current_question
    )
    st.markdown(
        "".join(
            [
                chip_html("下一问来源", display_source(st.session_state.current_question_source), "blue"),
                chip_html("当前覆盖域", DOMAIN_LABELS.get(current_domain, "待判断"), "green"),
                chip_html("已覆盖", "、".join(covered_labels) if covered_labels else "暂无", "neutral"),
            ]
        ),
        unsafe_allow_html=True,
    )
    st.caption(f"下一问来源：{display_source(st.session_state.current_question_source)}")
    if st.session_state.current_question_source == "preset" and st.session_state.current_question_source_reason:
        st.caption(f"来源说明：{st.session_state.current_question_source_reason}")
    st.caption(f"当前覆盖域：{DOMAIN_LABELS.get(current_domain, '待判断')}")

    example_answers = get_dialog_example_answers(
        st.session_state.current_question,
        target_domain=current_domain,
        sample_answers=st.session_state.current_sample_answers,
    )

    st.markdown("### 快捷预选回答")
    _render_example_answer_buttons(example_answers, key_prefix="quick")

    st.markdown("### 文字快速测试")
    st.caption("开发调试时可直接打字提交；提交后会立刻进入下一轮文字问题，不需要生成 TTS 或使用 ASR。")
    st.caption(
        "开发测试状态："
        f"来源 {display_source(st.session_state.current_question_source)}；"
        f"目标域 {DOMAIN_LABELS.get(current_domain, '待判断')}；"
        f"说明 {st.session_state.current_question_source_reason or '无'}。"
    )
    with st.expander("开发测试详情"):
        st.write(
            {
                "当前问题": st.session_state.current_question,
                "下一问来源": display_source(st.session_state.current_question_source),
                "目标认知域": DOMAIN_LABELS.get(current_domain, current_domain or "待判断"),
                "来源说明": st.session_state.current_question_source_reason or "无",
                "已覆盖认知域": covered_labels,
                "示例回答来源": "Qwen sample_answers 或本地规则校验后的结果",
            }
        )
    with st.form("dialog_text_quick_test_form"):
        st.text_area(
            "老人文字回答",
            key="answer_draft",
            placeholder=f"例如：{example_answers['normal']}",
            height=110,
        )
        st.form_submit_button("提交文字回答并生成下一问", on_click=_submit_manual_answer)

    st.markdown("### 系统语音播放（可选）")
    st.caption(
        "系统高质量语音：Qwen-TTS 合成，音质较好，首次可能较慢；"
        "系统快速朗读：浏览器本地语音，低延迟，不消耗 TTS API；"
        "用户语音回答：麦克风录音或上传音频，经 ASR 转文字。"
    )
    st.caption(
        "默认模式为高质量 TTS。本阶段使用非实时 Qwen-TTS，适合高质量语音演示；"
        "实时流式 TTS 可作为后续优化。"
    )
    system_voice_mode = st.radio(
        "系统语音模式",
        SYSTEM_VOICE_MODES,
        index=DEFAULT_SYSTEM_VOICE_MODE_INDEX,
        horizontal=True,
        key="system_voice_mode",
    )

    if system_voice_mode == SYSTEM_VOICE_MODE_HIGH_QUALITY:
        with st.expander("系统语音配置状态", expanded=False):
            tts_status_columns = st.columns(4)
            tts_status_columns[0].metric("TTS 配置", "完整" if tts_config_complete else "不完整")
            tts_status_columns[1].metric(
                "TTS API Key",
                "已配置" if config.tts_api_key.strip() else "未配置",
            )
            tts_status_columns[2].metric("TTS 模型", display_model_name(config.tts_model))
            tts_status_columns[3].metric("系统音色", config.tts_voice_assistant or "未配置")
        if st.button(
            "生成并播放系统语音",
            key="play_current_question",
            width="stretch",
        ):
            with st.spinner("正在生成系统语音，首次生成可能需要 10-20 秒，请稍候……"):
                _synthesize_current_question()

        tts_audio, tts_mime_type = _current_tts_audio()
        if tts_audio:
            st.success(st.session_state.tts_status or "当前问题语音已生成。")
            tts_metadata = (st.session_state.get("tts_result") or {}).get("metadata", {})
            st.caption(
                "TTS 结果来源："
                f"{_display_tts_source(tts_metadata.get('source', 'unknown'))}；"
                f"模型：{display_model_name(tts_metadata.get('model', '未配置'))}；"
                f"音色：{tts_metadata.get('voice', config.tts_voice_assistant or '未配置')}"
            )
            if tts_metadata.get("cache_path"):
                st.caption(f"缓存位置：{tts_metadata.get('cache_path')}")
            st.audio(tts_audio, format=tts_mime_type)
        elif st.session_state.tts_status:
            tts_metadata = (st.session_state.get("tts_result") or {}).get("metadata", {})
            st.warning("当前未生成真实音频，可继续使用文本问题和语音回答流程。")
            if tts_metadata.get("source"):
                st.caption(
                    "TTS 结果来源："
                    f"{_display_tts_source(tts_metadata.get('source', 'unknown'))}；"
                    f"模型：{display_model_name(tts_metadata.get('model', '未配置'))}；"
                    f"音色：{tts_metadata.get('voice', config.tts_voice_assistant or '未配置')}"
                )
            if tts_metadata.get("reason"):
                st.caption(f"原因：{tts_metadata.get('reason')}")
            st.caption(f"说明：{st.session_state.tts_status}")
    else:
        st.caption(
            "快速朗读使用浏览器本地语音，低延迟，但音色取决于浏览器/系统。"
            "该模式不调用 Qwen-TTS，不消耗 TTS API，不读取麦克风，也不保存音频。"
        )
        _render_browser_speech_button(st.session_state.current_question)

    st.markdown("### 语音回答")
    st.caption(
        "用户需要主动点击录音控件并授权麦克风；也可以上传音频文件。"
        "系统只把本段音频送去 ASR，页面不保存原始音频，只保存或提交转写文本。"
    )
    st.caption(
        "ASR 配置："
        + ("完整" if asr_config_complete else "不完整")
        + "；ASR API Key："
        + ("已配置" if config.asr_api_key.strip() else "未配置")
        + "。页面不会显示 API Key。"
    )
    voice_columns = st.columns(2)
    with voice_columns[0]:
        recorded_audio = None
        if hasattr(st, "audio_input"):
            recorded_audio = st.audio_input(
                "录音回答",
                key="voice_answer_recording",
                help="点击后浏览器会请求麦克风权限；停止录音后再点击转写。",
            )
        else:
            st.warning("当前 Streamlit 版本不支持 st.audio_input，请升级到 streamlit>=1.40。")
        if st.button(
            "转写录音",
            key="transcribe_recorded_audio",
            width="stretch",
        ):
            _transcribe_answer_audio(recorded_audio)
    with voice_columns[1]:
        uploaded_audio = st.file_uploader(
            "上传音频回答",
            type=["wav", "mp3", "m4a", "mp4", "webm", "ogg"],
            key="voice_answer_upload",
        )
        if st.button(
            "转写上传音频",
            key="transcribe_uploaded_audio",
            width="stretch",
        ):
            _transcribe_answer_audio(uploaded_audio)

    asr_text = _current_asr_text()
    if st.session_state.asr_status:
        if asr_text:
            st.success(st.session_state.asr_status)
        else:
            st.warning(st.session_state.asr_status)
    if st.session_state.asr_result:
        asr_metadata = st.session_state.asr_result.get("metadata", {})
        st.caption(
            "ASR 结果来源："
            f"{_display_asr_source(asr_metadata.get('source', 'unknown'))}；"
            f"模型：{display_model_name(asr_metadata.get('model', '未配置'))}"
        )
        if asr_metadata.get("reason"):
            st.caption(f"说明：{asr_metadata.get('reason')}")
    if asr_text:
        st.text_area(
            "ASR 转写文本",
            value=asr_text,
            height=80,
            disabled=True,
        )
    asr_action_columns = st.columns(2)
    with asr_action_columns[0]:
        st.button(
            "填入回答框",
            key="fill_answer_from_asr",
            on_click=_fill_answer_from_asr,
            disabled=not bool(asr_text),
            width="stretch",
        )
    with asr_action_columns[1]:
        st.button(
            "提交为当前回答",
            key="submit_answer_from_asr",
            on_click=_submit_answer_from_asr,
            disabled=not bool(asr_text),
            width="stretch",
        )

else:
    st.info("可以点击“生成评估并判断下一步”查看本轮访谈结果和后续任务。")

st.markdown("### 当前对话")
if st.session_state.dialog_turns:
    for index, turn in enumerate(st.session_state.dialog_turns, start=1):
        assistant_text = escape(str(turn.get("assistant", "")))
        user_text = escape(str(turn.get("user", "")))
        st.markdown(
            f"""
<div class="cg-chat-turn">
  <div class="cg-chat-role">{index}. 小顾（AI 访谈问题）</div>
  <div class="cg-chat-text">{assistant_text}</div>
  <div class="cg-chat-role" style="margin-top:0.75rem;">老人回答</div>
  <div class="cg-chat-text">{user_text}</div>
</div>
""",
            unsafe_allow_html=True,
        )
else:
    st.info("还没有对话。可以直接回答上方问题，或点击示例回答快速完成一轮访谈。")

with st.expander("重置当前访谈"):
    st.caption("清空后会重新从第一轮预设问题开始，不会删除已经保存到 SQLite 的历史记录。")
    st.button("清空当前访谈", on_click=_clear_dialog)

report = st.session_state.dialog_report
if report:
    metadata = report.get("metadata", {})
    result_source = metadata.get("source", "unknown")
    result_source_label = display_source(result_source)
    model_name = display_model_name(metadata.get("model", "未配置"))

    st.markdown("### 单次评估结果")
    st.write(f"结果来源：{result_source_label}")
    st.caption(
        "Qwen 文本模型表示真实文本评估；模拟结果表示演示模式或配置不完整；"
        "兜底结果表示真实调用失败后返回安全兜底。"
    )
    st.caption(f"模型：{model_name}")
    if result_source == "fallback":
        st.warning(f"已启用兜底结果：{metadata.get('reason', '模型输出不可用。')}")
    with st.expander("调试信息"):
        st.write(
            {
                "结果来源": result_source_label,
                "模型": model_name,
                "原因": metadata.get("reason", ""),
            }
        )

    flow_summary = build_assessment_flow_summary(report)
    score_cards = {card["id"]: card for card in flow_summary["score_cards"]}
    dialogue_card = score_cards["dialogue"]
    st.markdown(
        status_strip_html(
            [
                {"label": "结果来源", "value": result_source_label, "tone": "blue"},
                {"label": "模型", "value": model_name, "tone": "neutral"},
                {
                    "label": "风险等级",
                    "value": display_risk_level(report["risk_level"]),
                    "tone": "amber",
                },
                {
                    "label": "对话评估参考分",
                    "value": dialogue_card["value"],
                    "tone": "green",
                },
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown(risk_badge_html(report["risk_level"]), unsafe_allow_html=True)
    st.markdown(callout_html("结果解释", report["explanation"], tone="green"), unsafe_allow_html=True)
    st.caption(dialogue_card["explanation"])

    clock_recommendation = flow_summary["clock_recommendation"]
    next_task = flow_summary["next_task"]
    st.markdown("#### 后续任务建议")
    if clock_recommendation["recommended"]:
        st.warning(clock_recommendation["message"])
    else:
        st.success(clock_recommendation["message"])
    for reason in clock_recommendation["reasons"]:
        st.write(f"- {reason}")
    if next_task.get("step_id") == FLOW_STEP_CLOCK_TEST:
        st.info(f"小顾：{CLOCK_TRANSITION_MESSAGE}")
    else:
        st.write(f"老人端提示：{next_task['elder_message']}")
    st.write(f"工作人员操作：{next_task['primary_action_label']}")
    st.caption(flow_summary["score_policy"]["notice"])
    st.caption("这是技术原型中的本地流程规则，不是医学诊断；工作人员可根据现场情况手动补充画钟测试。")
    if st.session_state.dialog_save_status:
        if st.session_state.dialog_save_status.startswith("已保存"):
            st.success(st.session_state.dialog_save_status)
        else:
            st.warning(st.session_state.dialog_save_status)

    if next_task.get("step_id") == FLOW_STEP_CLOCK_TEST:
        st.markdown(
            callout_html(
                "下一步：继续画钟拍照",
                "这轮快捷访谈已触发画钟建议。点击后进入现场画钟、摄像头拍照、自动分析和自动保存流程。",
                tone="amber",
            ),
            unsafe_allow_html=True,
        )
        if st.button("继续画钟拍照", type="primary", width="stretch"):
            if _prepare_clock_entry_from_quick_interview(report):
                st.switch_page("pages/2_画钟测试.py")
        st.caption("继续画钟测试：保存画钟结果时会优先合并到本轮快捷访谈记录。")
    else:
        st.markdown(
            callout_html(
                "下一步：查看认知简报",
                "本次访谈可以先结束；当前结果已保存，可以查看认知简报。",
                tone="green",
            ),
            unsafe_allow_html=True,
        )
        if st.button("查看认知简报", type="primary", width="stretch"):
            st.switch_page("pages/3_认知简报.py")

    st.markdown("#### 认知域得分")
    domain_cards = []
    for domain, score in report["domain_scores"].items():
        label = DOMAIN_LABELS.get(domain, domain)
        domain_cards.append(
            metric_card_html(label, "暂无" if score is None else f"{score:.2f}", tone="blue")
        )
    st.markdown(
        '<div class="cg-metric-grid">' + "".join(domain_cards) + "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("#### 证据")
    evidence_cards = []
    for item in report["evidence"]:
        label = DOMAIN_LABELS.get(item["domain"], item["domain"])
        evidence_cards.append(evidence_card_html(label, item["text"], tone="green"))
    if evidence_cards:
        st.markdown(
            '<div class="cg-evidence-grid">' + "".join(evidence_cards) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("暂无可展示证据。")

    st.markdown("#### SQLite / memory")
    save_user_id = current_user_id
    st.write(f"保存对象：{current_display_name}")
    st.caption(f"user_id: {save_user_id}；技术原型演示档案，不包含真实隐私信息。")
    if st.button("保存当前 session"):
        _save_dialog_report(report, force=True)

    if st.session_state.dialog_save_status:
        if st.session_state.dialog_save_status.startswith("已保存"):
            st.success(st.session_state.dialog_save_status)
        else:
            st.error(st.session_state.dialog_save_status)

    st.caption(report.get("disclaimer", DISCLAIMER))
else:
    st.caption(DISCLAIMER)
