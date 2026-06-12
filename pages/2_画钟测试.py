import base64
import hashlib
import json
import struct
from html import escape
from io import BytesIO
from typing import Any, Optional

import streamlit as st
import streamlit.components.v1 as components

from core.config import load_config
from core.db import save_session
from core.mock_data import get_clock_sample_paths
from core.report import compute_clock_structure_score, format_session_time
from core.vlm_client import analyze_clock_image
from core.schemas import (
    DISCLAIMER,
    DOMAIN_LABELS,
    display_cdt_feature_value,
    display_risk_level,
    display_source,
)
from core.session_history import (
    build_clock_assessment_record,
    find_assessment_record,
    get_current_user_profile,
)
from core.staff_gate import hide_sidebar_nav, render_staff_gate
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


MIN_CLOCK_IMAGE_BYTES = 2048
MIN_CLOCK_IMAGE_SIDE = 120
MIN_CLOCK_IMAGE_PIXELS = 25000
INVALID_CLOCK_CAPTURE_MESSAGE = "没有拍清楚画钟，请把纸放稳后重新拍一次。"


def _read_image_bytes(image: Any) -> bytes:
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


def _image_hash(image_bytes: bytes) -> str:
    if not image_bytes:
        return ""
    return hashlib.sha256(image_bytes).hexdigest()


def _png_dimensions(image_bytes: bytes) -> Optional[tuple[int, int]]:
    if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n") or len(image_bytes) < 24:
        return None
    try:
        width, height = struct.unpack(">II", image_bytes[16:24])
    except struct.error:
        return None
    return int(width), int(height)


def _jpeg_dimensions(image_bytes: bytes) -> Optional[tuple[int, int]]:
    if not image_bytes.startswith(b"\xff\xd8"):
        return None
    index = 2
    size = len(image_bytes)
    while index + 9 < size:
        if image_bytes[index] != 0xFF:
            index += 1
            continue
        marker = image_bytes[index + 1]
        index += 2
        if marker in {0xD8, 0xD9}:
            continue
        if index + 2 > size:
            return None
        segment_length = int.from_bytes(image_bytes[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > size:
            return None
        if marker in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            height = int.from_bytes(image_bytes[index + 3 : index + 5], "big")
            width = int.from_bytes(image_bytes[index + 5 : index + 7], "big")
            return int(width), int(height)
        index += segment_length
    return None


def _pil_dimensions(image_bytes: bytes) -> Optional[tuple[int, int]]:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            return int(image.width), int(image.height)
    except Exception:
        return None


def _image_dimensions(image_bytes: bytes) -> Optional[tuple[int, int]]:
    return (
        _pil_dimensions(image_bytes)
        or _png_dimensions(image_bytes)
        or _jpeg_dimensions(image_bytes)
    )


def _image_luminance_stats(image_bytes: bytes) -> Optional[tuple[float, float]]:
    try:
        from PIL import Image
    except Exception:
        return None
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            grayscale = image.convert("L")
            grayscale.thumbnail((96, 96))
            pixels = list(grayscale.getdata())
    except Exception:
        return None
    if not pixels:
        return None
    mean = sum(pixels) / len(pixels)
    variance = sum((pixel - mean) ** 2 for pixel in pixels) / len(pixels)
    return mean, variance ** 0.5


def _validate_clock_capture(image_bytes: bytes) -> tuple[bool, str]:
    if not image_bytes:
        return False, INVALID_CLOCK_CAPTURE_MESSAGE
    if len(image_bytes) < MIN_CLOCK_IMAGE_BYTES:
        return False, "照片内容太少，没有拍清楚画钟，请重新拍一次。"

    dimensions = _image_dimensions(image_bytes)
    if dimensions is None:
        return False, "照片无法读取，没有拍清楚画钟，请重新拍一次。"

    width, height = dimensions
    if (
        width < MIN_CLOCK_IMAGE_SIDE
        or height < MIN_CLOCK_IMAGE_SIDE
        or width * height < MIN_CLOCK_IMAGE_PIXELS
    ):
        return False, "照片尺寸太小，没有拍清楚画钟，请靠近纸面后重新拍一次。"

    luminance_stats = _image_luminance_stats(image_bytes)
    if luminance_stats is None:
        return True, ""
    mean, contrast = luminance_stats
    if mean < 14 and contrast < 10:
        return False, "照片太暗，没有拍清楚画钟，请增加光线后重新拍一次。"
    if mean > 248 and contrast < 7:
        return False, "照片太亮，没有拍清楚画钟，请避开反光后重新拍一次。"
    if contrast < 4:
        return False, "照片几乎没有对比度，可能没有拍到纸面或画钟，请重新拍一次。"
    return True, ""


def _speak_browser_feedback(message: str) -> None:
    message_json = json.dumps(message, ensure_ascii=False)
    components.html(
        f"""
<script>
try {{
  const text = {message_json};
  if ("speechSynthesis" in window) {{
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = "zh-CN";
    utterance.rate = 0.92;
    window.speechSynthesis.speak(utterance);
  }}
}} catch (error) {{}}
</script>
""",
        height=0,
    )


def _preview_mime_type(
    image_bytes: bytes,
    filename: str = "",
    uploaded_type: str = "",
) -> str:
    if uploaded_type in {"image/png", "image/jpeg"}:
        return uploaded_type
    lower_name = filename.lower()
    if lower_name.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    return "image/png"


def _render_clock_preview(
    image_bytes: bytes,
    caption: str,
    filename: str = "",
    uploaded_type: str = "",
) -> None:
    if not image_bytes:
        return
    mime_type = _preview_mime_type(
        image_bytes,
        filename=filename,
        uploaded_type=uploaded_type,
    )
    data_url = "data:{};base64,{}".format(
        mime_type,
        base64.b64encode(image_bytes).decode("ascii"),
    )
    safe_caption = escape(str(caption or "画钟图片预览"), quote=True)
    st.markdown(
        f"""
<div class="cg-clock-sample-preview">
  <img src="{data_url}" alt="{safe_caption}" />
  <div class="cg-clock-sample-caption">{safe_caption}</div>
</div>
""",
        unsafe_allow_html=True,
    )


st.set_page_config(page_title="画钟拍照测试", layout="wide")
config = load_config()
hide_sidebar_nav()
inject_staff_theme()
clock_entry_from_elder = bool(st.session_state.get("clock_entry_from_elder"))
if not clock_entry_from_elder:
    render_staff_gate(config)

current_user = get_current_user_profile(st.session_state)
current_user_id = current_user["user_id"]
current_display_name = current_user["display_name"]

st.markdown(
    """
<style>
.cg-clock-capture-hero {
    border: 1px solid #cbdccb;
    border-radius: 20px;
    background: linear-gradient(180deg, #fffdf8 0%, #edf5ef 100%);
    padding: clamp(1rem, 2.2vw, 1.45rem);
    margin: 0 0 0.9rem;
    box-shadow: 0 16px 38px rgba(31, 36, 33, 0.07);
}
.cg-clock-capture-kicker {
    color: var(--cg-terracotta);
    font-size: 0.82rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.35rem;
}
.cg-clock-capture-title {
    font-family: var(--cg-serif);
    color: var(--cg-green-dark);
    font-size: clamp(2rem, 4.2vw, 3.5rem);
    line-height: 1.12;
    font-weight: 700;
}
.cg-clock-capture-copy {
    color: var(--cg-navy-muted);
    font-size: clamp(1.05rem, 1.9vw, 1.25rem);
    line-height: 1.55;
    margin-top: 0.55rem;
}
.st-key-clock_camera_capture div[data-testid="stCameraInput"] {
    margin-top: 0.6rem;
}
.st-key-clock_camera_capture div[data-testid="stCameraInput"] button {
    min-height: 5.35rem;
    font-size: clamp(1.38rem, 2.7vw, 1.85rem);
}
.st-key-clock_camera_capture div[data-testid="stCameraInput"] [data-testid="stCameraInputButton"]::after {
    content: "拍照";
    font-size: clamp(1.38rem, 2.7vw, 1.85rem);
}
.st-key-clock_view_brief_button div.stButton > button {
    min-height: 4.7rem;
    border-radius: 18px;
    font-size: clamp(1.25rem, 2.4vw, 1.75rem);
    font-weight: 800;
}
.cg-clock-sample-preview {
    display: grid;
    justify-items: center;
    gap: 0.55rem;
    width: min(100%, 920px);
    max-width: 920px;
    border: 1px solid var(--cg-border);
    border-radius: 16px;
    background: rgba(255, 253, 248, 0.82);
    padding: clamp(0.7rem, 1.6vw, 1rem);
    margin: 0.75rem auto 0.35rem;
    box-shadow: 0 10px 24px rgba(31, 36, 33, 0.055);
}
.cg-clock-sample-preview img {
    display: block;
    width: 100%;
    max-height: min(62vh, 620px);
    height: auto;
    object-fit: contain;
    border-radius: 12px;
    border: 1px solid #e5e0d5;
    background: #fffdf8;
}
.cg-clock-sample-caption {
    color: var(--cg-muted);
    font-size: 0.92rem;
    line-height: 1.5;
    text-align: center;
}
@media (max-width: 760px) {
    .cg-clock-sample-preview img {
        width: 100%;
        max-height: 54vh;
    }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="cg-clock-capture-hero">
  <div class="cg-clock-capture-kicker">Clock Drawing Test</div>
  <div class="cg-clock-capture-title">画一个 11 点 10 分的钟</div>
  <div class="cg-clock-capture-copy">请把纸面放进摄像头范围，画好后点击拍照。拍完会自动分析并保存到{current_display_name}的本次综合评估。</div>
</div>
""",
    unsafe_allow_html=True,
)
vlm_config_complete = all(
    [
        config.vlm_base_url.strip(),
        config.vlm_api_key.strip(),
        config.vlm_model.strip(),
    ]
)

clock_sample_options = {
    "normal": "正常表现示例",
    "spatial_shift": "轻度下降示例",
    "wrong_hands": "明显异常示例",
}
clock_sample_paths = get_clock_sample_paths()

clock_state_defaults = {
    "clock_sample_path": None,
    "clock_report": None,
    "clock_save_status": "",
    "clock_capture_mode": "camera",
    "clock_analysis_in_progress": False,
    "clock_pending_analysis": False,
    "clock_pending_analysis_source": "",
    "clock_pending_analysis_bytes": b"",
    "clock_pending_analysis_filename": "",
    "clock_last_analyzed_hash": "",
    "clock_auto_saved_hash": "",
    "clock_auto_save_attempted_hash": "",
    "clock_last_invalid_hash": "",
    "clock_current_input_hash": "",
    "current_assessment_id": None,
}
for state_key, default_value in clock_state_defaults.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default_value

if st.session_state.get("current_assessment_user_id") not in (None, current_user_id):
    st.session_state.current_assessment_id = None
    st.session_state.clock_auto_saved_hash = ""
    st.session_state.clock_auto_save_attempted_hash = ""
st.session_state.current_assessment_user_id = current_user_id

def _save_clock_report(
    report: dict,
    target_time: str,
    image_hash: str = "",
) -> bool:
    try:
        existing_record = find_assessment_record(
            st.session_state.get("current_assessment_id"),
            user_id=current_user_id,
        )
        record = build_clock_assessment_record(
            report,
            user_id=current_user_id,
            assessment_id=st.session_state.get("current_assessment_id"),
            existing_record=existing_record,
            target_time=target_time,
        )
        save_session(record)
        st.session_state.current_assessment_id = record["assessment_id"]
        st.session_state.current_assessment_user_id = current_user_id
        if image_hash:
            st.session_state.clock_auto_saved_hash = image_hash
        st.session_state.clock_save_status = (
            f"画钟结果已保存到{current_display_name}的本次综合评估，保存时间："
            f"{format_session_time(record['created_at'])}"
        )
        return True
    except Exception as error:
        st.session_state.clock_save_status = (
            f"保存失败：{type(error).__name__}: {str(error)[:160]}"
        )
        return False


def _auto_save_clock_report(report: dict, target_time: str, image_hash: str) -> None:
    if not image_hash:
        return
    if st.session_state.clock_auto_saved_hash == image_hash:
        return
    if st.session_state.clock_auto_save_attempted_hash == image_hash:
        return
    st.session_state.clock_auto_save_attempted_hash = image_hash
    _save_clock_report(report, target_time, image_hash=image_hash)


def _analyze_clock_bytes(image_bytes: bytes, filename: str, target_time: str) -> Optional[dict]:
    try:
        return analyze_clock_image(
            image_bytes,
            filename=filename,
            config=config,
            target_time=target_time,
        )
    except Exception as error:
        st.session_state.clock_save_status = (
            f"分析失败：{type(error).__name__}: {str(error)[:160]}"
        )
        return None


def _queue_clock_analysis(
    source: str,
    image_bytes: bytes = b"",
    filename: str = "",
) -> None:
    st.session_state.clock_pending_analysis = True
    st.session_state.clock_analysis_in_progress = True
    st.session_state.clock_pending_analysis_source = source
    st.session_state.clock_pending_analysis_bytes = image_bytes
    st.session_state.clock_pending_analysis_filename = filename


def _clear_clock_analysis_queue() -> None:
    st.session_state.clock_analysis_in_progress = False
    st.session_state.clock_pending_analysis = False
    st.session_state.clock_pending_analysis_source = ""
    st.session_state.clock_pending_analysis_bytes = b""
    st.session_state.clock_pending_analysis_filename = ""


active_target = "11:10"
st.session_state.clock_target_time = active_target
analysis_busy = bool(st.session_state.get("clock_analysis_in_progress"))
assessment_id = st.session_state.get("current_assessment_id")
flow_status = (
    "访谈后画钟，保存后合并本轮综合评估"
    if clock_entry_from_elder and assessment_id
    else "工作人员补充画钟，保存到当前长者档案"
)
st.markdown(
    status_strip_html(
        [
            {"label": "当前长者", "value": current_display_name, "tone": "green"},
            {"label": "目标时间", "value": active_target, "tone": "blue"},
            {"label": "当前流程", "value": flow_status, "tone": "amber"},
        ]
    ),
    unsafe_allow_html=True,
)

st.markdown(
    callout_html(
        "现场拍照",
        "纸面放平、钟面完整入镜后，点击下方拍照按钮。目标时间固定为 11:10。",
        tone="green",
    ),
    unsafe_allow_html=True,
)
camera_file = st.camera_input(
    "拍照画钟",
    key="clock_camera_capture",
    help="摄像头不可用时，可展开下方备选方式上传图片或加载示例画钟。",
    disabled=analysis_busy,
)

if analysis_busy:
    st.info("正在分析画钟图片，请稍等，暂时不用重复拍照或点击按钮。")
elif camera_file is None:
    st.info("等待摄像头拍照。没有摄像头或浏览器未授权时，请使用下方备选方式。")
else:
    camera_bytes = _read_image_bytes(camera_file)
    camera_hash = _image_hash(camera_bytes)
    st.session_state.clock_current_input_hash = camera_hash
    is_valid_image, validation_message = _validate_clock_capture(camera_bytes)
    st.image(camera_file, caption="本次摄像头拍照预览", width="stretch")

    if not is_valid_image:
        st.session_state.clock_report = None
        st.session_state.clock_save_status = ""
        st.markdown(
            callout_html(
                "没有拍清楚画钟",
                validation_message or INVALID_CLOCK_CAPTURE_MESSAGE,
                tone="terracotta",
            ),
            unsafe_allow_html=True,
        )
        st.warning("没有拍清楚画钟，请把纸放稳后重新拍一次。")
        if camera_hash and st.session_state.clock_last_invalid_hash != camera_hash:
            st.session_state.clock_last_invalid_hash = camera_hash
            _speak_browser_feedback("没有拍清楚，请重新拍一次。")
    elif (
        camera_hash != st.session_state.clock_last_analyzed_hash
        or not isinstance(st.session_state.clock_report, dict)
    ):
        with st.spinner("照片已收到，正在自动分析画钟，请稍等……"):
            report_result = _analyze_clock_bytes(
                camera_bytes,
                filename="camera-clock.jpg",
                target_time=active_target,
            )
        if isinstance(report_result, dict):
            st.session_state.clock_report = report_result
            st.session_state.clock_last_analyzed_hash = camera_hash
            st.session_state.clock_save_status = ""
            st.session_state.clock_sample_path = None
            st.session_state.clock_capture_mode = "camera"
            _auto_save_clock_report(report_result, active_target, camera_hash)
    else:
        st.success("这张照片已经完成自动分析。")
        if isinstance(st.session_state.clock_report, dict):
            _auto_save_clock_report(
                st.session_state.clock_report,
                active_target,
                camera_hash,
            )

with st.expander("备选方式：上传图片或加载示例画钟", expanded=analysis_busy):
    st.caption(
        "上传图片和示例画钟仅作为工作人员兜底：用于演示、摄像头不可用或开发测试。"
    )
    uploaded_file = st.file_uploader(
        "上传画钟图片",
        type=["png", "jpg", "jpeg"],
        help="当前阶段仅在页面中预览图片，不保存上传文件。",
        disabled=analysis_busy,
    )
    selected_sample = st.selectbox(
        "选择示例画钟",
        options=list(clock_sample_options.keys()),
        format_func=lambda key: clock_sample_options[key],
        disabled=analysis_busy,
    )
    if st.button("加载示例画钟", width="stretch", disabled=analysis_busy):
        st.session_state.clock_sample_path = str(clock_sample_paths[selected_sample])
        st.session_state.clock_report = None
        st.session_state.clock_save_status = ""
        st.session_state.clock_current_input_hash = ""

    if uploaded_file is not None:
        st.session_state.clock_sample_path = None
        uploaded_preview_bytes = _read_image_bytes(uploaded_file)
        _render_clock_preview(
            uploaded_preview_bytes,
            "已上传图片预览",
            filename=getattr(uploaded_file, "name", ""),
            uploaded_type=getattr(uploaded_file, "type", ""),
        )
    elif st.session_state.clock_sample_path:
        sample_path = st.session_state.clock_sample_path
        with open(sample_path, "rb") as file:
            sample_preview_bytes = file.read()
        _render_clock_preview(
            sample_preview_bytes,
            "示例画钟预览",
            filename=sample_path,
        )
    else:
        st.markdown(
            callout_html(
                "等待备选图片",
                "可以上传 png/jpg 图片，也可以先加载演示示例。未选择图片时仍可触发 mock/fallback 分析。",
                tone="blue",
            ),
            unsafe_allow_html=True,
        )

    analysis_feedback_slot = st.empty()
    if analysis_busy:
        analysis_feedback_slot.markdown(
            callout_html(
                "正在分析",
                "画钟图片正在分析，请稍等，不需要重复点击；完成后会自动显示结果。",
                tone="amber",
            ),
            unsafe_allow_html=True,
        )

    analyze_clicked = st.button(
        "正在分析画钟图片..." if analysis_busy else "分析画钟图片",
        type="primary",
        width="stretch",
        disabled=analysis_busy,
    )
    if analyze_clicked:
        if uploaded_file is not None:
            pending_bytes = _read_image_bytes(uploaded_file)
            _queue_clock_analysis(
                "upload",
                image_bytes=pending_bytes,
                filename=getattr(uploaded_file, "name", "uploaded-clock.png"),
            )
        elif st.session_state.clock_sample_path:
            sample_path = st.session_state.clock_sample_path
            with open(sample_path, "rb") as file:
                pending_bytes = file.read()
            _queue_clock_analysis(
                "sample",
                image_bytes=pending_bytes,
                filename=sample_path.split("\\")[-1].split("/")[-1],
            )
        else:
            _queue_clock_analysis("demo", filename="demo-clock.png")
        st.rerun()

with st.expander("工作人员配置状态", expanded=False):
    status_columns = st.columns(4)
    status_columns[0].metric("DEMO_MODE", str(config.demo_mode).lower())
    status_columns[1].metric("VLM_MODEL", display_model_name(config.vlm_model))
    status_columns[2].metric("VLM 配置", "完整" if vlm_config_complete else "不完整")
    status_columns[3].metric(
        "API Key",
        "已配置" if config.vlm_api_key.strip() else "未配置",
    )

if st.session_state.get("clock_pending_analysis"):
    pending_source = st.session_state.get("clock_pending_analysis_source", "")
    pending_bytes = st.session_state.get("clock_pending_analysis_bytes", b"")
    pending_filename = st.session_state.get("clock_pending_analysis_filename", "")
    if "analysis_feedback_slot" in locals():
        analysis_feedback_slot.markdown(
            callout_html(
                "正在分析",
                "画钟图片正在分析，请稍等，不需要重复点击；完成后会自动显示结果。",
                tone="amber",
            ),
            unsafe_allow_html=True,
        )
    try:
        with st.spinner("正在分析画钟图片，请稍等……"):
            if pending_source in {"upload", "sample"}:
                report_result = _analyze_clock_bytes(
                    pending_bytes,
                    filename=pending_filename or "clock.png",
                    target_time=active_target,
                )
                st.session_state.clock_report = report_result
                st.session_state.clock_current_input_hash = _image_hash(pending_bytes)
                if report_result:
                    st.session_state.clock_save_status = ""
            else:
                try:
                    st.session_state.clock_report = analyze_clock_image(
                        filename=pending_filename or "demo-clock.png",
                        config=config,
                        target_time=active_target,
                    )
                except Exception as error:
                    st.session_state.clock_report = None
                    st.session_state.clock_save_status = (
                        f"分析失败：{type(error).__name__}: {str(error)[:160]}"
                    )
                st.session_state.clock_current_input_hash = ""
                if st.session_state.clock_report:
                    st.session_state.clock_save_status = ""
    finally:
        _clear_clock_analysis_queue()
    st.rerun()

report = st.session_state.clock_report

if report:
    metadata = report.get("metadata", {})
    source = metadata.get("source", "unknown")
    source_label = display_source(source)
    model = display_model_name(metadata.get("model", "未配置"))
    findings = report.get("clock_findings", {})
    cdt_features = report.get("cdt_features", {})
    clock_score = compute_clock_structure_score(report)

    st.markdown("### 画钟分析结果")
    st.markdown(
        section_header_html(
            "分析结果",
            eyebrow="Structured CDT Output",
            body="结果保留技术原型风险提示口径，不构成医学诊断；异常时会安全回退到兜底结果。",
        ),
        unsafe_allow_html=True,
    )
    st.markdown(
        status_strip_html(
            [
                {"label": "结果来源", "value": source_label, "tone": "blue"},
                {"label": "模型", "value": model, "tone": "neutral"},
                {
                    "label": "风险等级",
                    "value": display_risk_level(report["risk_level"]),
                    "tone": "amber",
                },
                {
                    "label": "画钟结构分",
                    "value": (
                        "暂无"
                        if clock_score["score"] is None
                        else f"{clock_score['score']} / 10"
                    ),
                    "tone": "green",
                },
            ]
        ),
        unsafe_allow_html=True,
    )
    st.markdown(risk_badge_html(report["risk_level"]), unsafe_allow_html=True)
    st.caption(
        "Qwen-VL 视觉模型表示真实画钟分析；模拟结果表示演示模式或配置不完整；"
        "兜底结果表示真实调用失败后返回安全兜底。"
    )
    if source == "fallback":
        st.warning(f"已启用兜底结果：{metadata.get('reason', '模型输出不可用。')}")

    st.markdown(callout_html("结果解释", report.get("explanation", ""), tone="green"), unsafe_allow_html=True)
    if clock_score["score"] is not None:
        st.caption(clock_score["explanation"])

    st.markdown("#### 认知域得分")
    score_cards = []
    for domain, score in report.get("domain_scores", {}).items():
        label = DOMAIN_LABELS.get(domain, domain)
        score_cards.append(
            metric_card_html(label, "暂无" if score is None else f"{score:.2f}", tone="blue")
        )
    if score_cards:
        st.markdown(
            '<div class="cg-metric-grid">' + "".join(score_cards) + "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("#### 画钟观察")
    st.markdown(
        '<div class="cg-evidence-grid">'
        + evidence_card_html("数字布局", findings.get("number_placement", "暂无"), tone="blue")
        + evidence_card_html("指针准确性", findings.get("hand_accuracy", "暂无"), tone="terracotta")
        + "</div>",
        unsafe_allow_html=True,
    )

    if isinstance(cdt_features, dict) and cdt_features:
        st.markdown("#### CDT 特征")
        cdt_chips = [
            chip_html(
                "数字完整",
                display_cdt_feature_value(
                    "numbers_complete",
                    cdt_features.get("numbers_complete"),
                ),
                "green",
            ),
            chip_html(
                "数字顺序",
                display_cdt_feature_value(
                    "number_order_correct",
                    cdt_features.get("number_order_correct"),
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
                "数字分布",
                display_cdt_feature_value(
                    "number_distribution",
                    cdt_features.get("number_distribution"),
                ),
                "blue",
            ),
            chip_html(
                "指针存在",
                display_cdt_feature_value("hands_present", cdt_features.get("hands_present")),
                "green",
            ),
            chip_html(
                "目标时间",
                display_cdt_feature_value(
                    "target_time_match",
                    cdt_features.get("target_time_match"),
                ),
                "terracotta",
            ),
            chip_html(
                "中心锚点",
                display_cdt_feature_value(
                    "center_anchor_clear",
                    cdt_features.get("center_anchor_clear"),
                ),
                "neutral",
            ),
        ]
        st.markdown("".join(cdt_chips), unsafe_allow_html=True)

    st.markdown("#### 证据")
    evidence_cards = []
    for item in report.get("evidence", []):
        if isinstance(item, dict):
            label = DOMAIN_LABELS.get(item.get("domain"), item.get("domain", "证据"))
            evidence_cards.append(evidence_card_html(label, item.get("text", ""), tone="green"))
        else:
            evidence_cards.append(evidence_card_html("证据", item, tone="green"))
    if evidence_cards:
        st.markdown(
            '<div class="cg-evidence-grid">' + "".join(evidence_cards) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.caption("暂无可展示证据。")

    with st.expander("调试信息"):
        st.write(
            {
                "结果来源": source_label,
                "模型": model,
                "原因": metadata.get("reason", ""),
                "校验错误": metadata.get("validation_errors", []),
                "已校准": report.get("calibrated", False),
                "校准说明": report.get("calibration_notes", []),
            }
        )

    st.markdown("#### 保存与下一步")
    st.write(f"保存对象：{current_display_name}")
    st.caption(f"user_id: {current_user_id}；技术原型演示档案，不包含真实隐私信息。")

    with st.expander("手动保存兜底", expanded=False):
        st.caption("摄像头拍照会自动保存；如果自动保存失败，工作人员可在这里手动重试。")
        if st.button("保存画钟结果"):
            _save_clock_report(
                report,
                active_target,
                image_hash=st.session_state.get("clock_current_input_hash", ""),
            )

    if st.session_state.clock_save_status:
        save_succeeded = st.session_state.clock_save_status.startswith(
            ("画钟结果已保存", "已保存")
        )
        if save_succeeded:
            st.success(st.session_state.clock_save_status)
            st.markdown(
                callout_html(
                    "下一步：查看认知简报",
                    "画钟已记录，可以查看认知简报。",
                    tone="green",
                ),
                unsafe_allow_html=True,
            )
            if st.button(
                "查看认知简报",
                type="primary",
                width="stretch",
                key="clock_view_brief_button",
            ):
                st.session_state.clock_entry_from_elder = False
                st.switch_page("pages/3_认知简报.py")
        else:
            st.error(st.session_state.clock_save_status)
            st.warning("自动保存未完成，工作人员可以使用上方手动保存兜底。")

    st.caption(report.get("disclaimer", DISCLAIMER))
else:
    st.caption(DISCLAIMER)
