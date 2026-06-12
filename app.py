import html

import streamlit as st

from core.config import build_runtime_status, load_config
from core.db import authenticate_user, list_demo_users
from core.report import format_session_time, infer_session_test_type
from core.schemas import DISCLAIMER, display_risk_level
from core.session_history import (
    clear_current_user_profile,
    get_current_user_profile,
    load_current_user_sessions,
    store_current_user_profile,
)
from core.staff_gate import hide_sidebar_nav, is_staff_unlocked, verify_staff_password
from core.ui import (
    callout_html,
    inject_elder_theme,
    logo_mark_svg,
    section_header_html,
    status_strip_html,
)


st.set_page_config(
    page_title="CogniGuard",
    layout="wide",
    initial_sidebar_state="collapsed",
)
hide_sidebar_nav()
inject_elder_theme()

logo_svg = logo_mark_svg()


def _reset_assessment_context() -> None:
    for key in (
        "current_assessment_id",
        "current_assessment_user_id",
        "elder_current_assessment_id",
    ):
        st.session_state[key] = None


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - compatibility for older Streamlit.
        st.experimental_rerun()


def _recent_status_text(profile: dict) -> str:
    current_data = load_current_user_sessions(limit=1, user_profile=profile)
    sessions = current_data["sessions"]
    if not sessions:
        return "暂无历史记录"

    latest = sessions[-1]
    test_type = infer_session_test_type(latest)
    created_at = format_session_time(latest.get("created_at"))
    risk_level = display_risk_level(str(latest.get("risk_level", "unknown")))
    return f"{test_type} · {risk_level} · {created_at}"


def _display_model_name(model_name: str) -> str:
    model = str(model_name or "").strip()
    if not model or model == "未配置":
        return model or "未配置"
    return model.upper()


current_user = get_current_user_profile(st.session_state)
safe_display_name = html.escape(str(current_user["display_name"]))

st.markdown(
    f"""
<div class="cg-home-hero">
  <div class="cg-home-brand-row">
    <div class="cg-home-logo-mark" aria-label="CogniGuard logo">
      {logo_svg}
    </div>
    <div>
      <div class="cg-home-kicker">CogniGuard</div>
      <div class="cg-home-brand-sub">长者友好访谈入口 · 温柔的认知健康助手</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([1.15, 0.85], gap="large", vertical_alignment="center")
with hero_left:
    st.markdown(
        """
<div class="cg-home-headline">
  <h1 class="cg-home-title">您好，我是小顾。</h1>
  <div class="cg-home-subtitle">我们像聊天一样，慢慢完成今天的认知健康访谈。</div>
  <div class="cg-home-copy">
    我会慢慢问几个日常问题，您想到什么就慢慢说。
    页面会一步一步提示当前进度，后续需要做什么也会清楚显示。
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="cg-home-cta-hint">语音访谈，由小顾一步步引导</div>', unsafe_allow_html=True)
    if st.button(
        "点这里开始和小顾聊天",
        type="primary",
        width="stretch",
        key="home_start_chat",
    ):
        st.session_state.elder_autostart_requested = True
        st.switch_page("pages/4_长者简易版.py")
with hero_right:
    st.markdown(
        f"""
<div class="cg-home-visit-panel">
  <div class="cg-home-panel-kicker">今日访谈状态</div>
  <div class="cg-home-visit-name">陪 {safe_display_name} 聊一会儿</div>
  <span class="cg-status-pill cg-pill-green">准备就绪</span>
  <div class="cg-home-visit-row"><span>方式</span><strong>听问题，说回答</strong></div>
  <div class="cg-home-visit-row"><span>节奏</span><strong>慢慢来，不考试</strong></div>
  <div class="cg-home-visit-row"><span>档案</span><strong>{safe_display_name}</strong></div>
  <div class="cg-home-privacy">只保存转写后的文字，不保存原始录音。</div>
</div>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
<div class="cg-home-how">
  <div class="cg-section-eyebrow">怎么进行</div>
  <div class="cg-home-steps">
    <div class="cg-home-step">
      <div class="cg-home-step-number">01</div>
      <div class="cg-home-step-title">听小顾提问</div>
      <div class="cg-home-step-copy">问题会用大字显示，也可以播放语音。</div>
    </div>
    <div class="cg-home-step">
      <div class="cg-home-step-number">02</div>
      <div class="cg-home-step-title">像聊天一样回答</div>
      <div class="cg-home-step-copy">说错、停顿都没关系，小顾会慢慢等。</div>
    </div>
    <div class="cg-home-step">
      <div class="cg-home-step-number">03</div>
      <div class="cg-home-step-title">查看后续提示</div>
      <div class="cg-home-step-copy">需要画钟时，小顾会先说明原因，再进入拍照或简报。</div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

config = load_config()
runtime_status = build_runtime_status(config)

st.markdown('<div class="cg-home-staff"></div>', unsafe_allow_html=True)
with st.expander("工作人员设置与辅助功能（展开）", expanded=False):
    st.markdown("### 后台安全系统")
    if not is_staff_unlocked(st.session_state):
        st.warning("后台已加锁。老人端默认开放，管理员入口需要密码解锁。")
        st.caption(f"管理员临时访问口令：{config.staff_password}")
        with st.form("home_staff_gate_form"):
            staff_password = st.text_input("管理员密码", type="password")
            staff_submitted = st.form_submit_button("解锁后台", type="primary")
        if staff_submitted:
            if verify_staff_password(staff_password, config):
                st.session_state.staff_unlocked = True
                st.session_state.home_staff_gate_status = "管理员已解锁，可以进入工作人员设置。"
                _rerun()
            else:
                st.session_state.home_staff_gate_status = "管理员密码不正确，后台仍保持加锁。"
        if st.session_state.get("home_staff_gate_status"):
            status_text = str(st.session_state.home_staff_gate_status)
            if status_text.startswith("管理员密码"):
                st.error(status_text)
            else:
                st.success(status_text)
        st.caption("后台安全系统已启用；未解锁时只显示管理员密码输入。")
    else:
        st.success("后台安全系统已启用，管理员已解锁。")
        auth_label = "账号已登录" if current_user["is_authenticated"] else "未登录，使用默认长者档案"
        recent_status = _recent_status_text(current_user)
        safe_auth_label = html.escape(auth_label)
        safe_recent_status = html.escape(recent_status)
        st.markdown(
            section_header_html(
                "工作人员后台",
                eyebrow="Operator Console",
                body="后台入口用于演示、工作人员复核和辅助测试；老人端入口仍保持默认开放。",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            status_strip_html(
                [
                    {"label": "当前长者", "value": current_user["display_name"], "tone": "green"},
                    {"label": "登录状态", "value": auth_label, "tone": "blue"},
                    {"label": "最近评估", "value": recent_status, "tone": "amber"},
                ]
            ),
            unsafe_allow_html=True,
        )

        st.markdown("### 当前长者档案")
        st.markdown(
            f"""
<div class="profile-panel">
  <div class="profile-title">当前长者档案：{safe_display_name}</div>
  <div class="profile-line">{safe_auth_label}</div>
  <div class="profile-line">最近一次评估：{safe_recent_status}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown("### 登录账号")
        demo_users = list_demo_users()
        demo_hint = "；".join(
            f"{user['username']} / 123456 / {user['display_name']}"
            for user in demo_users
        )
        st.caption(f"可用登录账号：{demo_hint}")
        st.caption("新增账号由系统管理员维护，本页面不开放注册。")
        if current_user["is_authenticated"]:
            if st.button("退出当前账号", width="stretch"):
                clear_current_user_profile(st.session_state)
                _reset_assessment_context()
                st.session_state.demo_login_status = "已回到默认长者档案：张奶奶。"
                st.rerun()
        else:
            with st.form("demo_login_form"):
                username = st.text_input("账号", value="zhang")
                password = st.text_input("密码", type="password")
                submitted = st.form_submit_button("登录账号")
            if submitted:
                user = authenticate_user(username, password)
                if user is None:
                    st.session_state.demo_login_status = "账号或密码不正确，请使用已配置的登录账号。"
                else:
                    previous_user_id = current_user["user_id"]
                    profile = store_current_user_profile(st.session_state, user)
                    if previous_user_id != profile["user_id"]:
                        _reset_assessment_context()
                    st.session_state.demo_login_status = (
                        f"已登录账号：{profile['display_name']}。"
                    )
                    st.rerun()
        if st.session_state.get("demo_login_status"):
            status_text = str(st.session_state.demo_login_status)
            if status_text.startswith("账号或密码"):
                st.warning(status_text)
            else:
                st.success(status_text)

        st.markdown("### 工作人员辅助功能页")
        st.markdown(
            callout_html(
                "辅助功能说明",
                (
                    "解锁后，工作人员可以查看认知简报、补充画钟测试、进入演示或使用快捷访谈评估。"
                    "隐藏侧边栏导航后，管理员入口集中在这里。"
                ),
                tone="blue",
            ),
            unsafe_allow_html=True,
        )

        st.markdown("#### 认知简报")
        if st.button(
            "查看认知简报",
            type="primary",
            key="home_brief_entry",
            width="stretch",
        ):
            st.switch_page("pages/3_认知简报.py")
        st.caption("查看最近报告、趋势图、家属端提醒和非诊断说明。")

        st.markdown("#### 其他管理员工具")
        staff_left, staff_middle, staff_right = st.columns(3)
        with staff_left:
            if st.button(
                "继续画钟拍照",
                key="home_clock_entry",
                width="stretch",
            ):
                st.switch_page("pages/2_画钟测试.py")
            st.caption("工作人员辅助拍照或加载示例画钟，补充本轮观察材料。")
        with staff_middle:
            if st.button(
                "快捷访谈评估",
                key="home_interview_test_entry",
                width="stretch",
            ):
                st.switch_page("pages/1_对话评估.py")
            st.caption("工作人员快速跑通访谈；触发画钟时先呈现小顾说明，再进入拍照/简报链路。")
        with staff_right:
            if st.button(
                "演示模式",
                key="home_classroom_entry",
                width="stretch",
            ):
                st.switch_page("pages/5_演示模式.py")
            st.caption("使用模拟数据快速展示正常、轻度下降和明显异常三类效果。")

        st.write("- 快捷访谈评估：预选/文字/语音回答，快速复刻老人端访谈到画钟/简报链路。")
        st.write("- 画钟测试：上传或加载示例画钟，调用 Qwen-VL 或 fallback 分析结构化结果。")
        st.write("- 认知简报：读取 SQLite 或 fixture，展示最近报告、趋势和家属提醒。")
        st.write("- 和小顾聊天：老人端正式语音访谈流程，由首页大按钮进入。")
        st.write("- 演示：选择认知水平，一键生成模拟流程、语音和评估报告。")

        st.markdown(
            section_header_html(
                "运行状态",
                eyebrow="Runtime Status",
                body="系统支持真实 Qwen / Qwen-VL / ASR / TTS 配置；配置缺失或 DEMO_MODE=true 时会安全回退到 mock/fallback。",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            status_strip_html(
                [
                    {
                        "label": "当前运行模式",
                        "value": runtime_status["运行模式"],
                        "tone": "green",
                    },
                    {
                        "label": "LLM 模型",
                        "value": _display_model_name(runtime_status["LLM 模型"]),
                        "tone": "blue",
                    },
                    {
                        "label": "VLM 模型",
                        "value": _display_model_name(runtime_status["VLM 模型"]),
                        "tone": "amber",
                    },
                ]
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            callout_html(
                "运行说明",
                "真实 API 可用于演示；任何模型配置缺失或调用失败时，页面都会保留 mock/fallback 兜底，不影响主流程演示。",
                tone="green",
            ),
            unsafe_allow_html=True,
        )

        st.markdown(
            section_header_html(
                "安全边界",
                eyebrow="Safety Boundary",
                body="技术原型只展示认知健康风险提示，不采集真实医疗数据，也不显示本地密钥。",
            ),
            unsafe_allow_html=True,
        )
        st.markdown(
            status_strip_html(
                [
                    {"label": "医疗数据", "value": "不采集真实数据", "tone": "green"},
                    {"label": "原始音频", "value": "不保存原始用户音频", "tone": "blue"},
                    {"label": "输出性质", "value": "非医学诊断", "tone": "amber"},
                    {
                        "label": "API Key",
                        "value": "API Key 只保存在本地 .env 或服务器 .env，不显示在页面中。",
                        "tone": "terracotta",
                    },
                ]
            ),
            unsafe_allow_html=True,
        )

st.caption(DISCLAIMER)
