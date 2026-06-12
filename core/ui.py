"""CogniGuard 共享前端主题层。

视觉风格：暖调编辑感（Warm Editorial）。
纸白底、近黑墨字、哑光鼠尾草绿主色、陶土橙点睛；衬线中文标题 + 无衬线正文。
靠留白与发丝分隔线分组，几乎不用投影；字重收敛到三档（400/500-600/700）。

页面只负责状态和流程编排，颜色、基础控件和可复用展示片段集中在这里。
"""

from __future__ import annotations

import base64
from functools import lru_cache
from html import escape
from pathlib import Path
from typing import Any, Iterable


# Core color tokens — "Warm Editorial" palette.
# 克制：纸白底 + 近黑墨字 + 哑光鼠尾草绿主色 + 陶土橙点睛。
# 蓝色 accent 已并入主色，避免四色彩虹。
BRAND_GREEN = "#5C7A6B"
BRAND_GREEN_DARK = "#4A6557"
BRAND_GREEN_SOFT = "#E9EFEA"
TITLE_NAVY = "#1F2421"        # 近黑墨（变量名保留以兼容引用）
NAVY_MUTED = "#3A413B"
PAGE_BG = "#F6F3EC"           # 纸白
PANEL_BG = "#FBF8F1"          # 微暖面板
CARD_BG = "#FFFFFF"
ACCENT_TERRACOTTA = "#B5654A"
ACCENT_TERRACOTTA_SOFT = "#F3E4DC"
ACCENT_AMBER = "#A9762B"
ACCENT_AMBER_SOFT = "#F6ECD8"
ACCENT_BLUE = "#5C7A6B"       # 蓝并入主色，去彩虹
ACCENT_BLUE_SOFT = "#E9EFEA"
TEXT_MUTED = "#5A615C"
TEXT_SOFT = "#7A817B"
BORDER_SOFT = "#E5E0D5"       # 统一发丝分隔线
BORDER_COOL = "#E5E0D5"
SUCCESS_SOFT = "#E6EFE6"
DANGER_SOFT = "#F4E2DD"
UNKNOWN_SOFT = "#ECEAE2"


def _html(value: Any) -> str:
    return escape(str(value), quote=True)


def display_model_name(model_name: Any) -> str:
    model = str(model_name or "").strip()
    if not model or model == "未配置":
        return "未配置"
    return model.upper()


@lru_cache(maxsize=1)
def logo_mark_svg() -> str:
    """Return the inline logo mark used by the homepage hero."""
    logo_path = (
        Path(__file__).resolve().parent.parent
        / "assets"
        / "branding"
        / "cogniguard-logo-05-transparent-ui.png"
    )
    if logo_path.exists():
        encoded = base64.b64encode(logo_path.read_bytes()).decode("ascii")
        return (
            '<img src="data:image/png;base64,'
            f'{encoded}" alt="" loading="eager" />'
        )

    return """
<svg viewBox="0 0 512 512" fill="none" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
  <path d="M256 60L392 111V222C392 310 337 390 256 427C175 390 120 310 120 222V111L256 60Z" fill="#5C7A6B"/>
  <path d="M256 91L363 131V221C363 292 320 359 256 390C192 359 149 292 149 221V131L256 91Z" fill="#FBF8F1"/>
  <path d="M182 181C182 152.833 204.833 130 233 130H283C317.242 130 345 157.758 345 192C345 226.242 317.242 254 283 254H246L206 289V254H233C204.833 254 182 231.167 182 203V181Z" fill="#E9EFEA"/>
  <circle cx="256" cy="194" r="57" fill="#FBF8F1" stroke="#3A413B" stroke-width="18"/>
  <path d="M256 157V196L288 215" stroke="#3A413B" stroke-width="20" stroke-linecap="round" stroke-linejoin="round"/>
  <circle cx="256" cy="194" r="10" fill="#B5654A"/>
  <circle cx="211" cy="318" r="16" fill="#8AA591"/>
  <circle cx="256" cy="335" r="18" fill="#3A413B"/>
  <circle cx="303" cy="318" r="16" fill="#8AA591"/>
  <path d="M225 322L241 329M272 329L289 322" stroke="#3A413B" stroke-width="12" stroke-linecap="round"/>
  <path d="M198 352C225 380 287 380 314 352" stroke="#B5654A" stroke-width="16" stroke-linecap="round"/>
</svg>
""".strip()


def risk_badge_html(risk_level: str, label: str | None = None) -> str:
    risk = str(risk_level or "unknown").strip().lower()
    labels = {
        "low": "低风险",
        "medium": "中等风险",
        "high": "高风险",
        "unknown": "无法评估",
    }
    return (
        f'<span class="cg-risk-badge cg-risk-{_html(risk if risk in labels else "unknown")}">'
        f"{_html(label or labels.get(risk, labels['unknown']))}</span>"
    )


def chip_html(label: str, value: Any = "", tone: str = "neutral") -> str:
    value_text = f"<strong>{_html(value)}</strong>" if str(value).strip() else ""
    return f'<span class="cg-chip cg-chip-{_html(tone)}">{_html(label)}{value_text}</span>'


def status_pill_html(label: str, tone: str = "green") -> str:
    return f'<span class="cg-status-pill cg-pill-{_html(tone)}">{_html(label)}</span>'


def metric_card_html(
    label: str,
    value: Any,
    caption: str = "",
    tone: str = "green",
) -> str:
    caption_html = (
        f'<div class="cg-metric-caption">{_html(caption)}</div>' if caption else ""
    )
    return f"""
<div class="cg-metric-card cg-card-{_html(tone)}">
  <div class="cg-metric-label">{_html(label)}</div>
  <div class="cg-metric-value">{_html(value)}</div>
  {caption_html}
</div>
""".strip()


def section_header_html(title: str, eyebrow: str = "", body: str = "") -> str:
    eyebrow_html = f'<div class="cg-section-eyebrow">{_html(eyebrow)}</div>' if eyebrow else ""
    body_html = f'<div class="cg-section-body">{_html(body)}</div>' if body else ""
    return f"""
<div class="cg-section-header">
  {eyebrow_html}
  <div class="cg-section-title">{_html(title)}</div>
  {body_html}
</div>
""".strip()


def page_brand_header_html(title: str, eyebrow: str = "", body: str = "", meta: str = "") -> str:
    eyebrow_html = (
        f'<div class="cg-page-brand-kicker">{_html(eyebrow)}</div>' if eyebrow else ""
    )
    body_html = f'<div class="cg-page-brand-body">{_html(body)}</div>' if body else ""
    meta_html = f'<div class="cg-page-brand-meta">{_html(meta)}</div>' if meta else ""
    return f"""
<div class="cg-page-brand">
  <div class="cg-page-brand-mark" aria-label="CogniGuard logo">
    {logo_mark_svg()}
  </div>
  <div class="cg-page-brand-copy">
    {eyebrow_html}
    <h1 class="cg-page-brand-title">{_html(title)}</h1>
    {body_html}
    {meta_html}
  </div>
</div>
""".strip()


def callout_html(title: str, body: str, tone: str = "green") -> str:
    return f"""
<div class="cg-callout cg-callout-{_html(tone)}">
  <div class="cg-callout-title">{_html(title)}</div>
  <div class="cg-callout-body">{_html(body)}</div>
</div>
""".strip()


def evidence_card_html(
    title: str,
    body: str,
    meta: str = "",
    tone: str = "neutral",
) -> str:
    meta_html = f'<div class="cg-evidence-meta">{_html(meta)}</div>' if meta else ""
    return f"""
<div class="cg-evidence-card cg-evidence-{_html(tone)}">
  <div class="cg-evidence-title">{_html(title)}</div>
  <div class="cg-evidence-body">{_html(body)}</div>
  {meta_html}
</div>
""".strip()


def timeline_item_html(
    index: Any,
    title: str,
    body: str,
    meta: str = "",
    tone: str = "green",
) -> str:
    meta_html = f'<div class="cg-timeline-meta">{_html(meta)}</div>' if meta else ""
    return f"""
<div class="cg-timeline-item cg-timeline-{_html(tone)}">
  <div class="cg-timeline-index">{_html(index)}</div>
  <div class="cg-timeline-content">
    <div class="cg-timeline-title">{_html(title)}</div>
    <div class="cg-timeline-body">{_html(body)}</div>
    {meta_html}
  </div>
</div>
""".strip()


def status_strip_html(items: Iterable[dict[str, Any]]) -> str:
    rendered_items = []
    for item in items:
        label = _html(item.get("label", ""))
        value = _html(item.get("value", ""))
        tone = _html(item.get("tone", "green"))
        rendered_items.append(
            f"""
<div class="cg-status-item cg-status-{tone}">
  <span>{label}</span>
  <strong>{value}</strong>
</div>
""".strip()
        )
    return '<div class="cg-status-strip">' + "".join(rendered_items) + "</div>"


def _base_theme_css() -> str:
    """Shared visual baseline for Streamlit pages — Warm Editorial."""
    return f"""
<style>
/* 本地自托管字体（Streamlit 静态托管：app/static/...）。缺失时回落系统字体。 */
@font-face {{
    font-family: "Noto Serif SC";
    src: url("app/static/fonts/NotoSerifSC.woff2") format("woff2");
    font-weight: 700;
    font-display: swap;
}}
@font-face {{
    font-family: "Noto Sans SC";
    src: url("app/static/fonts/NotoSansSC.woff2") format("woff2");
    font-weight: 400;
    font-display: swap;
}}
@font-face {{
    font-family: "Noto Sans SC";
    src: url("app/static/fonts/NotoSansSC-Medium.woff2") format("woff2");
    font-weight: 500;
    font-display: swap;
}}
:root {{
    --cg-green: {BRAND_GREEN};
    --cg-green-dark: {BRAND_GREEN_DARK};
    --cg-green-soft: {BRAND_GREEN_SOFT};
    --cg-navy: {TITLE_NAVY};
    --cg-navy-muted: {NAVY_MUTED};
    --cg-bg: {PAGE_BG};
    --cg-panel: {PANEL_BG};
    --cg-card: {CARD_BG};
    --cg-terracotta: {ACCENT_TERRACOTTA};
    --cg-terracotta-soft: {ACCENT_TERRACOTTA_SOFT};
    --cg-amber: {ACCENT_AMBER};
    --cg-amber-soft: {ACCENT_AMBER_SOFT};
    --cg-blue: {ACCENT_BLUE};
    --cg-blue-soft: {ACCENT_BLUE_SOFT};
    --cg-muted: {TEXT_MUTED};
    --cg-soft: {TEXT_SOFT};
    --cg-border: {BORDER_SOFT};
    --cg-border-cool: {BORDER_COOL};
    --cg-radius: 13px;
    --cg-radius-sm: 10px;
    --cg-serif: "Noto Serif SC", Georgia, "Songti SC", "SimSun", serif;
    --cg-sans: "Noto Sans SC", system-ui, -apple-system, "Segoe UI", "PingFang SC",
        "Microsoft YaHei", sans-serif;
}}
html, body, .stApp, [class*="css"] {{
    font-family: var(--cg-sans);
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
    text-rendering: optimizeLegibility;
}}
.stApp {{
    color: var(--cg-navy);
    /* 纸面：底色 + 顶部极淡暖光，去掉「纯平面」的 AI 感 */
    background-color: var(--cg-bg);
    background-image:
        radial-gradient(1200px 480px at 18% -8%, rgba(92, 122, 107, 0.07), transparent 60%),
        radial-gradient(900px 420px at 100% 0%, rgba(181, 101, 74, 0.05), transparent 55%);
    background-attachment: fixed;
}}
/* 收掉 Streamlit 自带外壳：顶部彩条 / 工具栏 / 主菜单 / 页脚 —— 去模板感关键一步 */
[data-testid="stHeader"] {{
    background: transparent;
}}
[data-testid="stDecoration"] {{ display: none; }}
[data-testid="stToolbar"] {{ right: 0.5rem; }}
#MainMenu {{ visibility: hidden; }}
footer {{ visibility: hidden; }}
[data-testid="stStatusWidget"] {{ display: none; }}
div[data-testid="stAppViewContainer"] .main .block-container,
section[data-testid="stMain"] .block-container,
div[data-testid="stMainBlockContainer"],
section.main > div {{
    max-width: min(1840px, calc(100vw - clamp(3rem, 8vw, 8rem)));
    padding-top: clamp(0.45rem, 1vw, 0.85rem);
    padding-bottom: 4rem;
}}
/* 可访问性：统一柔和焦点环，替代被 Streamlit 默认弱化的焦点态 */
:where(button, a, input, textarea, select, [tabindex]):focus-visible {{
    outline: 2px solid var(--cg-green);
    outline-offset: 2px;
    border-radius: var(--cg-radius-sm);
}}
/* 字重三档：标题 700（衬线）/ 强调 500-600 / 正文 400 */
h1, h2, h3, h4 {{
    font-family: var(--cg-serif);
    color: var(--cg-navy);
    font-weight: 700;
    letter-spacing: -0.01em;
}}
h1 {{ letter-spacing: -0.02em; line-height: 1.18; }}
p, li, label, span {{
    letter-spacing: 0;
    font-weight: 400;
}}
.stApp p, .stApp li {{
    line-height: 1.72;
}}
hr {{
    border: none;
    border-top: 1px solid var(--cg-border);
    margin: 1.6rem 0;
}}
/* 次按钮：纸白底 + 发丝边 + 极轻投影，悬停才点亮主色并微抬 */
div.stButton > button {{
    border-radius: var(--cg-radius-sm);
    min-height: 2.95rem;
    padding: 0.52rem 1.3rem;
    font-weight: 600;
    letter-spacing: 0.01em;
    cursor: pointer;
    white-space: normal;
    overflow-wrap: anywhere;
    border: 1px solid var(--cg-border);
    background: var(--cg-card);
    color: var(--cg-navy);
    box-shadow: 0 1px 2px rgba(31, 36, 33, 0.05);
    transition: background 0.18s ease, border-color 0.18s ease,
        transform 0.12s ease, box-shadow 0.18s ease, color 0.18s ease;
}}
div.stButton > button:hover {{
    border-color: var(--cg-green);
    color: var(--cg-green-dark);
    background: var(--cg-green-soft);
    transform: translateY(-1px);
    box-shadow: 0 5px 14px rgba(92, 122, 107, 0.14);
}}
div.stButton > button:active {{
    transform: translateY(0);
    box-shadow: 0 1px 2px rgba(31, 36, 33, 0.05);
}}
/* 主按钮：克制哑光渐变 + 顶部内高光 + 分层投影，质感而不浮夸 */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="stBaseButton-primary"] {{
    background: linear-gradient(180deg, #647F70 0%, #54705F 100%);
    border-color: transparent;
    color: #ffffff;
    box-shadow: 0 4px 14px rgba(60, 84, 70, 0.20),
        0 1px 3px rgba(60, 84, 70, 0.14);
}}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="stBaseButton-primary"]:hover {{
    background: linear-gradient(180deg, #56735F 0%, #466050 100%);
    border-color: transparent;
    color: #ffffff;
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(60, 84, 70, 0.26),
        0 2px 6px rgba(60, 84, 70, 0.18);
}}
div.stButton > button[kind="primary"]:active,
div.stButton > button[data-testid="stBaseButton-primary"]:active {{
    transform: translateY(0);
    box-shadow: inset 0 1px 2px rgba(43, 62, 51, 0.25),
        0 4px 12px rgba(60, 84, 70, 0.22);
}}
/* 首页主行动按钮：与左侧文案对齐的宽度、稳重圆角、导向箭头 */
.cg-home-cta-hint {{
    color: var(--cg-soft);
    font-size: 0.92rem;
    font-weight: 500;
    letter-spacing: 0.01em;
    margin: 1.5rem 0 0.55rem;
}}
.st-key-home_start_chat {{
    max-width: 560px;
}}
.st-key-home_start_chat div.stButton > button {{
    min-height: 4.4rem;
    border-radius: 18px;
    font-size: clamp(1.3rem, 1.9vw, 1.72rem);
    font-weight: 600;
    letter-spacing: 0.02em;
    box-shadow: 0 10px 28px rgba(60, 84, 70, 0.22),
        0 2px 6px rgba(60, 84, 70, 0.14);
}}
.st-key-home_start_chat div.stButton > button::after {{
    content: "→";
    display: inline-block;
    margin-left: 0.7rem;
    font-weight: 400;
    transform: translateX(0);
    transition: transform 0.2s ease;
}}
.st-key-home_start_chat div.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 16px 38px rgba(60, 84, 70, 0.28),
        0 3px 10px rgba(60, 84, 70, 0.18);
}}
.st-key-home_start_chat div.stButton > button:hover::after {{
    transform: translateX(5px);
}}
.st-key-home_brief_entry div.stButton > button,
.st-key-home_clock_entry div.stButton > button,
.st-key-home_interview_test_entry div.stButton > button,
.st-key-home_classroom_entry div.stButton > button {{
    min-height: 3.35rem;
    border-radius: 14px;
    border: 1.5px solid #D8CCBA;
    background: linear-gradient(180deg, #FFFDF8 0%, #F8F1E6 100%);
    color: var(--cg-navy);
    font-weight: 700;
    box-shadow: 0 8px 22px rgba(31, 36, 33, 0.06);
}}
.st-key-home_brief_entry div.stButton > button {{
    border-color: #4E6B5A;
    background: linear-gradient(180deg, #668574 0%, #526F5F 100%);
    color: #ffffff;
    box-shadow: 0 10px 24px rgba(60, 84, 70, 0.22);
}}
.st-key-home_clock_entry div.stButton > button {{
    border-left: 5px solid var(--cg-green);
}}
.st-key-home_interview_test_entry div.stButton > button {{
    border-left: 5px solid var(--cg-amber);
}}
.st-key-home_classroom_entry div.stButton > button {{
    border-left: 5px solid var(--cg-terracotta);
}}
.st-key-home_brief_entry div.stButton > button:hover,
.st-key-home_clock_entry div.stButton > button:hover,
.st-key-home_interview_test_entry div.stButton > button:hover,
.st-key-home_classroom_entry div.stButton > button:hover {{
    transform: translateY(-1px);
    border-color: var(--cg-green);
    box-shadow: 0 12px 28px rgba(92, 122, 107, 0.16);
}}
.st-key-quick_normal_example_answer div.stButton > button,
.st-key-quick_mild_example_answer div.stButton > button,
.st-key-quick_vague_example_answer div.stButton > button {{
    min-height: 3.12rem;
    border-radius: 12px;
    border: 1px solid transparent;
    color: var(--cg-navy-muted);
    font-weight: 650;
    box-shadow: 0 5px 14px rgba(31, 36, 33, 0.05);
}}
.st-key-quick_normal_example_answer div.stButton > button {{
    background: #EAF2ED;
    border-color: #C8D9CD;
    color: #345743;
}}
.st-key-quick_mild_example_answer div.stButton > button {{
    background: #F4EBD5;
    border-color: #DDC99B;
    color: #72531E;
}}
.st-key-quick_vague_example_answer div.stButton > button {{
    background: #F2E4DE;
    border-color: #D9B8AA;
    color: #7A3D2C;
}}
.st-key-quick_normal_example_answer div.stButton > button:hover,
.st-key-quick_mild_example_answer div.stButton > button:hover,
.st-key-quick_vague_example_answer div.stButton > button:hover {{
    color: var(--cg-navy);
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(31, 36, 33, 0.085);
}}
.st-key-quick_normal_example_answer div.stButton > button:hover {{
    background: #E4EEE7;
    border-color: #B9D0C0;
}}
.st-key-quick_mild_example_answer div.stButton > button:hover {{
    background: #EFE3C9;
    border-color: #D5BE87;
}}
.st-key-quick_vague_example_answer div.stButton > button:hover {{
    background: #EEDBD4;
    border-color: #D0AA9A;
}}
div[data-testid="stExpander"] {{
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    background: rgba(255, 255, 255, 0.55);
    box-shadow: none;
}}
div[data-testid="stExpander"] summary {{
    font-weight: 600;
    color: var(--cg-navy);
    padding: 0.85rem 1.05rem;
}}
div[data-testid="stExpander"] summary:hover {{
    color: var(--cg-green-dark);
}}
/* 页面跳转链接：从默认样式改为克制的主色文字链接 */
[data-testid="stPageLink"] a {{
    color: var(--cg-green-dark) !important;
    font-weight: 600;
    border-radius: var(--cg-radius-sm);
    display: inline-flex;
    align-items: center;
    width: fit-content;
    min-height: 2.7rem;
    padding: 0.5rem 0.9rem;
    border: 1.5px solid #D8CCBA;
    background: rgba(255, 253, 248, 0.82);
    box-shadow: 0 6px 16px rgba(31, 36, 33, 0.045);
    transition: background 0.18s ease, border-color 0.18s ease,
        box-shadow 0.18s ease, color 0.18s ease, transform 0.12s ease;
}}
[data-testid="stPageLink"] a:hover {{
    background: var(--cg-green-soft);
    border-color: var(--cg-green);
    color: var(--cg-green-dark) !important;
    transform: translateY(-1px);
    box-shadow: 0 10px 22px rgba(92, 122, 107, 0.12);
}}
[data-testid="stPageLink"] a p {{
    font-weight: 600;
}}
div[data-testid="stTextInput"] [data-baseweb="input"],
div[data-testid="stNumberInput"] [data-baseweb="input"],
div[data-testid="stTextArea"] [data-baseweb="textarea"],
div[data-testid="stSelectbox"] [data-baseweb="select"] {{
    background: #FFFDF7;
    border: 1.5px solid #D5CCBA;
    border-radius: var(--cg-radius-sm);
    box-shadow: 0 1px 2px rgba(31, 36, 33, 0.04);
    transition: border-color 0.18s ease, box-shadow 0.18s ease, background 0.18s ease;
}}
div[data-testid="stTextInput"] [data-baseweb="input"]:hover,
div[data-testid="stNumberInput"] [data-baseweb="input"]:hover,
div[data-testid="stTextArea"] [data-baseweb="textarea"]:hover,
div[data-testid="stSelectbox"] [data-baseweb="select"]:hover {{
    border-color: #BFB6A2;
    background: #FFFCF5;
}}
div[data-testid="stTextInput"] [data-baseweb="input"]:focus-within,
div[data-testid="stNumberInput"] [data-baseweb="input"]:focus-within,
div[data-testid="stTextArea"] [data-baseweb="textarea"]:focus-within,
div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within {{
    border-color: var(--cg-green);
    box-shadow: 0 0 0 3px rgba(92, 122, 107, 0.16);
    background: var(--cg-card);
}}
div[data-testid="stTextInput"] input,
div[data-testid="stNumberInput"] input,
div[data-testid="stTextArea"] textarea {{
    min-height: 2.85rem;
    color: var(--cg-navy);
    font-weight: 500;
    caret-color: var(--cg-green-dark);
}}
div[data-testid="stTextInput"] input::placeholder,
div[data-testid="stNumberInput"] input::placeholder,
div[data-testid="stTextArea"] textarea::placeholder {{
    color: var(--cg-soft);
    opacity: 1;
}}
div[data-testid="stTextInput"] [data-baseweb="input"] button {{
    color: var(--cg-navy-muted);
}}
div[data-testid="InputInstructions"] {{
    display: none;
}}
div[data-testid="stFileUploader"] section,
div[data-testid="stCameraInput"] section {{
    border: 1.5px dashed #BFB6A6;
    border-radius: var(--cg-radius-sm);
    background: rgba(255, 253, 247, 0.68);
    transition: border-color 0.18s ease, background 0.18s ease;
}}
div[data-testid="stFileUploader"] section:hover,
div[data-testid="stCameraInput"] section:hover {{
    border-color: var(--cg-green);
    background: #FFFDF7;
}}
.cg-camera-capture-guide {{
    border: 1px solid #C9DBCF;
    border-radius: var(--cg-radius);
    background: linear-gradient(180deg, #F5FAF6 0%, #EEF5EF 100%);
    padding: 0.95rem 1.05rem;
    margin: 0.75rem 0 0.55rem;
    box-shadow: 0 8px 22px rgba(92, 122, 107, 0.08);
}}
.cg-camera-capture-title {{
    color: var(--cg-green-dark);
    font-family: var(--cg-serif);
    font-size: 1.15rem;
    font-weight: 700;
    line-height: 1.25;
}}
.cg-camera-capture-body {{
    color: var(--cg-muted);
    font-size: 0.98rem;
    line-height: 1.58;
    margin-top: 0.22rem;
}}
.st-key-clock_camera_capture div[data-testid="stCameraInput"] {{
    border: 1px solid #D6E3D9;
    border-radius: 16px;
    background: rgba(255, 255, 255, 0.68);
    padding: clamp(0.85rem, 1.8vw, 1.2rem);
    box-shadow: 0 12px 30px rgba(31, 36, 33, 0.055);
}}
.st-key-clock_camera_capture div[data-testid="stCameraInput"] section {{
    border-style: solid;
    border-color: #C7D9CC;
    border-radius: 14px;
    background: #FFFDF8;
}}
.st-key-clock_camera_capture div[data-testid="stCameraInput"] label p {{
    color: var(--cg-navy);
    font-size: 1.05rem;
    font-weight: 700;
}}
.st-key-clock_camera_capture div[data-testid="stCameraInput"] button {{
    width: 100%;
    min-height: 4.35rem;
    border-radius: 16px;
    border: 1px solid transparent;
    background: linear-gradient(180deg, #647F70 0%, #54705F 100%);
    color: #ffffff;
    font-size: clamp(1.12rem, 2.2vw, 1.34rem);
    font-weight: 700;
    letter-spacing: 0;
    box-shadow: 0 10px 24px rgba(60, 84, 70, 0.22),
        0 2px 6px rgba(60, 84, 70, 0.14);
    transition: background 0.18s ease, transform 0.12s ease,
        box-shadow 0.18s ease;
}}
.st-key-clock_camera_capture div[data-testid="stCameraInput"] [data-testid="stCameraInputButton"] {{
    font-size: 0;
}}
.st-key-clock_camera_capture div[data-testid="stCameraInput"] [data-testid="stCameraInputButton"] > span {{
    display: none;
}}
.st-key-clock_camera_capture div[data-testid="stCameraInput"] [data-testid="stCameraInputButton"]::after {{
    content: "拍照";
    font-size: clamp(1.12rem, 2.2vw, 1.34rem);
    line-height: 1.25;
}}
.st-key-clock_camera_capture div[data-testid="stCameraInput"]:has(img[alt="Snapshot"]) [data-testid="stCameraInputButton"]::after {{
    content: "重新拍照";
}}
.st-key-clock_camera_capture div[data-testid="stCameraInput"] button:hover {{
    background: linear-gradient(180deg, #56735F 0%, #466050 100%);
    transform: translateY(-1px);
    box-shadow: 0 14px 30px rgba(60, 84, 70, 0.26),
        0 3px 8px rgba(60, 84, 70, 0.18);
}}
.st-key-clock_camera_capture div[data-testid="stCameraInput"] button:active {{
    transform: translateY(0);
    box-shadow: inset 0 1px 2px rgba(43, 62, 51, 0.25),
        0 6px 16px rgba(60, 84, 70, 0.20);
}}
div[data-testid="stTextInput"] input:focus,
div[data-testid="stTextArea"] textarea:focus {{
    border-color: var(--cg-green);
    box-shadow: 0 0 0 2px rgba(92, 122, 107, 0.18);
}}
/* 单选（st.radio）：每个选项做成可点选的胶囊，悬停/选中点亮主色 */
div[data-testid="stRadio"] [role="radiogroup"] {{
    gap: 0.4rem;
}}
div[data-testid="stRadio"] [role="radiogroup"] > label {{
    border: 1px solid transparent;
    border-radius: 999px;
    padding: 0.3rem 0.95rem 0.3rem 0.55rem;
    margin: 0;
    transition: background 0.18s ease, border-color 0.18s ease;
}}
div[data-testid="stRadio"] [role="radiogroup"] > label:hover {{
    background: var(--cg-green-soft);
}}
div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) {{
    background: var(--cg-green-soft);
    border-color: #CBDDCB;
}}
div[data-testid="stRadio"] [role="radiogroup"] > label:has(input:checked) p {{
    color: var(--cg-green-dark);
    font-weight: 600;
}}
/* 下拉选择（st.selectbox）的浮层菜单：纸白卡 + 发丝边 + 柔和投影，去默认观感 */
div[data-baseweb="popover"] [data-baseweb="menu"],
ul[data-baseweb="menu"] {{
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius-sm);
    box-shadow: 0 14px 36px rgba(31, 36, 33, 0.12);
    padding: 0.3rem;
}}
ul[data-baseweb="menu"] li {{
    border-radius: 8px;
    transition: background 0.14s ease;
}}
ul[data-baseweb="menu"] li:hover {{
    background: var(--cg-green-soft) !important;
    color: var(--cg-green-dark) !important;
}}
ul[data-baseweb="menu"] li[aria-selected="true"] {{
    background: var(--cg-green-soft) !important;
    color: var(--cg-green-dark) !important;
    font-weight: 600;
}}
div[data-testid="stForm"] {{
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    background: var(--cg-panel);
    padding: 1.1rem 1.2rem;
    box-shadow: none;
}}
div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] * {{
    color: var(--cg-muted) !important;
}}
/* 原生提示（st.success / warning / error / info）：去掉毛坯大色块，
   改为柔和底 + 细左线 + 发丝边的克制提示条，按类型上语义色。 */
[data-testid="stAlertContainer"] {{
    border-radius: var(--cg-radius-sm) !important;
    border: 1px solid var(--cg-border);
    border-left: 3px solid var(--cg-soft);
    box-shadow: none;
    padding: 0.85rem 1rem !important;
    color: var(--cg-navy);
}}
[data-testid="stAlertContainer"] p {{
    color: var(--cg-navy);
    font-weight: 500;
}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
    background: {SUCCESS_SOFT};
    border-color: #CBDDCB;
    border-left-color: var(--cg-green);
}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
    background: var(--cg-amber-soft);
    border-color: #E6D2A4;
    border-left-color: var(--cg-amber);
}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{
    background: {DANGER_SOFT};
    border-color: #E3BCAF;
    border-left-color: var(--cg-terracotta);
}}
[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{
    background: var(--cg-panel);
    border-color: var(--cg-border);
    border-left-color: var(--cg-navy-muted);
}}
.cg-page-kicker {{
    color: var(--cg-terracotta);
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}
.cg-page-brand {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.55rem 0 0.95rem;
    margin: 0 0 1.05rem;
    border-bottom: 1px solid var(--cg-border);
}}
.cg-page-brand-mark {{
    width: 4rem;
    height: 4rem;
    flex: 0 0 auto;
    display: grid;
    place-items: center;
}}
.cg-page-brand-mark img,
.cg-page-brand-mark svg {{
    width: 100%;
    height: 100%;
    display: block;
    object-fit: contain;
}}
.cg-page-brand-copy {{
    min-width: 0;
}}
.cg-page-brand-kicker {{
    color: var(--cg-terracotta);
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.18rem;
}}
.cg-page-brand-title {{
    font-family: var(--cg-serif);
    color: var(--cg-navy);
    font-size: 2.35rem;
    line-height: 1.18;
    font-weight: 700;
    letter-spacing: 0;
    margin: 0;
}}
.cg-page-brand-body {{
    color: var(--cg-muted);
    font-size: 1rem;
    line-height: 1.66;
    max-width: 820px;
    margin-top: 0.35rem;
}}
.cg-page-brand-meta {{
    color: var(--cg-soft);
    font-size: 0.88rem;
    line-height: 1.55;
    margin-top: 0.45rem;
}}
/* Hero：去厚投影、去彩边，纸白面板 + 发丝边 */
.cg-hero {{
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    background: var(--cg-panel);
    padding: clamp(1.2rem, 2.4vw, 1.8rem);
    margin: 0.4rem 0 1.4rem;
}}
.cg-hero-title {{
    font-family: var(--cg-serif);
    color: var(--cg-navy);
    font-size: clamp(1.7rem, 3vw, 2.5rem);
    line-height: 1.22;
    font-weight: 700;
    margin: 0;
}}
.cg-hero-body {{
    color: var(--cg-muted);
    font-size: 1rem;
    line-height: 1.72;
    margin-top: 0.6rem;
    max-width: 820px;
}}
/* Section header：eyebrow 细标 + 衬线标题，靠间距分组 */
.cg-section-header {{
    margin: 1.8rem 0 0.9rem;
}}
.cg-section-eyebrow {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--cg-terracotta);
    font-size: 0.76rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}}
.cg-section-eyebrow::before {{
    content: "";
    width: 1.6rem;
    height: 2px;
    background: var(--cg-terracotta);
    border-radius: 2px;
}}
.cg-section-title {{
    font-family: var(--cg-serif);
    color: var(--cg-navy);
    font-size: clamp(1.2rem, 2vw, 1.6rem);
    font-weight: 700;
    line-height: 1.25;
}}
.cg-section-body {{
    color: var(--cg-muted);
    font-size: 0.98rem;
    line-height: 1.66;
    margin-top: 0.3rem;
}}
/* 指标卡：白底发丝边，无投影、无彩色顶边 */
.cg-metric-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0.85rem;
    margin: 0.85rem 0 1.2rem;
}}
.cg-metric-card {{
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    padding: 1.1rem 1.15rem;
    min-height: 7rem;
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}}
.cg-metric-card:hover {{
    border-color: #D4CDBC;
    box-shadow: 0 8px 24px rgba(31, 36, 33, 0.05);
}}
/* tone 仅保留一条细顶线表达语义，去掉粗 4px 彩条 */
.cg-card-green {{ border-top: 2px solid var(--cg-green); }}
.cg-card-blue {{ border-top: 2px solid var(--cg-green); }}
.cg-card-amber {{ border-top: 2px solid var(--cg-amber); }}
.cg-card-terracotta {{ border-top: 2px solid var(--cg-terracotta); }}
.cg-card-neutral {{ border-top: 2px solid #C9C3B4; }}
.cg-metric-label {{
    color: var(--cg-muted);
    font-size: 0.84rem;
    font-weight: 500;
    margin-bottom: 0.5rem;
}}
.cg-metric-value {{
    font-family: var(--cg-serif);
    color: var(--cg-navy);
    font-size: clamp(1.5rem, 2.6vw, 2.1rem);
    font-weight: 700;
    line-height: 1.12;
    overflow-wrap: anywhere;
}}
.cg-metric-caption {{
    color: var(--cg-muted);
    font-size: 0.85rem;
    line-height: 1.5;
    margin-top: 0.55rem;
}}
/* Pill / badge / chip：哑光、去发光 */
.cg-risk-badge,
.cg-status-pill,
.cg-chip {{
    display: inline-flex;
    align-items: center;
    width: fit-content;
    min-height: 1.9rem;
    border-radius: 999px;
    padding: 0 0.78rem;
    font-size: 0.86rem;
    font-weight: 500;
    line-height: 1;
    margin: 0.15rem 0.25rem 0.15rem 0;
    white-space: normal;
    overflow-wrap: anywhere;
}}
.cg-risk-low,
.cg-pill-green,
.cg-chip-green {{
    background: {SUCCESS_SOFT};
    color: #2C4A35;
    border: 1px solid #CBDDCB;
}}
.cg-risk-medium,
.cg-pill-amber,
.cg-chip-amber {{
    background: var(--cg-amber-soft);
    color: #7A5212;
    border: 1px solid #E6D2A4;
}}
.cg-risk-high,
.cg-pill-red,
.cg-chip-red {{
    background: {DANGER_SOFT};
    color: #8A3322;
    border: 1px solid #E3BCAF;
}}
.cg-risk-unknown,
.cg-pill-neutral,
.cg-chip-neutral {{
    background: {UNKNOWN_SOFT};
    color: #565B52;
    border: 1px solid #D8D2C4;
}}
.cg-pill-blue,
.cg-chip-blue {{
    background: var(--cg-green-soft);
    color: #2C4A35;
    border: 1px solid #CBDDCB;
}}
.cg-pill-terracotta,
.cg-chip-terracotta {{
    background: var(--cg-terracotta-soft);
    color: #8A4030;
    border: 1px solid #E6CABE;
}}
.cg-chip strong {{
    margin-left: 0.4rem;
    color: inherit;
    font-weight: 600;
}}
/* Status strip：白卡发丝边 */
.cg-status-strip {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.7rem;
    margin: 0.85rem 0 1.2rem;
}}
.cg-status-item {{
    border-radius: var(--cg-radius);
    border: 1px solid var(--cg-border);
    background: var(--cg-card);
    padding: 0.85rem 0.95rem;
}}
.cg-status-item span {{
    display: block;
    color: var(--cg-muted);
    font-size: 0.78rem;
    font-weight: 500;
}}
.cg-status-item strong {{
    display: block;
    font-family: var(--cg-serif);
    color: var(--cg-navy);
    font-size: 1.08rem;
    font-weight: 700;
    margin-top: 0.22rem;
    overflow-wrap: anywhere;
}}
/* Callout：单根细左线表达语义，背景近底色 */
.cg-callout {{
    border-radius: var(--cg-radius-sm);
    border: 1px solid var(--cg-border);
    border-left: 3px solid var(--cg-green);
    padding: 1rem 1.1rem;
    margin: 0.75rem 0;
    background: var(--cg-card);
}}
.cg-callout-green {{ border-left-color: var(--cg-green); }}
.cg-callout-blue {{ border-left-color: var(--cg-green); }}
.cg-callout-amber {{ border-left-color: var(--cg-amber); }}
.cg-callout-terracotta {{ border-left-color: var(--cg-terracotta); }}
.cg-callout-title {{
    color: var(--cg-navy);
    font-weight: 600;
    margin-bottom: 0.35rem;
}}
.cg-callout-body {{
    color: var(--cg-muted);
    line-height: 1.66;
}}
/* Evidence 卡 */
.cg-evidence-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
    gap: 0.85rem;
    margin: 0.85rem 0;
}}
.cg-evidence-card {{
    border: 1px solid var(--cg-border);
    border-left: 3px solid var(--cg-green);
    border-radius: var(--cg-radius-sm);
    background: var(--cg-card);
    padding: 1rem 1.05rem;
    min-height: 6rem;
    box-shadow: 0 5px 16px rgba(31, 36, 33, 0.035);
}}
.cg-evidence-green {{
    background: #F5FAF6;
    border-color: #D7E5DA;
    border-left-color: var(--cg-green);
}}
.cg-evidence-blue {{
    background: #F4F8FC;
    border-color: #D7E3EE;
    border-left-color: #547895;
}}
.cg-evidence-amber {{
    background: #FCF8EE;
    border-color: #E9D9B8;
    border-left-color: var(--cg-amber);
}}
.cg-evidence-terracotta {{
    background: #FCF4EF;
    border-color: #E9D0C5;
    border-left-color: var(--cg-terracotta);
}}
.cg-evidence-neutral {{
    background: #FFFEFA;
    border-color: var(--cg-border);
    border-left-color: var(--cg-border-cool);
}}
.cg-evidence-orientation {{
    background: #F4F8FC;
    border-color: #D7E3EE;
    border-left-color: #547895;
}}
.cg-evidence-memory {{
    background: #F5FAF6;
    border-color: #D7E5DA;
    border-left-color: var(--cg-green);
}}
.cg-evidence-language {{
    background: #FCF8EE;
    border-color: #E9D9B8;
    border-left-color: var(--cg-amber);
}}
.cg-evidence-executive {{
    background: #FCF4EF;
    border-color: #E9D0C5;
    border-left-color: var(--cg-terracotta);
}}
.cg-evidence-attention {{
    background: #F5F7FA;
    border-color: #D7DEE8;
    border-left-color: var(--cg-navy-muted);
}}
.cg-evidence-visuospatial {{
    background: #F1F8F6;
    border-color: #CFE4DF;
    border-left-color: #4F8A7A;
}}
.cg-evidence-title {{
    color: var(--cg-navy);
    font-weight: 600;
    margin-bottom: 0.35rem;
}}
.cg-evidence-body {{
    color: var(--cg-muted);
    font-size: 0.96rem;
    line-height: 1.66;
}}
.cg-evidence-meta {{
    color: var(--cg-soft);
    font-size: 0.82rem;
    margin-top: 0.55rem;
}}
/* Trend chart：简报页专用趋势卡，低噪音浅表面 + 常显数值摘要 */
.st-key-brief_trend_card [data-testid="stVerticalBlockBorderWrapper"] {{
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    background: linear-gradient(180deg, #FFFDF8 0%, #FBF8F0 100%);
    box-shadow: 0 10px 28px rgba(31, 36, 33, 0.045);
}}
.cg-trend-card {{
    background: transparent;
    border: 0;
    padding: 0;
    margin: 0.15rem 0 0.2rem;
}}
.cg-trend-card-head {{
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    margin-bottom: 0.85rem;
}}
.cg-trend-card-title {{
    color: var(--cg-navy);
    font-family: var(--cg-serif);
    font-size: clamp(1.12rem, 2vw, 1.38rem);
    font-weight: 700;
    line-height: 1.25;
}}
.cg-trend-card-caption {{
    color: var(--cg-muted);
    font-size: 0.9rem;
    line-height: 1.58;
    margin-top: 0.35rem;
}}
.cg-trend-stat-grid {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.65rem;
    margin: 0.75rem 0 0.65rem;
}}
.cg-trend-stat {{
    border: 1px solid #E1D8C7;
    border-radius: var(--cg-radius-sm);
    background: rgba(255, 255, 255, 0.72);
    padding: 0.72rem 0.78rem;
}}
.cg-trend-stat-label {{
    color: var(--cg-soft);
    font-size: 0.76rem;
    font-weight: 500;
    line-height: 1.4;
}}
.cg-trend-stat-value {{
    color: var(--cg-navy);
    font-family: var(--cg-serif);
    font-size: clamp(1.18rem, 2.2vw, 1.55rem);
    font-weight: 700;
    line-height: 1.18;
    margin-top: 0.22rem;
    overflow-wrap: anywhere;
}}
.cg-trend-stat-note {{
    color: var(--cg-muted);
    font-size: 0.78rem;
    line-height: 1.45;
    margin-top: 0.2rem;
}}
@media (max-width: 760px) {{
    .cg-trend-card-head {{
        display: block;
    }}
    .cg-trend-stat-grid {{
        grid-template-columns: 1fr;
    }}
}}
/* Timeline */
.cg-timeline {{
    display: grid;
    gap: 0.75rem;
    margin: 0.85rem 0;
}}
.cg-timeline-item {{
    display: grid;
    grid-template-columns: 2.6rem minmax(0, 1fr);
    gap: 0.85rem;
    align-items: start;
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    background: var(--cg-card);
    padding: 0.95rem;
}}
.cg-timeline-index {{
    width: 2.3rem;
    height: 2.3rem;
    border-radius: 999px;
    display: grid;
    place-items: center;
    background: var(--cg-green-soft);
    color: var(--cg-green-dark);
    font-family: var(--cg-serif);
    font-weight: 700;
}}
.cg-timeline-title {{
    color: var(--cg-navy);
    font-weight: 600;
    margin-bottom: 0.25rem;
}}
.cg-timeline-body {{
    color: var(--cg-muted);
    line-height: 1.6;
}}
.cg-timeline-meta {{
    color: var(--cg-soft);
    font-size: 0.84rem;
    margin-top: 0.5rem;
}}
.cg-action-bar {{
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    background: rgba(255, 255, 255, 0.55);
    padding: 1rem;
    margin: 0.85rem 0 1.2rem;
}}
.cg-workbench {{
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    background: var(--cg-card);
    padding: 1.15rem;
    margin: 0.85rem 0;
}}
.cg-preview-panel {{
    border: 1px dashed #C9C3B4;
    border-radius: var(--cg-radius);
    background: var(--cg-panel);
    padding: 1rem;
    min-height: 15rem;
}}
.cg-chat-turn {{
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    background: var(--cg-card);
    padding: 1rem;
    margin: 0.6rem 0;
}}
.cg-chat-role {{
    color: var(--cg-terracotta);
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    margin-bottom: 0.25rem;
}}
.cg-chat-text {{
    color: var(--cg-navy);
    line-height: 1.7;
}}
.cg-classroom-speech {{
    border-width: 1.5px;
    border-left-width: 5px;
    padding: 1.05rem 1.2rem;
    margin: 0.75rem 0 0.65rem;
    box-shadow: 0 8px 22px rgba(31, 36, 33, 0.06);
}}
.cg-classroom-speech .cg-chat-role {{
    font-size: 0.82rem;
    letter-spacing: 0.08em;
    margin-bottom: 0.38rem;
}}
.cg-classroom-speech .cg-chat-text {{
    font-size: 1.08rem;
    line-height: 1.72;
    font-weight: 600;
}}
.cg-classroom-speech-assistant {{
    background: #FFF8EA;
    border-color: #DED1B8;
    border-left-color: var(--cg-navy-muted);
}}
.cg-classroom-speech-assistant .cg-chat-role {{
    color: var(--cg-navy);
}}
.cg-classroom-speech-elder {{
    background: #E8F1E8;
    border-color: #BFD5C4;
    border-left-color: var(--cg-green);
}}
.cg-classroom-speech-elder .cg-chat-role {{
    color: var(--cg-green-dark);
}}
/* ---------- 首页 hero（编辑式版面）---------- */
.cg-home-hero {{
    padding: 0 0 0.25rem;
}}
.cg-home-brand-row {{
    display: flex;
    align-items: center;
    gap: clamp(0.85rem, 1.5vw, 1.1rem);
    padding-bottom: clamp(0.7rem, 1.4vw, 1rem);
    margin-bottom: clamp(0.65rem, 1.25vw, 1rem);
    border-bottom: 1px solid var(--cg-border);
}}
.cg-home-logo-mark {{
    width: clamp(3.55rem, 5.8vw, 4.55rem);
    height: clamp(3.55rem, 5.8vw, 4.55rem);
    flex: 0 0 auto;
    border-radius: 0;
    display: grid;
    place-items: center;
    background: transparent;
    border: 0;
}}
.cg-home-logo-mark img,
.cg-home-logo-mark svg {{
    width: 100%;
    height: 100%;
    display: block;
    object-fit: contain;
}}
.cg-home-hero-grid {{
    display: grid;
    grid-template-columns: minmax(0, 1.15fr) minmax(320px, 0.85fr);
    gap: clamp(1.5rem, 3.8vw, 3.4rem);
    align-items: center;
}}
.cg-home-kicker {{
    color: var(--cg-terracotta);
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    margin: 0;
}}
.cg-home-brand-sub {{
    color: var(--cg-soft);
    font-size: 0.96rem;
    line-height: 1.5;
    margin-top: 0.15rem;
}}
.cg-home-title {{
    font-family: var(--cg-serif);
    color: var(--cg-navy);
    font-size: clamp(2.75rem, 5.2vw, 4.45rem);
    line-height: 1.04;
    font-weight: 700;
    letter-spacing: -0.025em;
    margin: 0;
}}
.cg-home-subtitle {{
    color: var(--cg-navy-muted);
    font-size: clamp(1.3rem, 2.25vw, 1.78rem);
    line-height: 1.34;
    font-weight: 500;
    margin: 0.8rem 0 0.75rem;
    max-width: 600px;
}}
.cg-home-copy {{
    position: relative;
    color: var(--cg-muted);
    font-size: clamp(1.02rem, 1.7vw, 1.2rem);
    line-height: 1.68;
    max-width: 560px;
    margin: 0;
    padding-left: 1.1rem;
    border-left: 2px solid var(--cg-green-soft);
}}
/* 右侧今日访谈卡：纸面之上唯一有重量的卡片，形成版面张力 */
.cg-home-visit-panel {{
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    background: var(--cg-panel);
    padding: clamp(1rem, 2.1vw, 1.35rem);
    box-shadow: 0 18px 48px rgba(31, 36, 33, 0.07);
}}
.cg-home-panel-kicker {{
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    color: var(--cg-terracotta);
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    margin-bottom: 0.45rem;
}}
.cg-home-panel-kicker::before {{
    content: "";
    width: 1.4rem;
    height: 2px;
    background: var(--cg-terracotta);
    border-radius: 2px;
}}
.cg-home-visit-name {{
    font-family: var(--cg-serif);
    color: var(--cg-navy);
    font-size: clamp(1.35rem, 2.1vw, 1.75rem);
    font-weight: 700;
    line-height: 1.22;
    margin-bottom: 0.65rem;
}}
.cg-home-visit-row {{
    display: flex;
    justify-content: space-between;
    gap: 1rem;
    border-top: 1px solid var(--cg-border);
    padding: 0.62rem 0;
    color: var(--cg-soft);
    font-size: 1rem;
}}
.cg-home-visit-row strong {{
    color: var(--cg-navy);
    font-weight: 600;
    text-align: right;
}}
.cg-home-privacy {{
    color: var(--cg-muted);
    font-size: 0.94rem;
    line-height: 1.55;
    background: var(--cg-green-soft);
    border-radius: var(--cg-radius-sm);
    padding: 0.62rem 0.78rem;
    margin-top: 0.65rem;
}}
.cg-home-headline {{
    max-width: 640px;
}}
/* 「怎么进行」整行三步带：与上方 hero 拉开间距 + 顶部发丝线分区 */
.cg-home-how {{
    margin-top: clamp(0.85rem, 1.55vw, 1.25rem);
    padding-top: clamp(0.72rem, 1.25vw, 0.95rem);
    border-top: 1px solid var(--cg-border);
}}
.cg-home-how .cg-section-eyebrow {{
    margin-bottom: clamp(0.75rem, 1.35vw, 1rem);
}}
/* 步骤：去盒子，改顶部细引线 + 悬挂式衬线序号，像版面栏目 */
.cg-home-steps {{
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: clamp(1.4rem, 3vw, 2.6rem);
}}
.cg-home-step {{
    border: none;
    border-top: 2px solid var(--cg-navy);
    border-radius: 0;
    background: transparent;
    padding: 0.68rem 0 0;
    min-height: auto;
}}
.cg-home-step-number {{
    font-family: var(--cg-serif);
    color: var(--cg-terracotta);
    font-size: 1.6rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.42rem;
}}
.cg-home-step-title {{
    color: var(--cg-navy);
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 0.3rem;
}}
.cg-home-step-copy {{
    color: var(--cg-soft);
    font-size: 0.92rem;
    line-height: 1.6;
}}
.cg-home-staff {{
    border-top: 1px solid var(--cg-border);
    padding-top: 1.3rem;
    margin-top: 1.9rem;
}}
.profile-panel {{
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    background: var(--cg-panel);
    padding: 1.1rem 1.2rem;
    margin: 0.85rem 0 1.2rem;
}}
.profile-title {{
    font-family: var(--cg-serif);
    color: var(--cg-navy);
    font-size: 1.18rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
}}
.profile-line {{
    color: var(--cg-muted);
    font-size: 1rem;
    line-height: 1.6;
}}
@media (max-width: 760px) {{
    section.main > div {{
        padding-top: 0.7rem;
    }}
    .cg-home-hero-grid,
    .cg-home-steps {{
        grid-template-columns: 1fr;
    }}
    .cg-home-title {{
        font-size: clamp(2.4rem, 13vw, 4rem);
    }}
    .cg-page-brand {{
        align-items: flex-start;
        gap: 0.8rem;
        padding-top: 0.35rem;
    }}
    .cg-page-brand-mark {{
        width: 3.25rem;
        height: 3.25rem;
    }}
    .cg-page-brand-title {{
        font-size: 1.85rem;
    }}
    .cg-metric-grid,
    .cg-status-strip,
    .cg-evidence-grid {{
        grid-template-columns: 1fr;
    }}
    .cg-timeline-item {{
        grid-template-columns: 2.3rem minmax(0, 1fr);
    }}
}}
</style>
"""


def _elder_theme_css() -> str:
    return f"""
<style>
div.stProgress > div > div > div > div {{
    background-color: var(--cg-green);
}}
div.stProgress p {{
    color: var(--cg-navy);
    font-size: 1.25rem;
    font-weight: 600;
}}
/* 长者端大按钮：可访问性保留，字重降到 600、阴影减轻 */
div.stButton > button {{
    min-height: 5.2rem;
    font-size: clamp(1.4rem, 2.4vw, 2.05rem);
    font-weight: 600;
    border-radius: 16px;
}}
div[data-testid="stExpander"] div.stButton > button {{
    min-height: 3rem;
    font-size: 1rem;
    font-weight: 500;
    background: var(--cg-green-soft);
    border: 1px solid #CBDDCB;
    color: var(--cg-navy);
    box-shadow: none;
}}
.elder-start-screen {{
    min-height: 74vh;
    display: grid;
    align-content: center;
    gap: 1rem;
    padding: 1rem 0;
}}
.elder-title {{
    font-family: var(--cg-serif);
    font-size: clamp(3rem, 8vw, 6rem);
    line-height: 1.12;
    font-weight: 700;
    color: var(--cg-navy);
    margin: 0 0 1.1rem;
}}
.elder-big {{
    font-family: var(--cg-serif);
    font-size: clamp(1.85rem, 4.6vw, 3.75rem);
    line-height: 1.28;
    font-weight: 700;
    color: var(--cg-navy);
    margin: 0 0 0.8rem;
}}
.elder-question {{
    position: relative;
    font-family: var(--cg-serif);
    font-size: clamp(2rem, 5.1vw, 4.25rem);
    line-height: 1.28;
    font-weight: 700;
    color: var(--cg-navy);
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-left: 8px solid var(--cg-green);
    border-radius: 20px;
    padding: clamp(1.6rem, 3.2vw, 2.4rem) clamp(1.4rem, 3vw, 2.2rem)
        clamp(1.3rem, 3vw, 2.1rem);
    margin: 1rem 0;
    box-shadow: 0 16px 40px rgba(92, 122, 107, 0.08);
}}
.elder-question::before {{
    content: "“";
    position: absolute;
    top: clamp(-0.4rem, -0.4vw, -0.2rem);
    left: clamp(0.9rem, 2vw, 1.4rem);
    font-family: var(--cg-serif);
    font-size: clamp(3.5rem, 8vw, 6rem);
    line-height: 1;
    color: var(--cg-green);
    opacity: 0.22;
    pointer-events: none;
}}
.elder-soft {{
    font-size: clamp(1.22rem, 2.7vw, 2rem);
    line-height: 1.4;
    font-weight: 500;
    color: var(--cg-navy-muted);
    margin: 0.45rem 0;
}}
.elder-note {{
    font-size: clamp(1.1rem, 2.45vw, 1.75rem);
    line-height: 1.45;
    font-weight: 400;
    color: var(--cg-muted);
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-radius: 16px;
    padding: 1.1rem 1.2rem;
    margin: 0.9rem 0;
}}
.elder-screen {{
    min-height: 68vh;
    padding: 1.35rem 0 0.75rem;
    display: flex;
    flex-direction: column;
    justify-content: center;
}}
.elder-live-header {{
    padding: 0.35rem 0 0.15rem;
}}
.elder-live-question-panel,
.elder-voice-station {{
    border: 1px solid var(--cg-border);
    border-radius: 22px;
    background: rgba(255, 255, 255, 0.72);
    box-shadow: 0 18px 44px rgba(31, 36, 33, 0.065);
}}
.elder-live-question-panel {{
    padding: clamp(1rem, 2.2vw, 1.45rem);
}}
.elder-live-kicker {{
    color: var(--cg-terracotta);
    font-size: clamp(0.92rem, 1.6vw, 1.06rem);
    font-weight: 700;
    letter-spacing: 0.12em;
    margin-bottom: 0.65rem;
}}
.elder-live-question-panel .elder-big {{
    font-size: clamp(1.35rem, 2.8vw, 2.2rem);
    margin-bottom: 0.65rem;
}}
.elder-live-question-panel .elder-question {{
    font-size: clamp(1.72rem, 3.85vw, 3.35rem);
    min-height: clamp(9.5rem, 26vh, 18rem);
    display: flex;
    align-items: center;
    margin: 0.65rem 0 0.85rem;
    padding: clamp(1.35rem, 2.7vw, 2.05rem);
}}
.elder-live-brief,
.elder-live-privacy {{
    font-size: clamp(1.08rem, 2.1vw, 1.45rem);
    line-height: 1.48;
    font-weight: 600;
    color: var(--cg-navy-muted);
    margin-top: 0.35rem;
}}
.elder-live-privacy {{
    color: var(--cg-muted);
    font-weight: 500;
}}
.elder-transcript-card {{
    border: 1px solid #CFE0D3;
    border-left: 4px solid var(--cg-green);
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(255, 253, 248, 0.92), rgba(244, 248, 244, 0.86));
    box-shadow: 0 12px 28px rgba(31, 36, 33, 0.052);
    margin-top: 0.8rem;
    padding: clamp(0.85rem, 1.8vw, 1.05rem) clamp(1rem, 2vw, 1.2rem);
}}
.elder-transcript-label {{
    color: var(--cg-green-dark);
    font-size: clamp(0.92rem, 1.45vw, 1.02rem);
    font-weight: 800;
    letter-spacing: 0.08em;
    margin-bottom: 0.38rem;
}}
.elder-transcript-text {{
    color: var(--cg-navy);
    font-size: clamp(1.05rem, 1.9vw, 1.32rem);
    line-height: 1.48;
    font-weight: 650;
}}
.elder-transcript-status {{
    color: var(--cg-muted);
    font-size: clamp(0.92rem, 1.45vw, 1.04rem);
    line-height: 1.45;
    margin-top: 0.42rem;
}}
.elder-voice-station {{
    padding: clamp(0.95rem, 2vw, 1.25rem);
    margin-bottom: 0.7rem;
    background: linear-gradient(180deg, #FFFDF8 0%, #F4F8F4 100%);
    border-color: #CFE0D3;
}}
.elder-voice-title {{
    font-family: var(--cg-serif);
    color: var(--cg-green-dark);
    font-size: clamp(1.55rem, 3.2vw, 2.35rem);
    font-weight: 700;
    line-height: 1.18;
}}
.elder-voice-copy {{
    color: var(--cg-muted);
    font-size: clamp(1.02rem, 2vw, 1.28rem);
    line-height: 1.52;
    margin-top: 0.45rem;
}}
.elder-device-top {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    align-items: center;
    margin: 0.35rem 0 0.95rem;
}}
.elder-done {{
    font-family: var(--cg-serif);
    font-size: clamp(2.45rem, 6.3vw, 4.7rem);
    line-height: 1.18;
    font-weight: 700;
    color: var(--cg-green-dark);
    background: var(--cg-green-soft);
    border: 1px solid #CBDDCB;
    border-radius: 22px;
    padding: clamp(1.45rem, 3vw, 2.1rem);
}}
.elder-complete-screen {{
    padding: 1rem 0 0.75rem;
}}
.elder-next-main {{
    font-family: var(--cg-serif);
    font-size: clamp(2.25rem, 5.7vw, 4.35rem);
    line-height: 1.22;
    font-weight: 700;
    color: var(--cg-green-dark);
    background: var(--cg-green-soft);
    border: 1px solid #CBDDCB;
    border-radius: 22px;
    padding: clamp(1.35rem, 3vw, 1.85rem);
    margin: 0.9rem 0;
}}
.elder-staff-note {{
    font-size: clamp(1.08rem, 2.35vw, 1.65rem);
    line-height: 1.48;
    font-weight: 400;
    color: var(--cg-muted);
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-radius: 16px;
    padding: 1rem 1.15rem;
    margin: 0.75rem 0;
}}
@media (max-width: 760px) {{
    div.stButton > button {{
        min-height: 4.7rem;
    }}
    .elder-screen {{
        min-height: 62vh;
        padding-top: 0.8rem;
    }}
    .elder-live-question-panel .elder-question {{
        min-height: auto;
    }}
    .elder-voice-station {{
        margin-top: 0.35rem;
    }}
    .elder-question {{
        border-radius: 16px;
    }}
}}
</style>
"""


def _staff_theme_css() -> str:
    return """
<style>
h1 {
    font-family: var(--cg-serif);
    color: var(--cg-navy) !important;
    font-weight: 700;
    margin-bottom: 0.9rem;
    letter-spacing: 0.01em;
}
div[data-testid="stMetric"] {
    background: var(--cg-card);
    border: 1px solid var(--cg-border);
    border-radius: var(--cg-radius);
    padding: 0.95rem 1.1rem;
    min-height: 104px;
    box-shadow: none;
}
div[data-testid="stMetricLabel"] p {
    color: var(--cg-muted) !important;
    font-weight: 500;
}
div[data-testid="stMetricValue"] {
    font-family: var(--cg-serif);
    color: var(--cg-navy);
    font-weight: 700;
}
</style>
"""


def inject_base_theme() -> None:
    """Inject the shared theme after st.set_page_config."""
    import streamlit as st

    st.markdown(_base_theme_css(), unsafe_allow_html=True)


def inject_elder_theme() -> None:
    """Inject the warm elder-facing theme."""
    import streamlit as st

    st.markdown(_base_theme_css(), unsafe_allow_html=True)
    st.markdown(_elder_theme_css(), unsafe_allow_html=True)


def inject_staff_theme() -> None:
    """Inject the staff dashboard theme."""
    import streamlit as st

    st.markdown(_base_theme_css(), unsafe_allow_html=True)
    st.markdown(_staff_theme_css(), unsafe_allow_html=True)
