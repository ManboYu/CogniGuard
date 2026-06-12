from html import escape
from typing import Optional

import pandas as pd
import streamlit as st

from core.config import load_config
from core.llm_client import generate_family_report, generate_trend_report
from core.report import (
    build_trend_chart_rows,
    compute_cogniguard_score,
    format_session_time,
    infer_session_test_type,
    sort_sessions_chronologically,
)
from core.schemas import (
    DISCLAIMER,
    DOMAIN_LABELS,
    display_cdt_feature_value,
    display_risk_level,
    display_source,
)
from core.session_history import (
    get_current_user_profile,
    load_current_user_sessions,
    load_sessions_for_brief,
)
from core.staff_gate import hide_sidebar_nav, render_staff_gate
from core.ui import (
    callout_html,
    chip_html,
    display_model_name,
    evidence_card_html,
    inject_staff_theme,
    metric_card_html,
    page_brand_header_html,
    risk_badge_html,
    section_header_html,
    status_strip_html,
)


st.set_page_config(page_title="家属/工作人员认知简报", layout="wide")
config = load_config()
hide_sidebar_nav()
inject_staff_theme()
render_staff_gate(config)

current_user = get_current_user_profile(st.session_state)
current_display_name = current_user["display_name"]

EVIDENCE_DOMAIN_TONES = {
    "orientation": "orientation",
    "memory": "memory",
    "language": "language",
    "executive_function": "executive",
    "attention": "attention",
    "visuospatial": "visuospatial",
}


def _evidence_tone_for_domain(domain: object, fallback: str = "neutral") -> str:
    return EVIDENCE_DOMAIN_TONES.get(str(domain), fallback)


def _trend_score_number(value: object) -> Optional[float]:
    if pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _trend_delta_text(first_score: object, latest_score: object) -> str:
    first_number = _trend_score_number(first_score)
    latest_number = _trend_score_number(latest_score)
    if first_number is None or latest_number is None:
        return "暂无变化"
    delta = round(latest_number - first_number, 1)
    if delta > 0:
        return f"+{delta}"
    return str(delta)


def _trend_direction_note(first_score: object, latest_score: object) -> str:
    first_number = _trend_score_number(first_score)
    latest_number = _trend_score_number(latest_score)
    if first_number is None or latest_number is None:
        return "至少两次有效分数后显示变化"
    delta = round(latest_number - first_number, 1)
    if delta > 0:
        return "较首次记录上升"
    if delta < 0:
        return "较首次记录下降"
    return "与首次记录持平"


def _component_label_text(record: dict) -> str:
    components = record.get("components", [])
    component_labels = []
    if isinstance(components, list):
        if "dialogue" in components:
            component_labels.append("对话评估")
        if "clock" in components:
            component_labels.append("画钟测试")
    return "、".join(component_labels) if component_labels else infer_session_test_type(record)


def _score_text(score_result: dict) -> str:
    score = score_result.get("score")
    return "暂无" if score is None else f"{score} / 100"


trajectory_options = {
    "current": f"当前用户：{current_display_name}",
    "normal": "演示模拟数据：整体稳定",
    "mild_decline": "演示模拟数据：轻度下降",
    "fluctuating": "演示模拟数据：波动",
}

with st.sidebar:
    st.markdown("### 简报数据")
    selected = st.selectbox(
        "选择数据视图",
        options=list(trajectory_options.keys()),
        format_func=lambda key: trajectory_options[key],
    )
    st.caption(
        f"默认读取当前测试对象“{current_display_name}”的 SQLite 记录；"
        "兜底演示可手动切换 fixture。"
    )

brief_data = (
    load_current_user_sessions(limit=10, user_profile=current_user)
    if selected == "current"
    else load_sessions_for_brief(selected)
)
sessions = brief_data["sessions"]
sessions = sort_sessions_chronologically(sessions)

source_label = display_source(brief_data["source"])
with st.sidebar:
    st.caption(f"数据来源：{source_label}")
    st.caption(f"user_id: {brief_data['user_id']}")

st.markdown(
    page_brand_header_html(
        "认知健康风险提示简报",
        eyebrow="家属 / 工作人员简报",
        body="用于查看最近一次评估结论、后续关注建议和历史变化；仅作技术原型风险提示参考。",
        meta=f"数据视图：{trajectory_options[selected]} · 来源：{source_label}",
    ),
    unsafe_allow_html=True,
)

if not sessions:
    st.info(f"{current_display_name}暂无记录，请先在对话评估或画钟测试页保存结果。")
    st.caption(DISCLAIMER)
    st.stop()

latest = sessions[-1]
saved_object = current_display_name if selected == "current" else "演示模拟对象"
cogniguard_score = compute_cogniguard_score(latest)
latest_risk_level = str(latest.get("risk_level", "unknown"))

if len(sessions) < 2:
    trend = {
        "trend_label": "unknown",
        "summary": "记录不足，暂不生成趋势；至少需要 2-3 次记录观察变化。",
    }
    brief = generate_family_report(sessions)
    family_reminders = brief["family_reminders"]
    family_reminders.append("当前记录还不足，建议保存至少 2-3 次记录后再观察趋势变化。")
else:
    trend = generate_trend_report(sessions)
    brief = generate_family_report(sessions)
    family_reminders = brief["family_reminders"]

summary_title = (
    f"{infer_session_test_type(latest)} · {format_session_time(latest.get('created_at'))}"
)
summary_body = str(latest.get("explanation", "暂无结论说明。"))
risk_label = display_risk_level(latest_risk_level)

st.markdown("### 最近一次测试结论")
st.markdown(
    section_header_html(
        "风险摘要",
        eyebrow="Care Companion OS",
        body="优先看综合提示分、风险等级和最近一次测试结论；下方再展开证据与趋势。",
    ),
    unsafe_allow_html=True,
)
st.markdown(
    f"""
<div class="cg-hero">
  <div class="cg-page-kicker">最近测试</div>
  <h2 class="cg-hero-title">{escape(summary_title)}</h2>
  <div style="margin-top:0.55rem;">{risk_badge_html(latest_risk_level, risk_label)}</div>
  <div class="cg-hero-body">{escape(summary_body)}</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="cg-metric-grid">'
    + metric_card_html("CogniGuard 综合提示分", _score_text(cogniguard_score), cogniguard_score["band"], "green")
    + metric_card_html("风险等级", risk_label, "非诊断风险提示", "terracotta")
    + metric_card_html("最近测试", infer_session_test_type(latest), f"包含：{_component_label_text(latest)}", "blue")
    + metric_card_html("保存对象", saved_object, f"数据来源：{source_label}", "amber")
    + "</div>",
    unsafe_allow_html=True,
)

st.caption(cogniguard_score["explanation"])

st.markdown("### 下一步关注建议")
if family_reminders:
    reminder_cards = "".join(
        evidence_card_html(f"建议 {index}", item, tone="amber")
        for index, item in enumerate(family_reminders[:3], start=1)
    )
    st.markdown(f'<div class="cg-evidence-grid">{reminder_cards}</div>', unsafe_allow_html=True)
else:
    st.info("暂无家属端提醒。")

st.markdown("### 本次依据")
st.markdown(
    section_header_html(
        "依据证据",
        eyebrow="Evidence",
        body="把模型依据拆成对话观察、画钟观察和其他补充，便于讲清楚系统为什么这样提示。",
    ),
    unsafe_allow_html=True,
)

evidence = latest.get("evidence", [])
if evidence:
    dialog_evidence = [
        item for item in evidence if isinstance(item, dict) and item.get("source") == "dialog"
    ]
    clock_evidence = [
        item for item in evidence if isinstance(item, dict) and item.get("source") == "clock"
    ]
    other_evidence = [
        item for item in evidence if item not in dialog_evidence and item not in clock_evidence
    ]
    dialog_cards = []
    for item in dialog_evidence:
        domain = item.get("domain")
        label = DOMAIN_LABELS.get(domain, domain or "证据")
        dialog_cards.append(
            evidence_card_html(
                label,
                item.get("text", ""),
                meta="对话观察",
                tone=_evidence_tone_for_domain(domain, "green"),
            )
        )
    if not dialog_cards:
        dialog_cards.append(evidence_card_html("对话观察", "本次记录暂无对话证据。", tone="neutral"))

    clock_cards = []
    for item in clock_evidence:
        domain = item.get("domain")
        label = DOMAIN_LABELS.get(domain, domain or "证据")
        clock_cards.append(
            evidence_card_html(
                label,
                item.get("text", ""),
                meta="画钟观察",
                tone=_evidence_tone_for_domain(domain, "blue"),
            )
        )
    if not clock_cards:
        clock_cards.append(evidence_card_html("画钟观察", "本次记录暂无画钟证据。", tone="neutral"))

    st.markdown(
        '<div class="cg-evidence-grid">'
        + "".join(dialog_cards + clock_cards)
        + "</div>",
        unsafe_allow_html=True,
    )

    if other_evidence:
        other_cards = []
        for item in other_evidence:
            if isinstance(item, dict):
                domain = item.get("domain")
                label = DOMAIN_LABELS.get(domain, domain or "证据")
                other_cards.append(
                    evidence_card_html(
                        label,
                        item.get("text", ""),
                        tone=_evidence_tone_for_domain(domain, "amber"),
                    )
                )
            else:
                other_cards.append(evidence_card_html("补充证据", item, tone="amber"))
        st.markdown(
            '<div class="cg-evidence-grid">' + "".join(other_cards) + "</div>",
            unsafe_allow_html=True,
        )
else:
    st.info("本次记录没有可展示的证据。")

latest_clock = next(
    (
        session
        for session in reversed(sessions)
        if isinstance(session.get("clock_result"), dict)
        or isinstance(session.get("cdt_features"), dict)
    ),
    None,
)
clock_result = latest_clock.get("clock_result") if latest_clock else None
cdt_features = latest_clock.get("cdt_features") if latest_clock else None
if isinstance(clock_result, dict) or isinstance(cdt_features, dict):
    st.markdown("#### 最近画钟分析摘要")
    st.markdown(
        section_header_html(
            "画钟摘要",
            eyebrow="Clock Drawing Test",
            body="汇总最近一次画钟结构化特征，辅助解释视觉空间和执行功能线索。",
        ),
        unsafe_allow_html=True,
    )
    if latest_clock is not latest:
        st.caption("来自最近一次已保存的画钟记录。")
    clock_cards = []
    chip_items = []
    if isinstance(clock_result, dict):
        chip_items.append(chip_html("来源", display_source(clock_result.get("source", "unknown")), "blue"))
        chip_items.append(chip_html("模型", display_model_name(clock_result.get("model", "未配置")), "neutral"))
        findings = clock_result.get("clock_findings", {})
        if isinstance(findings, dict):
            clock_cards.append(
                evidence_card_html(
                    "数字布局",
                    findings.get("number_placement", "暂无"),
                    tone="blue",
                )
            )
            clock_cards.append(
                evidence_card_html(
                    "指针准确性",
                    findings.get("hand_accuracy", "暂无"),
                    tone="terracotta",
                )
            )
        cdt_features = clock_result.get("cdt_features", cdt_features)
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
                    "blue",
                ),
            ]
        )
    if chip_items:
        st.markdown("".join(chip_items), unsafe_allow_html=True)
    if clock_cards:
        st.markdown(
            '<div class="cg-evidence-grid">' + "".join(clock_cards) + "</div>",
            unsafe_allow_html=True,
        )

st.markdown("### 历史趋势与记录明细")
st.markdown(
    callout_html(
        "趋势图说明",
        "趋势图和明细用于工作人员复核，优先看上方最近结论和下一步建议。",
        tone="blue",
    ),
    unsafe_allow_html=True,
)
st.markdown("#### 趋势摘要")
st.markdown(
    status_strip_html(
        [
            {"label": "趋势判断", "value": trend["trend_label"], "tone": "blue"},
            {"label": "记录数量", "value": f"{len(sessions)} 次", "tone": "green"},
            {"label": "最近类型", "value": infer_session_test_type(latest), "tone": "amber"},
        ]
    ),
    unsafe_allow_html=True,
)
st.markdown(callout_html("趋势摘要", trend["summary"], tone="green"), unsafe_allow_html=True)

st.markdown("### CogniGuard 综合提示分趋势")
chart_rows = build_trend_chart_rows(sessions)
if len(sessions) < 2:
    st.info("记录不足，暂不生成趋势。")
else:
    chart_df = pd.DataFrame(chart_rows)
    score_column = "CogniGuard 综合提示分数值"
    if score_column in chart_df.columns and chart_df[score_column].notna().any():
        trend_df = chart_df[["display_label", score_column]].dropna(subset=[score_column]).copy()
        trend_df = trend_df.reset_index(drop=True)
        trend_df["score_label"] = trend_df[score_column].round(0).astype(int).astype(str) + "分"
        trend_df["is_latest"] = trend_df.index == len(trend_df) - 1
        x_sort_order = trend_df["display_label"].tolist()
        first_score = trend_df[score_column].iloc[0]
        latest_score = trend_df[score_column].iloc[-1]
        latest_score_text = f"{round(float(latest_score))} / 100"
        with st.container(border=True, key="brief_trend_card"):
            st.markdown(
                f"""
<div class="cg-trend-card">
  <div class="cg-trend-card-head">
    <div>
      <div class="cg-trend-card-title">综合提示分变化</div>
      <div class="cg-trend-card-caption">按保存时间从旧到新排列，最右侧为最近一次；图表用于观察演示数据的连续变化。</div>
    </div>
  </div>
  <div class="cg-trend-stat-grid">
    <div class="cg-trend-stat">
      <div class="cg-trend-stat-label">最近一次</div>
      <div class="cg-trend-stat-value">{escape(latest_score_text)}</div>
      <div class="cg-trend-stat-note">当前综合提示分</div>
    </div>
    <div class="cg-trend-stat">
      <div class="cg-trend-stat-label">首末变化</div>
      <div class="cg-trend-stat-value">{escape(_trend_delta_text(first_score, latest_score))}</div>
      <div class="cg-trend-stat-note">{escape(_trend_direction_note(first_score, latest_score))}</div>
    </div>
    <div class="cg-trend-stat">
      <div class="cg-trend-stat-label">记录数量</div>
      <div class="cg-trend-stat-value">{len(trend_df)} 次</div>
      <div class="cg-trend-stat-note">可展开查看明细</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
            st.vega_lite_chart(
                trend_df,
                {
                    "height": 300,
                    "background": "#FFFDF8",
                    "padding": {"top": 24, "right": 18, "bottom": 10, "left": 6},
                    "encoding": {
                        "x": {
                            "field": "display_label",
                            "type": "ordinal",
                            "title": "测试次序",
                            "sort": x_sort_order,
                            "axis": {
                                "labelAngle": 0,
                                "labelPadding": 8,
                                "titlePadding": 12,
                            },
                        },
                        "y": {
                            "field": score_column,
                            "type": "quantitative",
                            "title": "CogniGuard 综合提示分",
                            "scale": {"domain": [0, 100]},
                            "axis": {
                                "values": [0, 20, 40, 60, 80, 100],
                                "tickCount": 6,
                            },
                        },
                        "tooltip": [
                            {"field": "display_label", "type": "ordinal", "title": "测试次序"},
                            {"field": score_column, "type": "quantitative", "title": "综合提示分"},
                        ],
                    },
                    "layer": [
                        {
                            "mark": {
                                "type": "area",
                                "interpolate": "monotone",
                                "color": "#5C7A6B",
                                "opacity": 0.12,
                            }
                        },
                        {
                            "mark": {
                                "type": "line",
                                "interpolate": "monotone",
                                "stroke": "#5C7A6B",
                                "strokeWidth": 3,
                            }
                        },
                        {
                            "mark": {
                                "type": "circle",
                                "filled": True,
                                "stroke": "#FFFDF8",
                                "strokeWidth": 2,
                            },
                            "encoding": {
                                "size": {
                                    "condition": {"test": "datum.is_latest", "value": 130},
                                    "value": 62,
                                },
                                "color": {
                                    "condition": {"test": "datum.is_latest", "value": "#B5654A"},
                                    "value": "#5C7A6B",
                                },
                            },
                        },
                        {
                            "transform": [{"filter": "datum.is_latest"}],
                            "mark": {
                                "type": "text",
                                "align": "left",
                                "baseline": "middle",
                                "dx": 11,
                                "dy": -12,
                                "fontSize": 12,
                                "fontWeight": 700,
                                "color": "#1F2421",
                            },
                            "encoding": {"text": {"field": "score_label", "type": "nominal"}},
                        },
                    ],
                    "config": {
                        "view": {"stroke": None},
                        "axis": {
                            "domainColor": "#DDD2BF",
                            "gridColor": "#E8DFD0",
                            "gridOpacity": 0.78,
                            "labelColor": "#6A736C",
                            "labelFontSize": 12,
                            "titleColor": "#5A615C",
                            "titleFontSize": 13,
                            "tickColor": "#DDD2BF",
                        },
                        "axisX": {"grid": False},
                        "axisY": {"grid": True},
                    },
                },
                width="stretch",
            )
            st.caption("纵轴为 0-100 分；横轴按测试时间从旧到新排序，最右侧高亮点为最近一次。")
    else:
        st.info("暂无可绘制的 CogniGuard 综合提示分趋势。")

if chart_rows:
    trend_table = (
        pd.DataFrame(chart_rows)[
            ["display_label", "测试时间", "测试类型", "风险等级", "CogniGuard 综合提示分"]
        ]
        .rename(columns={"display_label": "测试次序"})
    )
    with st.expander("查看趋势明细表", expanded=False):
        st.dataframe(trend_table, width="stretch", hide_index=True)

st.markdown("### 家属端提醒")
if family_reminders:
    reminder_cards = "".join(
        evidence_card_html(f"提醒 {index}", item, tone="amber")
        for index, item in enumerate(family_reminders, start=1)
    )
    st.markdown(f'<div class="cg-evidence-grid">{reminder_cards}</div>', unsafe_allow_html=True)
else:
    st.info("暂无家属端提醒。")

with st.expander("技术信息（可选）"):
    st.write(
        {
            "session_id": latest.get("session_id", ""),
            "user_id": latest.get("user_id") or latest.get("participant_id", ""),
            "created_at": latest.get("created_at", ""),
        }
    )

st.caption(DISCLAIMER)
