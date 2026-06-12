from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from core import schemas
from core.assessment_flow import FLOW_STEP_CLOCK_TEST, build_assessment_flow_summary
from core.mock_data import (
    CLASSROOM_DEMO_LEVEL_LABELS,
    DIALOG_EXAMPLE_ANSWER_TYPES,
    FIXTURE_FILES,
    INTERVIEW_COMPLETED_MESSAGE,
    PRESET_INTERVIEW_QUESTIONS,
    all_dialog_domains_covered,
    build_classroom_clock_report,
    build_classroom_demo_interview,
    get_clock_sample_paths,
    get_covered_dialog_domains,
    get_next_preset_interview_question,
    get_dialog_example_answers,
    infer_dialog_question_type,
    load_fixture_sessions,
)
from core.report import (
    build_trend_chart_rows,
    build_family_brief,
    compute_cogniguard_score,
    compute_dialogue_score,
    format_session_time,
    generate_mock_clock_report,
    generate_mock_dialog_report,
    infer_session_test_type,
    summarize_trend,
)
from core.schemas import (
    display_cdt_feature_value,
    display_risk_level,
    display_source,
)
from core.session_history import (
    build_history_personalized_start,
    infer_history_focus_domain,
)
from core.staff_gate import is_staff_unlocked, verify_staff_password


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_schema_constants_match_project_contract() -> None:
    assert schemas.COGNITIVE_DOMAINS == (
        "orientation",
        "memory",
        "language",
        "executive_function",
        "attention",
        "visuospatial",
    )
    assert set(schemas.DOMAIN_LABELS) == set(schemas.COGNITIVE_DOMAINS)
    assert schemas.RISK_LEVELS == ("low", "medium", "high", "unknown")
    assert "技术原型" in schemas.DISCLAIMER
    assert "不构成医学诊断" in schemas.DISCLAIMER


def test_fallback_result_uses_unknown_without_fabricated_scores() -> None:
    fallback = schemas.fallback_result()

    assert set(fallback["domain_scores"]) == set(schemas.COGNITIVE_DOMAINS)
    assert all(value is None for value in fallback["domain_scores"].values())
    assert fallback["evidence"] == []
    assert fallback["risk_level"] == "unknown"
    assert fallback["explanation"] == schemas.FALLBACK_EXPLANATION
    assert fallback["disclaimer"] == schemas.DISCLAIMER

    fallback["domain_scores"]["memory"] = 0.5
    assert schemas.fallback_result()["domain_scores"]["memory"] is None


def test_fixture_sessions_match_minimum_schema() -> None:
    for trajectory in FIXTURE_FILES:
        sessions = load_fixture_sessions(trajectory)

        assert len(sessions) >= 3
        assert {session["trajectory"] for session in sessions} == {trajectory}

        for session in sessions:
            assert session["is_mock"] is True
            assert session["risk_level"] in schemas.RISK_LEVELS
            assert session["disclaimer"] == schemas.DISCLAIMER
            assert set(session["domain_scores"]) == set(schemas.COGNITIVE_DOMAINS)
            assert session["evidence"]
            assert session["explanation"]
            assert session["session_id"]
            assert session["participant_id"]
            assert session["created_at"]

            for score in session["domain_scores"].values():
                assert isinstance(score, (float, int))
                assert 0 <= score <= 1

            for item in session["evidence"]:
                assert item["domain"] in schemas.COGNITIVE_DOMAINS
                assert item["source"] in {"dialog", "clock"}
                assert item["text"]


@pytest.mark.parametrize(
    ("question", "expected_type"),
    [
        ("今天大概是星期几？您是怎么判断的？", "orientation"),
        ("您还记得今天早上吃了什么吗？", "memory"),
        ("刚才我们聊到的事情里，您还记得一件吗？", "memory"),
        ("谢谢您。小顾想先请您记住三个词：梨子、雨伞、公交卡。", "memory"),
        ("请用一句话描述一下房间里您看到的一个东西。", "language"),
        ("如果下午要出门散步，您会先准备什么？", "executive_function"),
        ("小顾今天继续陪您慢慢聊。从客厅走到厨房，通常会经过哪里？", "visuospatial"),
        ("请从 20 往回数三个数，慢慢来就好。", "attention"),
        ("从客厅走到厨房，通常会经过哪里？", "visuospatial"),
    ],
)
def test_dialog_example_answers_cover_question_types(
    question: str, expected_type: str
) -> None:
    assert infer_dialog_question_type(question) == expected_type

    answers = get_dialog_example_answers(question)
    assert set(answers) == set(DIALOG_EXAMPLE_ANSWER_TYPES)
    assert set(answers) == {"normal", "mild_decline", "vague"}
    assert all(answer.strip() for answer in answers.values())


def test_dialog_example_answers_fall_back_to_general_templates() -> None:
    answers = get_dialog_example_answers("您最近有什么想聊的吗？")

    assert set(answers) == {"normal", "mild_decline", "vague"}
    assert all(answer.strip() for answer in answers.values())


def test_dialog_example_answers_match_qwen_style_question_text() -> None:
    answers = get_dialog_example_answers("您最喜欢的季节是什么样的？", target_domain="language")

    assert "春天" in answers["normal"]
    assert "不冷不热" in answers["mild_decline"]
    assert "白色水杯" not in answers["normal"]


def test_dialog_example_answers_match_self_contained_digit_question() -> None:
    answers = get_dialog_example_answers("请您把这串数字倒着说一遍：3-8-1。", target_domain="attention")

    assert answers["normal"].startswith("1")
    assert "水杯" not in answers["normal"]


def test_dialog_example_answers_reverse_digits_from_current_question() -> None:
    answers = get_dialog_example_answers(
        "请您把这串数字倒着说一遍：7-2-5。",
        target_domain="attention",
        sample_answers={
            "normal": "7-2-5。",
            "mild_decline": "5-2-7。",
            "vague": "我不想回答。",
        },
    )

    assert answers["normal"] == "5-2-7。"
    assert answers["mild_decline"] == "5-7-2，我有点记混了。"
    assert answers["vague"] == "数字我记不住了，你再说一遍吧。"


def test_dialog_example_answers_extract_current_memory_words() -> None:
    answers = get_dialog_example_answers(
        "请记住这三个词：梨子、雨伞、公交卡。稍后我会再问您。",
        target_domain="memory",
    )

    assert "梨子、雨伞、公交卡" in answers["normal"]
    assert "梨子" in answers["mild_decline"]
    assert "苹果、钥匙、报纸" not in answers["normal"]


def test_dialog_example_answers_match_visuospatial_position_question() -> None:
    answers = get_dialog_example_answers(
        "如果我把钥匙放在水杯左边，眼镜放在水杯右边，您能说说它们的位置吗？",
        target_domain="visuospatial",
    )

    assert "左边" in answers["normal"]
    assert "右边" in answers["normal"]
    assert "位置" in answers["normal"]
    assert "洗手" not in answers["normal"]


def test_dialog_example_answers_reject_stale_fixed_qwen_samples() -> None:
    answers = get_dialog_example_answers(
        "您刚才说早上喝了粥。请记住这三个词：梨子、雨伞、公交卡。",
        target_domain="memory",
        sample_answers={
            "normal": "好的，我记住了：苹果、钥匙、报纸。",
            "mild_decline": "我记得有苹果，后面两个词有点模糊。",
            "vague": "我不太确定刚才是哪几个词。",
        },
    )

    assert "梨子、雨伞、公交卡" in answers["normal"]
    assert "苹果、钥匙、报纸" not in answers["normal"]


def test_dialog_example_answers_reject_unrelated_qwen_spatial_samples() -> None:
    answers = get_dialog_example_answers(
        "请您描述一下从客厅走到厨房时，通常会经过哪些地方？",
        target_domain="visuospatial",
        sample_answers={
            "normal": "我会先洗手，再拿起水杯喝水。",
            "mild_decline": "我可能会洗手。",
            "vague": "随便吧。",
        },
    )

    assert "客厅" in answers["normal"]
    assert "厨房" in answers["normal"]
    assert "洗手" not in answers["normal"]


def test_dialog_example_answers_reject_stale_digit_samples_for_plan_question() -> None:
    answers = get_dialog_example_answers(
        "如果今天想做点不一样的事情，您会怎么安排？",
        target_domain="executive_function",
        sample_answers={
            "normal": "1、8、3，我把它倒过来说。",
            "mild_decline": "1、3、8，我有点记混了。",
            "vague": "数字我记不住了。",
        },
    )

    assert "倒过来" not in answers["normal"]
    assert "数字" not in answers["vague"]
    assert "先" in answers["normal"] or "安排" in answers["normal"]


def test_ui_display_mappings_translate_internal_values() -> None:
    assert display_risk_level("medium") == "中等风险"
    assert display_risk_level("unknown") == "无法评估"
    assert display_source("qwen") == "Qwen 文本模型"
    assert display_source("qwen-vl") == "Qwen-VL 视觉模型"
    assert display_source("fallback") == "兜底结果"
    assert display_cdt_feature_value("number_distribution", "right_shifted") == "向右偏移"
    assert display_cdt_feature_value("number_spacing", "crowded") == "拥挤"
    assert display_cdt_feature_value("target_time_match", True) == "符合目标时间"
    assert display_cdt_feature_value("target_time_match", False) == "不符合目标时间"


def test_staff_gate_helpers_are_offline_and_config_based() -> None:
    config = SimpleNamespace(staff_password="8888")

    assert is_staff_unlocked({}) is False
    assert is_staff_unlocked({"staff_unlocked": False}) is False
    assert is_staff_unlocked({"staff_unlocked": True}) is True
    assert verify_staff_password("8888", config) is True
    assert verify_staff_password("123456", config) is False
    assert verify_staff_password(" 8888 ", config) is False


def test_preset_interview_questions_cover_each_domain_once_without_looping() -> None:
    covered: list[str] = []
    questions = []

    for _ in schemas.COGNITIVE_DOMAINS:
        item = get_next_preset_interview_question(covered)
        assert item is not None
        assert item["domain"] not in covered
        covered.append(item["domain"])
        questions.append(item["question"])

    assert set(covered) == set(schemas.COGNITIVE_DOMAINS)
    assert len(questions) == len(set(questions))
    assert "小顾" in questions[0]
    assert get_next_preset_interview_question(covered) is None


def test_covered_domains_prefer_saved_target_domain_for_history_start() -> None:
    turns = [
        {
            "assistant": "欢迎回来，张奶奶。小顾今天继续陪您慢慢聊。从客厅走到厨房，通常会经过哪里？",
            "user": "我会经过餐桌旁边，再走到厨房门口。",
            "target_domain": "visuospatial",
        }
    ]

    assert get_covered_dialog_domains(turns) == ["visuospatial"]


def test_history_personalized_start_prioritizes_low_recent_domain() -> None:
    sessions = [
        {
            "domain_scores": {
                "orientation": 0.9,
                "memory": 0.62,
                "language": 0.88,
                "executive_function": 0.86,
                "attention": 0.84,
                "visuospatial": 0.9,
            }
        }
    ]

    start = build_history_personalized_start(
        sessions,
        display_name="张奶奶",
        fallback_question=PRESET_INTERVIEW_QUESTIONS[0]["question"],
        fallback_domain=PRESET_INTERVIEW_QUESTIONS[0]["domain"],
    )

    assert infer_history_focus_domain(sessions) == "memory"
    assert start["has_history"] is True
    assert start["target_domain"] == "memory"
    assert "欢迎回来，张奶奶" in start["question"]
    assert "早上吃了什么" in start["question"]
    assert "系统" not in start["question"]
    assert "上次记录" not in start["question"]


def test_history_personalized_start_uses_general_greeting_without_history() -> None:
    start = build_history_personalized_start(
        [],
        display_name="张奶奶",
        fallback_question=PRESET_INTERVIEW_QUESTIONS[0]["question"],
        fallback_domain=PRESET_INTERVIEW_QUESTIONS[0]["domain"],
    )

    assert start["has_history"] is False
    assert start["target_domain"] == "orientation"
    assert start["question"] == PRESET_INTERVIEW_QUESTIONS[0]["question"]
    assert start["elder_hint"] == "您好，我是小顾，今天我陪您轻松聊一会儿。"


def test_interview_completion_message_after_all_domains_are_covered() -> None:
    covered = list(schemas.COGNITIVE_DOMAINS)

    assert all_dialog_domains_covered(covered) is True
    assert get_next_preset_interview_question(covered) is None
    assert "生成认知评估" in INTERVIEW_COMPLETED_MESSAGE


def test_clock_sample_images_exist_for_demo_page() -> None:
    sample_paths = get_clock_sample_paths()

    assert {"normal", "spatial_shift", "wrong_hands"} <= set(sample_paths)
    assert sample_paths["normal"].name == "classroom_clock_normal.png"
    assert sample_paths["spatial_shift"].name == "classroom_clock_mild_decline.png"
    assert sample_paths["wrong_hands"].name == "classroom_clock_obvious_issue.png"
    assert all("assets" in str(path) for path in sample_paths.values())
    for path in sample_paths.values():
        assert path.exists()
        assert path.suffix == ".png"
        assert path.stat().st_size > 0


def test_clock_page_uses_adaptive_sample_preview() -> None:
    page_text = (PROJECT_ROOT / "pages" / "2_画钟测试.py").read_text(encoding="utf-8")

    assert "正常表现示例" in page_text
    assert "轻度下降示例" in page_text
    assert "明显异常示例" in page_text
    assert "cg-clock-sample-preview" in page_text
    assert "max-height: min(62vh, 620px)" in page_text
    assert "_render_clock_preview(" in page_text
    assert "clock_analysis_in_progress" in page_text
    assert 'expanded=analysis_busy' in page_text
    assert "画钟图片正在分析，请稍等，不需要重复点击" in page_text
    assert 'st.image(uploaded_file, caption="已上传图片预览", width="stretch")' not in page_text
    assert 'caption="示例画钟预览"' not in page_text


def test_classroom_demo_interview_contains_simulated_turn_structure() -> None:
    classroom_questions: dict[str, list[str]] = {}
    for level in CLASSROOM_DEMO_LEVEL_LABELS:
        turns = build_classroom_demo_interview(level)
        classroom_questions[level] = [turn["system_question"] for turn in turns]

        assert len(turns) == 6
        assert {turn["cognitive_level"] for turn in turns} == {level}
        assert len({turn["system_question"] for turn in turns}) == 6
        clock_triggers = [turn for turn in turns if turn.get("clock_triggered")]
        if level == "正常表现":
            assert clock_triggers == []
        else:
            assert len(clock_triggers) == 1
            assert "画钟" in clock_triggers[0]["clock_trigger_title"]
            assert "拍" in clock_triggers[0]["clock_trigger_elder_message"]
        for turn in turns:
            assert turn["system_question"]
            assert turn["patient_answer"]
            assert "你" not in turn["system_question"]
            assert "您" not in turn["patient_answer"]
            assert turn["target_domain"] in schemas.COGNITIVE_DOMAINS
            assert turn["expected_risk"] in {"low", "medium", "high"}
            assert turn["is_simulated"] is True
            assert isinstance(turn["clock_triggered"], bool)
        assert "张奶奶" in turns[0]["system_question"]
        assert "您好" in turns[0]["system_question"]
        assert "您" in turns[0]["system_question"]
    assert classroom_questions["轻度下降"] != classroom_questions["明显异常"]


def test_classroom_clock_reports_cover_complete_report_inputs() -> None:
    expected_risk = {
        "正常表现": "low",
        "轻度下降": "medium",
        "明显异常": "high",
    }
    for level in CLASSROOM_DEMO_LEVEL_LABELS:
        report = build_classroom_clock_report(level, model="classroom-clock-demo")

        assert report["risk_level"] == expected_risk[level]
        assert report["is_mock"] is True
        assert report["is_simulated"] is True
        assert report["metadata"]["source"] == "mock"
        assert report["metadata"]["model"] == "classroom-clock-demo"
        assert set(report["domain_scores"]) == set(schemas.COGNITIVE_DOMAINS)
        assert report["domain_scores"]["visuospatial"] is not None
        assert report["domain_scores"]["executive_function"] is not None
        assert report["evidence"]
        assert all(item["source"] == "clock" for item in report["evidence"])
        assert report["clock_findings"]["number_placement"]
        assert report["cdt_features"]["number_distribution"] in {
            "balanced",
            "right_shifted",
            "clustered",
        }
        assert report["disclaimer"] == schemas.DISCLAIMER


@pytest.mark.parametrize(
    ("trajectory", "expected_label"),
    [
        ("normal", "稳定"),
        ("mild_decline", "下降"),
        ("fluctuating", "波动"),
    ],
)
def test_trend_summary_matches_demo_trajectory(
    trajectory: str, expected_label: str
) -> None:
    sessions = load_fixture_sessions(trajectory)
    trend = summarize_trend(sessions)
    brief = build_family_brief(sessions)

    assert trend["trend_label"] == expected_label
    assert trend["summary"]
    assert trend["disclaimer"] == schemas.DISCLAIMER
    assert brief["trend_label"] == expected_label
    assert brief["family_reminders"]
    assert "mock 趋势" not in " ".join(brief["family_reminders"])
    assert brief["disclaimer"] == schemas.DISCLAIMER


def test_family_brief_prioritizes_latest_risk_level() -> None:
    sessions = copy.deepcopy(load_fixture_sessions("normal")[:2])
    sessions[-1]["risk_level"] = "high"

    brief = build_family_brief(sessions)
    rendered_reminders = " ".join(brief["family_reminders"])

    assert "需要重点关注的信号" in rendered_reminders
    assert "不作为诊断结论" in rendered_reminders
    assert "mock 趋势" not in rendered_reminders
    assert brief["disclaimer"] == schemas.DISCLAIMER


def test_trend_summary_handles_missing_domain_scores_without_crashing() -> None:
    sessions = copy.deepcopy(load_fixture_sessions("normal")[:2])
    sessions[0]["domain_scores"]["memory"] = None
    sessions[-1]["domain_scores"]["attention"] = None

    trend = summarize_trend(sessions)

    assert trend["trend_label"] in {"稳定", "波动", "改善", "下降"}
    assert trend["domain_changes"]["memory"] is None
    assert trend["domain_changes"]["attention"] is None
    assert isinstance(trend["average_scores"], list)


def test_cogniguard_score_bounds_by_risk_level_and_handles_none_scores() -> None:
    base_scores = {
        "orientation": 0.9,
        "memory": 0.8,
        "language": None,
        "executive_function": 0.7,
        "attention": 0.8,
        "visuospatial": 0.9,
    }

    low = compute_cogniguard_score({"risk_level": "low", "domain_scores": base_scores})
    medium = compute_cogniguard_score({"risk_level": "medium", "domain_scores": base_scores})
    high = compute_cogniguard_score({"risk_level": "high", "domain_scores": base_scores})
    unknown = compute_cogniguard_score(
        {"risk_level": "unknown", "domain_scores": {domain: None for domain in schemas.COGNITIVE_DOMAINS}}
    )

    assert 75 <= low["score"] <= 100
    assert 50 <= medium["score"] <= 74
    assert high["score"] <= 49
    assert unknown["score"] is None
    assert unknown["band"] == "无法评估"
    assert "技术原型提示分" in medium["explanation"]


def test_cogniguard_score_raises_low_risk_floor() -> None:
    score = compute_cogniguard_score(
        {
            "risk_level": "low",
            "domain_scores": {domain: 0.6 for domain in schemas.COGNITIVE_DOMAINS},
        }
    )

    assert score["score"] == 75
    assert score["band"] == "轻微波动"
    assert "风险等级约束" in score["explanation"]


def test_cogniguard_score_keeps_high_low_risk_average() -> None:
    score = compute_cogniguard_score(
        {
            "risk_level": "low",
            "domain_scores": {
                "orientation": None,
                "memory": None,
                "language": None,
                "executive_function": 0.8,
                "attention": None,
                "visuospatial": 0.9,
            },
        }
    )

    assert 80 <= score["score"] <= 90
    assert score["score"] != 75


def test_dialogue_score_uses_dialog_domain_average() -> None:
    score = compute_dialogue_score(
        {
            "risk_level": "low",
            "domain_scores": {
                "orientation": 1.0,
                "memory": 0.95,
                "language": 0.95,
                "executive_function": 0.9,
                "attention": None,
                "visuospatial": None,
            },
        }
    )

    assert score["score"] == 95
    assert score["band"] == "整体稳定"
    assert "技术原型指标" in score["explanation"]


def test_cogniguard_score_combined_correct_dialogue_and_normal_clock_is_high() -> None:
    record = {
        "risk_level": "low",
        "components": ["dialogue", "clock"],
        "dialogue_result": {
            "risk_level": "low",
            "domain_scores": {domain: 0.95 for domain in schemas.COGNITIVE_DOMAINS},
        },
        "clock_result": {
            "risk_level": "low",
            "cdt_features": _normal_cdt_features(),
        },
        "domain_scores": {
            "orientation": 0.95,
            "memory": 0.95,
            "language": 0.95,
            "executive_function": 0.9,
            "attention": 0.95,
            "visuospatial": 0.9,
        },
    }

    score = compute_cogniguard_score(record)

    assert score["score"] >= 90
    assert score["band"] == "整体稳定"


def test_cogniguard_score_single_normal_clock_stays_high() -> None:
    score = compute_cogniguard_score(
        {
            "risk_level": "low",
            "clock_result": {"cdt_features": _normal_cdt_features()},
            "domain_scores": {
                "orientation": None,
                "memory": None,
                "language": None,
                "executive_function": 0.8,
                "attention": None,
                "visuospatial": 0.9,
            },
        }
    )

    assert 85 <= score["score"] <= 95


def test_trend_chart_rows_use_short_display_labels_and_times() -> None:
    sessions = list(reversed(copy.deepcopy(load_fixture_sessions("normal"))))
    expected_oldest = min(
        sessions,
        key=lambda session: session["created_at"],
    )

    rows = build_trend_chart_rows(sessions)

    assert [row["display_label"] for row in rows] == ["第1次", "第2次", "最近一次"]
    assert rows[0]["测试时间"] == format_session_time(expected_oldest["created_at"])
    assert len(rows[0]["测试时间"]) <= 16
    assert rows[0]["测试类型"] == "对话评估"
    assert rows[0]["风险等级"] in {"低风险", "中等风险", "高风险", "无法评估"}
    assert "CogniGuard 综合提示分" in rows[0]
    assert isinstance(rows[0]["CogniGuard 综合提示分数值"], int)


def test_infer_session_test_type_for_dialog_clock_and_combined_records() -> None:
    dialog = {"evidence": [{"source": "dialog"}]}
    clock = {"evidence": [{"source": "clock"}], "clock_result": {}}
    combined = {"evidence": [{"source": "dialog"}], "clock_result": {}}

    assert infer_session_test_type(dialog) == "对话评估"
    assert infer_session_test_type(clock) == "画钟测试"
    assert infer_session_test_type(combined) == "综合评估"


def test_mock_reports_include_disclaimer_and_valid_domains() -> None:
    dialog_report = generate_mock_dialog_report(["今天周六。", "早饭吃了粥。"])
    clock_report = generate_mock_clock_report("clock.png")

    assert set(dialog_report["domain_scores"]) == set(schemas.COGNITIVE_DOMAINS)
    assert dialog_report["risk_level"] in schemas.RISK_LEVELS
    assert dialog_report["disclaimer"] == schemas.DISCLAIMER
    assert dialog_report["evidence"]

    assert clock_report["risk_level"] in schemas.RISK_LEVELS
    assert clock_report["disclaimer"] == schemas.DISCLAIMER
    assert clock_report["clock_findings"]["visuospatial_evidence"]


def test_repeated_vague_mock_dialog_triggers_clock_followup() -> None:
    messages = []
    for item in PRESET_INTERVIEW_QUESTIONS:
        answer = get_dialog_example_answers(
            item["question"],
            target_domain=item["domain"],
        )["vague"]
        messages.append(f"AI访谈问题：{item['question']}")
        messages.append(f"老人回答：{answer}")

    report = generate_mock_dialog_report(messages)
    flow_summary = build_assessment_flow_summary(report)

    assert report["risk_level"] == "medium"
    assert report["domain_scores"]["executive_function"] <= 0.65
    assert report["domain_scores"]["visuospatial"] <= 0.65
    assert flow_summary["next_task"]["step_id"] == FLOW_STEP_CLOCK_TEST


def test_home_page_copy_lists_current_demo_entries() -> None:
    page_text = (PROJECT_ROOT / "app.py").read_text(encoding="utf-8")

    assert "CogniGuard" in page_text
    assert "hide_sidebar_nav()" in page_text
    assert "长者友好访谈入口 · 温柔的认知健康助手" in page_text
    assert "社康平板老人端" not in page_text
    assert "您好，我是小顾。" in page_text
    assert "我们像聊天一样，慢慢完成今天的认知健康访谈。" in page_text
    assert "页面会一步一步提示当前进度，后续需要做什么也会清楚显示。" in page_text
    assert "听小顾提问" in page_text
    assert "像聊天一样回答" in page_text
    assert "查看后续提示" in page_text
    assert "慢慢来，不考试" in page_text
    assert "工作人员会在旁边帮您处理麦克风、画钟小游戏和后续简报。" not in page_text
    assert "点这里开始和小顾聊天" in page_text
    assert "st.session_state.elder_autostart_requested = True" in page_text
    assert "工作人员设置与辅助功能（展开）" in page_text
    assert "后台安全系统" in page_text
    assert "后台安全系统已启用" in page_text
    assert "后台已加锁" in page_text
    assert "管理员密码" in page_text
    assert "管理员临时访问口令" in page_text
    assert "解锁后台" in page_text
    assert "管理员已解锁" in page_text
    assert "老人端默认开放，管理员入口需要密码解锁" in page_text
    assert "is_staff_unlocked" in page_text
    assert "verify_staff_password" in page_text
    assert "登录账号" in page_text
    assert "可用登录账号" in page_text
    assert "新增账号由系统管理员维护，本页面不开放注册。" in page_text
    assert "登录演示账号" not in page_text
    assert "课程演示账号" not in page_text
    assert "工作人员辅助功能页" in page_text
    assert "真实 Qwen / Qwen-VL / ASR / TTS 配置" in page_text
    assert "配置缺失或 DEMO_MODE=true 时会安全回退到 mock/fallback" in page_text
    assert "隐藏侧边栏导航后，管理员入口集中在这里。" in page_text
    assert "#### 认知简报" in page_text
    assert "重点入口" not in page_text
    assert "其他管理员工具" in page_text
    assert "st.switch_page(\"pages/3_认知简报.py\")" in page_text
    assert "继续画钟拍照" in page_text
    assert "查看认知简报" in page_text
    assert "演示模式" in page_text
    assert "快捷访谈评估" in page_text
    assert "快捷访谈评估：预选/文字/语音回答，快速复刻老人端访谈到画钟/简报链路。" in page_text
    assert "对话评估：文本/示例/语音回答，生成认知风险提示。" not in page_text
    assert "画钟测试：上传或加载示例画钟，调用 Qwen-VL 或 fallback 分析结构化结果。" in page_text
    assert "认知简报：读取 SQLite 或 fixture，展示最近报告、趋势和家属提醒。" in page_text
    assert "和小顾聊天：老人端正式语音访谈流程，由首页大按钮进入。" in page_text
    assert "返回长者访谈入口" not in page_text
    assert "演示：选择认知水平，一键生成模拟流程、语音和评估报告。" in page_text
    assert "不保存原始用户音频" in page_text
    assert "API Key 只保存在本地 .env 或服务器 .env，不显示在页面中。" in page_text
    assert "当前运行模式" in page_text
    assert "LLM 模型" in page_text
    assert "VLM 模型" in page_text
    assert "真实 API 调用" not in page_text


def test_staff_gate_is_applied_to_staff_pages_but_not_elder_page() -> None:
    staff_gate_text = (PROJECT_ROOT / "core" / "staff_gate.py").read_text(
        encoding="utf-8"
    )
    assert '[data-testid="stSidebar"]' in staff_gate_text
    assert '[data-testid="stSidebarCollapsedControl"]' in staff_gate_text
    assert '[data-testid="stSidebarNav"]' in staff_gate_text
    assert 'st.page_link("app.py", label="返回主页面")' in staff_gate_text
    assert "管理员临时访问口令" in staff_gate_text

    staff_pages = [
        PROJECT_ROOT / "pages" / "1_对话评估.py",
        PROJECT_ROOT / "pages" / "2_画钟测试.py",
        PROJECT_ROOT / "pages" / "3_认知简报.py",
        PROJECT_ROOT / "pages" / "5_演示模式.py",
    ]

    for path in staff_pages:
        page_text = path.read_text(encoding="utf-8")
        assert "hide_sidebar_nav()" in page_text
        assert "render_staff_gate(config)" in page_text

    elder_page_text = (PROJECT_ROOT / "pages" / "4_长者简易版.py").read_text(
        encoding="utf-8"
    )
    assert "hide_sidebar_nav()" in elder_page_text
    assert "render_staff_gate" not in elder_page_text


def test_env_example_documents_demo_staff_password() -> None:
    env_example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")

    assert "STAFF_PASSWORD=8888" in env_example
    assert "Demo administrator password" in env_example


def test_streamlit_files_can_resolve_core_imports() -> None:
    streamlit_files = [
        PROJECT_ROOT / "app.py",
        PROJECT_ROOT / "pages" / "1_对话评估.py",
        PROJECT_ROOT / "pages" / "2_画钟测试.py",
        PROJECT_ROOT / "pages" / "3_认知简报.py",
        PROJECT_ROOT / "pages" / "4_长者简易版.py",
        PROJECT_ROOT / "pages" / "5_演示模式.py",
    ]

    for path in streamlit_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        core_imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module is not None
            and node.module.startswith("core")
        ]

        assert core_imports, f"{path.name} should import from core"

    import core.mock_data  # noqa: F401
    import core.report  # noqa: F401
    import core.schemas  # noqa: F401


def test_family_brief_page_uses_single_cogniguard_score_trend() -> None:
    page_text = (PROJECT_ROOT / "pages" / "3_认知简报.py").read_text(encoding="utf-8")
    ui_text = (PROJECT_ROOT / "core" / "ui.py").read_text(encoding="utf-8")

    assert "sort_sessions_chronologically" in page_text
    assert "latest = sessions[-1]" in page_text
    assert '{"dialogue", "clock"} <= set(session.get("components", []))' not in page_text
    assert "### CogniGuard 综合提示分趋势" in page_text
    assert "记录不足，暂不生成趋势。" in page_text
    assert "纵轴为 0-100 分" in page_text
    assert "横轴按测试时间从旧到新排序，最右侧高亮点为最近一次" in page_text
    assert "CogniGuard 综合提示分数值" in page_text
    assert "st.vega_lite_chart" in page_text
    assert '"sort": x_sort_order' in page_text
    assert '"scale": {"domain": [0, 100]}' in page_text
    assert "综合提示分变化" in page_text
    assert 'st.container(border=True, key="brief_trend_card")' in page_text
    assert "cg-trend-stat-grid" in page_text
    assert "_trend_delta_text" in page_text
    assert "Optional[float]" in page_text
    assert "float | None" not in page_text
    assert "is_latest" in page_text
    assert '"type": "area"' in page_text
    assert '"interpolate": "monotone"' in page_text
    assert '"labelAngle": 0' in page_text
    assert ".st-key-brief_trend_card" in ui_text
    assert ".cg-trend-card" in ui_text
    assert ".cg-trend-stat-grid" in ui_text
    assert "st.line_chart" not in page_text
    assert "chart_columns = [label for label in DOMAIN_LABELS.values()" not in page_text
    assert "EVIDENCE_DOMAIN_TONES" in page_text
    assert 'tone=_evidence_tone_for_domain(domain, "green")' in page_text
    assert 'tone=_evidence_tone_for_domain(domain, "blue")' in page_text
    for tone in [
        ".cg-evidence-orientation",
        ".cg-evidence-memory",
        ".cg-evidence-language",
        ".cg-evidence-executive",
        ".cg-evidence-attention",
        ".cg-evidence-visuospatial",
    ]:
        assert tone in ui_text


def test_streamlit_pages_use_client_layer_for_model_tasks() -> None:
    expected_imports = {
        PROJECT_ROOT / "pages" / "1_对话评估.py": {
            ("core.llm_client", "evaluate_dialogue"),
            ("core.tts_client", "synthesize_speech"),
        },
        PROJECT_ROOT / "pages" / "2_画钟测试.py": {
            ("core.vlm_client", "analyze_clock_image")
        },
        PROJECT_ROOT / "pages" / "3_认知简报.py": {
            ("core.llm_client", "generate_trend_report"),
            ("core.llm_client", "generate_family_report"),
        },
        PROJECT_ROOT / "pages" / "4_长者简易版.py": {
            ("core.llm_client", "evaluate_dialogue"),
            ("core.llm_client", "generate_next_question"),
            ("core.asr_client", "transcribe_audio"),
            ("core.tts_client", "synthesize_speech"),
        },
        PROJECT_ROOT / "pages" / "5_演示模式.py": {
            ("core.llm_client", "evaluate_dialogue"),
            ("core.mock_data", "build_classroom_demo_interview"),
            ("core.tts_client", "synthesize_speech"),
        },
    }

    for path, expected in expected_imports.items():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        actual = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                for alias in node.names:
                    actual.add((node.module, alias.name))

        assert expected <= actual


def test_dialog_page_no_longer_exposes_full_sample_loading_buttons() -> None:
    page_text = (PROJECT_ROOT / "pages" / "1_对话评估.py").read_text(encoding="utf-8")

    assert 'st.set_page_config(page_title="快捷访谈评估", layout="wide")' in page_text
    assert 'st.title("快捷访谈评估")' in page_text
    assert "工作人员可在这里用预选回答、文字或语音快速跑通“小顾”访谈" in page_text
    assert "config=config" in page_text
    assert "### 小顾 · AI 访谈问题" in page_text
    assert "### 快捷预选回答" in page_text
    assert "### 文字快速测试" in page_text
    assert "画钟建议记录" in page_text
    assert "我们再做一个小小游戏" in page_text
    assert "请您在纸上画一个钟，指到 11 点 10 分" in page_text
    assert 'st.info(f"小顾：{CLOCK_TRANSITION_MESSAGE}")' in page_text
    assert "已记录到可能需要补充画钟" in page_text
    assert "本页不会在访谈中途跳转" in page_text
    assert "EARLY_CLOCK_TRIGGER_MIN_VAGUE_TURNS" in page_text
    assert "生成评估并判断下一步" in page_text
    assert "提交文字回答并生成下一问" in page_text
    assert "不需要生成 TTS 或使用 ASR" in page_text
    assert "开发测试状态" in page_text
    assert "开发测试详情" in page_text
    assert "Qwen sample_answers 或本地规则校验后的结果" in page_text
    assert "下一问由小顾根据当前对话生成" in page_text
    assert "加载正常样本对话" not in page_text
    assert "加载轻度下降样本对话" not in page_text
    assert "加载波动型样本对话" not in page_text
    assert "load_dialog_sample" not in page_text
    assert "正常回答" in page_text
    assert "继续画钟拍照" in page_text
    assert "pages/2_画钟测试.py" in page_text
    assert "pages/3_认知简报.py" in page_text
    assert "下一问来源" in page_text
    assert "对话评估参考分" in page_text
    assert "placeholder=f\"例如：{example_answers['normal']}\"" in page_text
    assert "例如：今天应该是周六，我早上吃了粥和鸡蛋。" not in page_text


def test_dialog_page_exposes_manual_audio_asr_without_replacing_text_flow() -> None:
    page_text = (PROJECT_ROOT / "pages" / "1_对话评估.py").read_text(encoding="utf-8")

    assert "from core.asr_client import transcribe_audio" in page_text
    assert "st.audio_input" in page_text
    assert "st.file_uploader" in page_text
    assert "转写录音" in page_text
    assert "转写上传音频" in page_text
    assert "填入回答框" in page_text
    assert "提交为当前回答" in page_text
    assert "不保存原始音频" in page_text
    assert "st.form(\"dialog_text_quick_test_form\")" in page_text
    assert "st.form_submit_button(\"提交文字回答并生成下一问\", on_click=_submit_manual_answer)" in page_text


def test_dialog_page_exposes_optional_tts_for_current_question() -> None:
    page_text = (PROJECT_ROOT / "pages" / "1_对话评估.py").read_text(encoding="utf-8")

    assert "synthesize_speech" in page_text
    assert "prepare_text_for_tts" in page_text
    assert "系统语音播放（可选）" in page_text
    assert "SYSTEM_VOICE_MODE_HIGH_QUALITY = \"高质量 TTS\"" in page_text
    assert "SYSTEM_VOICE_MODE_BROWSER_FAST = \"快速朗读（浏览器）\"" in page_text
    assert "DEFAULT_SYSTEM_VOICE_MODE_INDEX = 0" in page_text
    assert "st.radio(" in page_text
    assert "系统语音模式" in page_text
    assert "生成并播放系统语音" in page_text
    assert "正在生成系统语音，首次生成可能需要 10-20 秒，请稍候" in page_text
    assert "已使用缓存音频，可直接播放" in page_text
    assert "系统语音已生成。当前采用非实时高质量 TTS" in page_text
    assert "系统高质量语音：Qwen-TTS 合成" in page_text
    assert "系统快速朗读：浏览器本地语音" in page_text
    assert "用户语音回答：麦克风录音或上传音频" in page_text
    assert "本阶段使用非实时 Qwen-TTS，适合高质量语音演示" in page_text
    assert "实时流式 TTS 可作为后续优化" in page_text
    assert "快速朗读使用浏览器本地语音" in page_text
    assert "不调用 Qwen-TTS，不消耗 TTS API" in page_text
    assert "不读取麦克风，也不保存音频" in page_text
    assert "components.html" in page_text
    assert "speechSynthesis" in page_text
    assert "SpeechSynthesisUtterance" in page_text
    assert "zh-CN" in page_text
    assert "当前浏览器不支持快速朗读" in page_text
    assert "st.audio(tts_audio, format=tts_mime_type)" in page_text
    assert "当前未生成真实音频，可继续使用文本问题和语音回答流程" in page_text


def test_elder_simple_page_exposes_guided_voice_interview_flow() -> None:
    page_path = PROJECT_ROOT / "pages" / "4_长者简易版.py"
    component_module_path = PROJECT_ROOT / "core" / "elder_voice_component.py"
    component_path = PROJECT_ROOT / "components" / "elder_voice_recorder" / "index.html"
    page_text = page_path.read_text(encoding="utf-8")
    component_module_text = component_module_path.read_text(encoding="utf-8")
    component_text = component_path.read_text(encoding="utf-8")

    assert page_path.exists()
    assert component_module_path.exists()
    assert component_path.exists()
    assert 'st.set_page_config(page_title="和小顾聊天", layout="wide")' in page_text
    assert "和小顾聊天" in page_text
    assert "build_history_personalized_start" in page_text
    assert "_initial_context()" in page_text
    assert "上次记录已经在系统里" not in page_text
    assert 'st.session_state.get("elder_autostart_requested")' in page_text
    assert "st.session_state.elder_autostart_requested = False" in page_text
    assert "我准备好了，开始聊天" in page_text
    assert "听问题，说回答" in page_text
    assert "只保存文字，不保存录音" in page_text
    assert "请听问题" in page_text
    assert "听完以后，直接说出您的回答" in page_text
    assert "听到停顿后会自动停止录音" in page_text
    assert "elder-live-question-panel" in page_text
    assert "elder-voice-station" in page_text
    assert "elder-transcript-card" in page_text
    assert "上一题识别文字" in page_text
    assert "语音感应" in page_text
    assert "录音状态" in page_text
    assert "elder_voice_recorder" in page_text
    assert "components.declare_component" not in page_text
    assert "components.declare_component" in component_module_text
    assert "_process_auto_recorder_result(recorder_result)" in page_text
    assert "正在准备问题，请稍等" in page_text
    assert "正在识别，请稍等" in page_text
    assert "今天的访谈完成了，谢谢您" in page_text
    assert "结果已经整理好，请交给家人查看" in page_text
    assert "FLOW_STEP_CLOCK_TEST" in page_text
    assert "FLOW_STEP_FINISH" in page_text
    assert "import streamlit.components.v1 as components" in page_text
    assert "def _render_transition_audio_player" in page_text
    assert "transitionAudio.play()" in page_text
    assert "浏览器阻止了自动播放" in page_text
    assert "components.html(player_html, height=142)" in page_text
    assert "我们再做一个小小游戏" in page_text
    assert "请您在纸上画一个钟，指到 11 点 10 分" in page_text
    assert "接下来请交给工作人员，继续画钟小游戏" not in page_text
    assert "本次访谈可以先结束" in page_text
    assert "继续画钟测试" in page_text
    assert "查看认知简报" in page_text
    assert "pages/2_画钟测试.py" in page_text
    assert "pages/3_认知简报.py" in page_text
    assert 'st.page_link("pages/3_认知简报.py", label="稍后查看认知简报")' in page_text
    assert "st.session_state.current_assessment_id = assessment_id.strip()" in page_text
    assert "elder_last_audio_signature" in page_text
    assert "elder_last_processed_recording_key" in page_text
    assert "elder_recorder_auto_start" in page_text
    assert "elder_last_transcript" in page_text
    assert "recording_key != _current_recording_key()" in page_text
    assert "auto_start=auto_start" in page_text
    assert "_generate_and_save_report()" in page_text
    assert "MediaRecorder" in component_text
    assert "getUserMedia" in component_text
    assert "输入音量" in component_text
    assert "meterText" in component_text
    assert "statePill" in component_text
    assert "args.auto_start !== false" in component_text
    assert "waitForManualStart" in component_text
    assert 'body[data-state="listening"]' in component_text
    assert "正在听您回答" in component_text
    assert "说完后停顿一下，系统会自动停止" in component_text
    assert "我说完了" in component_text
    assert "Streamlit.setComponentValue" in component_text
    assert "st.audio_input" not in page_text
    assert "st.file_uploader" not in page_text
    assert "上传音频回答" not in page_text
    assert "st.text_area" not in page_text
    assert "SYSTEM_VOICE_MODE_BROWSER_FAST" not in page_text
    assert "确认并提交回答" not in page_text
    assert "transcribe_audio" in page_text
    assert "synthesize_speech" in page_text
    assert "evaluate_dialogue" in page_text


def test_classroom_demo_page_exposes_simulated_voice_demo_flow() -> None:
    page_path = PROJECT_ROOT / "pages" / "5_演示模式.py"
    ui_path = PROJECT_ROOT / "core" / "ui.py"
    page_text = page_path.read_text(encoding="utf-8")
    ui_text = ui_path.read_text(encoding="utf-8")

    assert page_path.exists()
    assert "演示模式" in page_text
    assert "展示链路" in page_text
    assert "访谈结束后进入" in page_text
    assert "进入画钟拍照" in page_text
    assert "_classroom_speech_html(\"小顾\", clock_prompt)" in page_text
    assert "cg-classroom-speech cg-classroom-speech-" in page_text
    assert "cg-classroom-speaker" not in page_text
    assert "cg-classroom-speaker" not in ui_text
    assert ".cg-classroom-speech-assistant" in ui_text
    assert ".cg-classroom-speech-elder" in ui_text
    assert "st.info(f\"小顾：{clock_prompt}\")" not in page_text
    assert "小顾提示" not in page_text
    assert "需要补充画钟拍照" not in page_text
    assert "展示页只表明需要拍照" in page_text
    assert "正常表现" in page_text
    assert "轻度下降" in page_text
    assert "明显异常" in page_text
    assert "模拟演示数据" in page_text
    assert "模拟演示数据，不是真实老人输入" in page_text
    assert "小顾声音" in page_text
    assert "老人声音" in page_text
    assert "TTS_MODEL_ASSISTANT" in page_text
    assert "TTS_MODEL_PATIENT_DEMO" in page_text
    assert "TTS_VOICE_ASSISTANT" in page_text
    assert "TTS_VOICE_PATIENT_DEMO" in page_text
    assert "生成完整演示流程" in page_text
    assert "生成全部演示语音" in page_text
    assert "正在并行生成演示语音" in page_text
    assert "重试失败语音" in page_text
    assert "DEMO_TTS_MAX_ATTEMPTS = 2" in page_text
    assert "DEMO_TTS_RETRY_DELAY_SECONDS = 0.75" in page_text
    assert "DEMO_TTS_MAX_WORKERS = 7" in page_text
    assert "ThreadPoolExecutor" in page_text
    assert "max_workers=max_workers" in page_text
    assert "播放完整演示" in page_text
    assert "暂停播放" in page_text
    assert "继续播放" in page_text
    assert "停止播放" in page_text
    assert "播放过程中可随时暂停、继续或停止" in page_text
    assert "答辩时可以只播放前几轮" not in page_text
    assert "可以继续讲解" not in page_text
    assert "已停止播放。如需重新播放，请点击播放完整演示。" in page_text
    assert "currentAudio.pause()" in page_text
    assert "currentAudio.currentTime = 0" in page_text
    assert "运行评估并查看报告" in page_text
    assert "### 完整评估报告" in page_text
    assert page_text.index("### 完整评估报告") < page_text.index('st.button(\n    "运行评估并查看报告"')
    assert "control_columns = st.columns(2)" in page_text
    assert "control_columns = st.columns(3)" not in page_text
    assert "正在预生成完整报告缓存" in page_text
    assert "完整报告缓存已准备好" in page_text
    assert "报告已提前缓存，包含对话评估和画钟分析" in page_text
    assert "报告已提前缓存。点击上方按钮即可展开对话 + 画钟综合报告" in page_text
    assert "讲到这里" not in page_text
    assert "讲完对话时间线" not in page_text
    assert "对话 + 画钟综合报告" in page_text
    assert "build_classroom_clock_report" in page_text
    assert "build_clock_assessment_record" in page_text
    assert "compute_cogniguard_score" in page_text
    assert "compute_clock_structure_score" in page_text
    assert "display_cdt_feature_value" in page_text
    assert "@st.cache_data" in page_text
    assert "首次生成可能需要较长时间，请稍候" in page_text
    assert "正在并行生成第 {job_range_label} 段语音" in page_text
    assert "刚完成：第 {segment['segment_index']} / {len(segments)} 段" in page_text
    assert "已完成 {completed_jobs}/{len(jobs)} 段" in page_text
    assert "固定音频或缓存命中会快速跳过，真实生成可能较慢" in page_text
    assert "CLASSROOM_DEMO_AUDIO_DIR" in page_text
    assert "source\": \"static_audio\"" in page_text
    assert "项目固定音频" in page_text
    assert "def _segment_range_label" in page_text
    assert "完整演示播放区" in page_text
    assert "浏览器限制自动播放" in page_text
    assert "assistant_model = (" in page_text
    assert "patient_model = config.tts_model_patient_demo.strip()" in page_text
    assert "patient_voice = config.tts_voice_patient_demo.strip()" in page_text
    assert "model=model or None" in page_text
    assert "模型 {patient_model}，音色 {patient_voice}" in page_text
    assert "TTS_MODEL_PATIENT_DEMO = {patient_model}" in page_text
    assert "单条诊断可用但批量失败时，通常是接口限流、超时或并发过高" in page_text
    assert "_is_audio_success(existing)" in page_text
    assert "retry_failed_only=True" in page_text
    assert "if retry_failed_only and not _is_audio_failure(existing)" in page_text
    assert "st.session_state.classroom_tts_results[segment[\"result_key\"]] = result" in page_text
    assert "cached：{cached}" in page_text
    assert "固定音频优先，最多 7 路并行" in page_text
    assert "建议为老人声音配置不同音色，便于演示区分" in page_text
    assert "小顾 → 老人" in page_text
    assert "系统问：" not in page_text
    assert "模拟老人答：" not in page_text
    assert "生成第1轮系统声音" not in page_text
    assert "生成第1轮模拟老人声音" not in page_text
    assert "保存为张奶奶的一次综合演示评估" in page_text
    assert "不会伪装成真实 ASR 结果" in page_text
    assert "老人音频不会再走 ASR" in page_text
    assert "classroom_clock_normal.png" in page_text
    assert "classroom_clock_mild_decline.png" in page_text
    assert "classroom_clock_obvious_issue.png" in page_text
    assert "演示示意图，仅展示本次画钟结果" in page_text
    assert "这里展示一张正常画钟示意图" in page_text
    assert "画钟结构分" in page_text
    assert "CogniGuard 综合提示分" in page_text
    assert "报告组成" in page_text
    assert "画钟分析" in page_text
    assert "目标时间" in page_text
    assert "对话和画钟合并后的非诊断提示" in page_text
    assert "本评估基于模拟回答和演示画钟示意结果生成" in page_text
    assert "工作人员在现场协助" not in page_text
    assert "工作人员手动补充时的画钟照片" not in page_text
    assert "components.html" in page_text

    clock_dir = PROJECT_ROOT / "assets" / "classroom_clock_samples"
    for filename in [
        "classroom_clock_normal.png",
        "classroom_clock_mild_decline.png",
        "classroom_clock_obvious_issue.png",
    ]:
        image_path = clock_dir / filename
        assert image_path.exists()
        assert image_path.stat().st_size > 0


def _normal_cdt_features() -> dict:
    return {
        "numbers_complete": True,
        "number_order_correct": True,
        "number_spacing": "normal",
        "number_distribution": "balanced",
        "hands_present": True,
        "target_time_match": True,
        "center_anchor_clear": True,
    }
