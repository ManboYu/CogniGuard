from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from core import asr_client, llm_client, tts_client, vlm_client
from core.asr_client import transcribe_audio
from core.config import (
    AppConfig,
    build_runtime_status,
    is_asr_config_complete,
    is_llm_config_complete,
    is_tts_config_complete,
    is_vlm_config_complete,
    load_config,
)
from core.embedding import embed_text
from core.llm_client import (
    evaluate_dialogue,
    generate_family_report,
    generate_next_question,
    generate_trend_report,
)
from core.mock_data import load_fixture_sessions
from core.schemas import COGNITIVE_DOMAINS, DISCLAIMER, RISK_LEVELS, fallback_result
from core.tts_client import synthesize_speech
from core.vlm_client import analyze_clock_image
from scripts import (
    check_asr_connection,
    check_qwen_connection,
    check_qwen_vl_connection,
    check_tts_connection,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_config_defaults_to_demo_mode_without_required_env(monkeypatch) -> None:
    for key in [
        "QWEN_BASE_URL",
        "QWEN_API_KEY",
        "LLM_API_KEY",
        "VLM_API_KEY",
        "EMBED_API_KEY",
        "ASR_API_KEY",
        "TTS_API_KEY",
        "TTS_MODEL_ASSISTANT",
        "TTS_MODEL_PATIENT_DEMO",
        "TTS_VOICE_PATIENT_DEMO",
        "DEMO_MODE",
        "VOICE_DEMO_MODE",
        "USE_CACHE",
        "STAFF_PASSWORD",
    ]:
        monkeypatch.setenv(key, "")

    config = load_config()

    assert config.demo_mode is True
    assert config.use_cache is True
    assert config.staff_password == "8888"


def test_config_reads_staff_password_from_env(monkeypatch) -> None:
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("STAFF_PASSWORD", "2468")

    config = load_config()

    assert config.staff_password == "2468"


def test_config_qwen_unified_fallbacks_make_qwen_models_complete(monkeypatch) -> None:
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("QWEN_API_KEY", "sk-qwen-shared")
    monkeypatch.setenv("LLM_MODEL", "qwen-max")
    monkeypatch.setenv("VLM_MODEL", "qwen-vl-max")
    monkeypatch.setenv("ASR_MODEL", "qwen3-asr-flash")
    monkeypatch.setenv("TTS_MODEL", "qwen-tts")
    monkeypatch.setenv("TTS_VOICE_PATIENT_DEMO", "DemoPatient")
    monkeypatch.setenv("DEMO_MODE", "false")

    config = load_config()
    rendered = json.dumps(build_runtime_status(config), ensure_ascii=False)

    assert config.llm_base_url == config.qwen_base_url
    assert config.vlm_base_url == config.qwen_base_url
    assert config.asr_base_url == config.qwen_base_url
    assert config.tts_base_url == config.qwen_base_url
    assert config.llm_api_key == config.qwen_api_key
    assert config.vlm_api_key == config.qwen_api_key
    assert config.asr_api_key == config.qwen_api_key
    assert config.tts_api_key == config.qwen_api_key
    assert config.tts_model_assistant == "qwen-tts"
    assert config.tts_model_patient_demo == "cosyvoice-v3-flash"
    assert config.tts_voice_patient_demo == "DemoPatient"
    assert is_llm_config_complete(config) is True
    assert is_vlm_config_complete(config) is True
    assert is_asr_config_complete(config) is True
    assert is_tts_config_complete(config) is True
    assert "sk-qwen-shared" not in rendered


def test_config_dedicated_api_keys_override_qwen_key(monkeypatch) -> None:
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
    monkeypatch.setenv("QWEN_API_KEY", "sk-qwen-shared")
    monkeypatch.setenv("LLM_API_KEY", "sk-dedicated-llm")
    monkeypatch.setenv("ASR_API_KEY", "sk-dedicated-asr")
    monkeypatch.setenv("TTS_API_KEY", "sk-dedicated-tts")
    monkeypatch.setenv("LLM_MODEL", "qwen-max")
    monkeypatch.setenv("VLM_MODEL", "qwen-vl-max")
    monkeypatch.setenv("ASR_MODEL", "qwen3-asr-flash")
    monkeypatch.setenv("TTS_MODEL", "qwen-tts")
    monkeypatch.setenv("DEMO_MODE", "false")

    config = load_config()

    assert config.llm_base_url == config.qwen_base_url
    assert config.vlm_base_url == config.qwen_base_url
    assert config.asr_base_url == config.qwen_base_url
    assert config.llm_api_key == "sk-dedicated-llm"
    assert config.vlm_api_key == "sk-qwen-shared"
    assert config.asr_api_key == "sk-dedicated-asr"
    assert config.tts_api_key == "sk-dedicated-tts"
    assert is_llm_config_complete(config) is True
    assert is_vlm_config_complete(config) is True
    assert is_asr_config_complete(config) is True
    assert is_tts_config_complete(config) is True


def test_config_patient_demo_voice_defaults_to_distinct_demo_voice(monkeypatch) -> None:
    _clear_model_env(monkeypatch)

    config = load_config()

    assert config.tts_model_assistant == "qwen-tts"
    assert config.tts_model_patient_demo == "cosyvoice-v3-flash"
    assert config.tts_voice_assistant == "Cherry"
    assert config.tts_voice_patient_demo == "longlaoyi_v3"
    assert config.tts_voice_patient_demo != config.tts_voice_assistant


def test_config_patient_demo_voice_uses_env_value(monkeypatch) -> None:
    _clear_model_env(monkeypatch)
    monkeypatch.setenv("TTS_MODEL_PATIENT_DEMO", "custom-cosyvoice")
    monkeypatch.setenv("TTS_VOICE_PATIENT_DEMO", "longlaoyi_v3")

    config = load_config()

    assert config.tts_model_patient_demo == "custom-cosyvoice"
    assert config.tts_voice_patient_demo == "longlaoyi_v3"
    assert config.tts_voice_patient_demo != config.tts_voice_assistant


def test_runtime_status_does_not_expose_api_keys() -> None:
    config = AppConfig(
        llm_base_url="https://example.test/v1",
        llm_api_key="sk-secret-llm",
        llm_model="qwen-max",
        vlm_base_url="https://example.test/v1",
        vlm_api_key="sk-secret-vlm",
        vlm_model="qwen-vl-max",
        demo_mode=False,
    )

    status = build_runtime_status(config)
    rendered = json.dumps(status, ensure_ascii=False)

    assert status == {
        "运行模式": "真实 API",
        "LLM 模型": "qwen-max",
        "VLM 模型": "qwen-vl-max",
    }
    assert "sk-secret" not in rendered


def test_llm_client_returns_mock_dialogue_report_without_api_key() -> None:
    config = AppConfig()
    report = evaluate_dialogue(["今天周六。", "早饭吃了粥。"], config=config)

    assert report["is_mock"] is True
    assert report["metadata"]["source"] == "mock"
    assert set(report["domain_scores"]) == set(COGNITIVE_DOMAINS)
    assert report["risk_level"] in RISK_LEVELS
    assert report["disclaimer"] == DISCLAIMER


def test_llm_client_demo_mode_does_not_create_real_client(monkeypatch) -> None:
    def fail_if_called(_config):
        raise AssertionError("real LLM client should not be created in demo mode")

    monkeypatch.setattr(llm_client, "_create_openai_client", fail_if_called)

    config = AppConfig(
        llm_base_url="https://example.test/v1",
        llm_api_key="test-key",
        llm_model="qwen-plus",
        demo_mode=True,
    )
    report = llm_client.evaluate_dialogue(["今天周六。"], config=config)

    assert report["is_mock"] is True
    assert report["metadata"]["source"] == "mock"
    assert report["metadata"]["model"] == "qwen-plus"
    assert report["metadata"]["reason"] == "DEMO_MODE=true"
    assert report["disclaimer"] == DISCLAIMER


def test_llm_client_missing_config_does_not_create_real_client(monkeypatch) -> None:
    def fail_if_called(_config):
        raise AssertionError("real LLM client should not be created without config")

    monkeypatch.setattr(llm_client, "_create_openai_client", fail_if_called)

    config = AppConfig(
        llm_base_url="https://example.test/v1",
        llm_api_key="",
        llm_model="qwen-plus",
        demo_mode=False,
    )
    report = llm_client.evaluate_dialogue(["今天周六。"], config=config)

    assert report["is_mock"] is True
    assert report["metadata"]["source"] == "mock"
    assert report["metadata"]["reason"] == "LLM 配置不完整"
    assert report["disclaimer"] == DISCLAIMER


def test_llm_client_real_dialogue_uses_qwen_compatible_chat_settings(
    monkeypatch,
) -> None:
    fake_client = _FakeChatClient(
        json.dumps(_valid_dialogue_payload(), ensure_ascii=False)
    )
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    config = _real_llm_config()
    report = llm_client.evaluate_dialogue(["今天周六。", "早饭吃了粥。"], config=config)
    call = fake_client.calls[0]

    assert report["is_mock"] is False
    assert report["metadata"]["source"] == "qwen"
    assert report["metadata"]["model"] == "qwen-plus"
    assert report["risk_level"] == "low"
    assert call["model"] == "qwen-plus"
    assert call["temperature"] == llm_client.LLM_TEMPERATURE
    assert call["max_tokens"] == llm_client.LLM_MAX_TOKENS
    assert call["response_format"] == {"type": "json_object"}
    assert _messages_contain_json(call["messages"])
    assert "早饭吃了粥" in call["messages"][1]["content"]


def test_llm_client_default_prompt_contains_json_when_prompt_missing(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(llm_client, "PROMPT_DIR", tmp_path)

    prompt = llm_client._read_prompt("dialog_eval.md")

    assert "JSON" in prompt


def test_qwen_connection_check_messages_contain_json() -> None:
    messages = check_qwen_connection._build_check_messages()

    assert _messages_contain_json(messages)


def test_llm_client_parses_json_code_fence(monkeypatch) -> None:
    content = "```json\n" + json.dumps(
        _valid_dialogue_payload(evidence=["能复述刚才提到的早餐内容。"]),
        ensure_ascii=False,
    ) + "\n```"
    fake_client = _FakeChatClient(content)
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    report = llm_client.evaluate_dialogue(["今天周六。"], config=_real_llm_config())

    assert report["metadata"]["source"] == "qwen"
    assert report["evidence"][0]["source"] == "dialog"
    assert report["evidence"][0]["text"] == "能复述刚才提到的早餐内容。"


def test_llm_client_normalizes_chinese_risk_level(monkeypatch) -> None:
    payload = _valid_dialogue_payload(risk_level="中风险")
    fake_client = _FakeChatClient(json.dumps(payload, ensure_ascii=False))
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    report = llm_client.evaluate_dialogue(["今天周六。"], config=_real_llm_config())

    assert report["metadata"]["source"] == "qwen"
    assert report["risk_level"] == "medium"


def test_llm_client_fills_missing_domain_scores_with_null(monkeypatch) -> None:
    payload = _valid_dialogue_payload()
    payload["domain_scores"] = {
        "orientation": 0.8,
        "memory": 0.7,
    }
    fake_client = _FakeChatClient(json.dumps(payload, ensure_ascii=False))
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    report = llm_client.evaluate_dialogue(["今天周六。"], config=_real_llm_config())

    assert report["metadata"]["source"] == "qwen"
    assert report["domain_scores"]["orientation"] == 0.8
    assert report["domain_scores"]["memory"] == 0.7
    assert report["domain_scores"]["language"] is None
    assert report["domain_scores"]["executive_function"] is None
    assert report["domain_scores"]["attention"] is None
    assert report["domain_scores"]["visuospatial"] is None


def test_llm_client_bad_json_returns_fallback(monkeypatch) -> None:
    fake_client = _FakeChatClient("not json")
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    report = llm_client.evaluate_dialogue(["今天周六。"], config=_real_llm_config())

    _assert_fallback_report(report, "json_error: 模型返回内容不是有效 JSON")


def test_llm_client_invalid_schema_returns_fallback(monkeypatch) -> None:
    invalid_payload = _valid_dialogue_payload()
    invalid_payload["explanation"] = ""
    fake_client = _FakeChatClient(json.dumps(invalid_payload, ensure_ascii=False))
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    report = llm_client.evaluate_dialogue(["今天周六。"], config=_real_llm_config())

    _assert_fallback_report(report, "schema_error: 模型返回 JSON 未通过 schema 校验")


def test_llm_client_api_exception_returns_fallback(monkeypatch) -> None:
    fake_client = _FakeChatClient(error=RuntimeError("simulated API failure"))
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    report = llm_client.evaluate_dialogue(["今天周六。"], config=_real_llm_config())

    _assert_fallback_report(report, "api_error: LLM 调用失败，已回退到安全结果")


def test_llm_client_returns_mock_trend_and_family_report_without_api_key() -> None:
    config = AppConfig()
    sessions = load_fixture_sessions("normal")

    trend = generate_trend_report(sessions, config=config)
    family = generate_family_report(sessions, config=config)
    question = generate_next_question(["今天早上吃了粥。"], config=config)

    assert trend["trend_label"] == "稳定"
    assert trend["disclaimer"] == DISCLAIMER
    assert family["is_mock"] is True
    assert family["family_reminders"]
    assert family["disclaimer"] == DISCLAIMER
    assert question["is_mock"] is True
    assert question["question"]


def test_generate_next_question_mock_fallback_keeps_domain_coverage_logic() -> None:
    config = AppConfig(demo_mode=True)

    first = generate_next_question([], config=config, covered_domains=[])
    second = generate_next_question(
        ["AI访谈问题：您好，我是小顾，今天我陪您轻松聊一会儿。今天大概是星期几？", "老人回答：周三。"],
        config=config,
        covered_domains=[first["target_domain"]],
    )

    assert first["metadata"]["source"] == "mock"
    assert first["question"]
    assert first["target_domain"] == "orientation"
    assert "小顾" in first["question"]
    assert second["metadata"]["source"] == "mock"
    assert second["question"]
    assert second["target_domain"] == "memory"
    assert second["target_domain"] != first["target_domain"]


def test_generate_next_question_uses_real_llm_when_configured(monkeypatch) -> None:
    payload = {
        "target_domain": "attention",
        "question": "您说周六，说得很清楚。请从 20 往回数三个数。",
        "reason": "观察注意力线索",
        "is_mock": False,
    }
    fake_client = _FakeChatClient(json.dumps(payload, ensure_ascii=False))
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    result = generate_next_question(
        ["AI访谈问题：今天大概是星期几？", "老人回答：周六。"],
        config=_real_llm_config(),
        covered_domains=["orientation", "memory"],
    )
    call = fake_client.calls[0]

    assert result["metadata"]["source"] == "qwen"
    assert result["target_domain"] == "attention"
    assert "您" in result["question"]
    assert "你" not in result["question"]
    assert set(result["sample_answers"]) == {"normal", "mild_decline", "vague"}
    assert result["is_mock"] is False
    assert call["response_format"] == {"type": "json_object"}
    assert _messages_contain_json(call["messages"])
    assert "orientation, memory" in call["messages"][1]["content"]
    assert "老人上一轮回答: 周六。" in call["messages"][1]["content"]
    assert "小顾" in call["messages"][1]["content"]
    assert "温柔的年轻晚辈" in call["messages"][1]["content"]
    assert "尽量不超过 70 个中文字符" in call["messages"][1]["content"]
    assert "从家的左边还是右边去公园" in call["messages"][1]["content"]
    assert "sample_answers" in call["messages"][0]["content"]


def test_generate_next_question_corrects_qwen_digit_sample_answers(monkeypatch) -> None:
    payload = {
        "target_domain": "attention",
        "question": "请您把这串数字倒着说一遍：7-2-5。",
        "reason": "观察注意力和工作记忆线索",
        "sample_answers": {
            "normal": "7-2-5。",
            "mild_decline": "我记得是 7-5-2。",
            "vague": "我不知道。",
        },
        "source": "qwen",
    }
    fake_client = _FakeChatClient(json.dumps(payload, ensure_ascii=False))
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    result = generate_next_question(
        ["AI访谈问题：今天大概是星期几？", "老人回答：周六。"],
        config=_real_llm_config(),
        covered_domains=["orientation"],
    )

    assert result["metadata"]["source"] == "qwen"
    assert result["sample_answers"]["normal"] == "5-2-7。"
    assert result["sample_answers"]["mild_decline"] == "5-7-2，我有点记混了。"


def test_generate_next_question_returns_completion_when_domains_covered(monkeypatch) -> None:
    def fail_if_called(_config):
        raise AssertionError("real LLM client should not be created after coverage is complete")

    monkeypatch.setattr(llm_client, "_create_openai_client", fail_if_called)

    result = generate_next_question(
        ["已有完整访谈。"],
        config=_real_llm_config(),
        covered_domains=list(COGNITIVE_DOMAINS),
    )

    assert result["completed"] is True
    assert result["metadata"]["source"] == "mock"
    assert "主要认知域已覆盖" in result["question"]


def test_generate_next_question_rejects_non_self_contained_real_question(monkeypatch) -> None:
    payload = {
        "target_domain": "attention",
        "question": "我们来玩数字游戏，我念一串数字，您再倒着说。",
        "reason": "观察注意力",
        "is_mock": False,
    }
    fake_client = _FakeChatClient(json.dumps(payload, ensure_ascii=False))
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    result = generate_next_question(
        ["AI访谈问题：今天大概是星期几？", "老人回答：周六。"],
        config=_real_llm_config(),
        covered_domains=["orientation"],
    )

    assert result["metadata"]["source"] == "fallback"
    assert "自包含" in result["metadata"]["reason"]


def test_generate_next_question_rejects_abstract_geometry_question(monkeypatch) -> None:
    payload = {
        "target_domain": "visuospatial",
        "question": "请您想象把一个正方形分成四个小正方形，应该怎么分？",
        "reason": "观察视觉空间能力",
        "source": "qwen",
    }
    fake_client = _FakeChatClient(json.dumps(payload, ensure_ascii=False))
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    result = generate_next_question(
        ["AI访谈问题：刚才早饭吃了什么？", "老人回答：我吃了粥。"],
        config=_real_llm_config(),
        covered_domains=["memory"],
    )

    assert result["metadata"]["source"] == "fallback"
    assert "自包含" in result["metadata"]["reason"]


def test_generate_next_question_rejects_stale_fixed_question_material(monkeypatch) -> None:
    payload = {
        "target_domain": "memory",
        "question": "您说得很清楚。请记住这三个词：苹果、钥匙、报纸。",
        "reason": "观察短时记忆",
        "source": "qwen",
    }
    fake_client = _FakeChatClient(json.dumps(payload, ensure_ascii=False))
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    result = generate_next_question(
        ["AI访谈问题：今天大概是星期几？", "老人回答：周六。"],
        config=_real_llm_config(),
        covered_domains=["orientation"],
    )

    assert result["metadata"]["source"] == "fallback"
    assert "过时固定题材" in result["metadata"]["reason"]


def test_generate_next_question_repairs_awkward_visuospatial_route_question(
    monkeypatch,
) -> None:
    bad_payload = {
        "target_domain": "visuospatial",
        "question": "您去公园散步时，是从家的左边还是右边走过去的？",
        "reason": "观察视觉空间能力",
        "source": "qwen",
    }
    repaired_payload = {
        "target_domain": "visuospatial",
        "question": "您刚才说会去公园，小顾听明白了。那出门时通常会先经过哪里？",
        "reason": "用更自然的生活路线描述观察视觉空间线索",
        "sample_answers": {
            "normal": "我一般先经过小区门口，再往公园那边走。",
            "mild_decline": "好像先到门口，后面路线我有点说不清。",
            "vague": "这个我记不太清了。",
        },
        "source": "qwen",
    }
    fake_client = _FakeChatClient(
        [
            json.dumps(bad_payload, ensure_ascii=False),
            json.dumps(repaired_payload, ensure_ascii=False),
        ]
    )
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    result = generate_next_question(
        ["AI访谈问题：您平时喜欢出门做什么？", "老人回答：我会去公园散步。"],
        config=_real_llm_config(),
        covered_domains=["orientation", "memory"],
    )
    repair_call = fake_client.calls[1]

    assert result["metadata"]["source"] == "qwen"
    assert result["target_domain"] == "visuospatial"
    assert "先经过哪里" in result["question"]
    assert "质量修复重试" in result["metadata"]["reason"]
    assert "不通过原因: schema_error: 下一问不够自然友好" in repair_call["messages"][1]["content"]
    assert "您去公园散步时，是从家的左边还是右边走过去的？" in repair_call["messages"][1]["content"]


def test_generate_next_question_repairs_repeated_intro_and_stitched_questions(
    monkeypatch,
) -> None:
    bad_payload = {
        "target_domain": "language",
        "question": "您好，我是小顾。今天是星期几呢？您记得早上吃了什么吗？那您能描述一下今天的天气怎么样吗？",
        "reason": "观察语言表达能力",
        "source": "qwen",
    }
    repaired_payload = {
        "target_domain": "language",
        "question": "您说有点紧张，没关系，慢慢来。那您今天早上做了哪件印象深的事？",
        "reason": "用单一生活问题观察语言表达",
        "sample_answers": {
            "normal": "我早上吃了粥，后来去楼下走了一圈。",
            "mild_decline": "早上做过一些事，但具体顺序有点想不清。",
            "vague": "这个我说不太上来。",
        },
        "source": "qwen",
    }
    fake_client = _FakeChatClient(
        [
            json.dumps(bad_payload, ensure_ascii=False),
            json.dumps(repaired_payload, ensure_ascii=False),
        ]
    )
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    result = generate_next_question(
        ["AI访谈问题：今天大概是星期几？", "老人回答：我怕答错，你问慢一点可以吗？"],
        config=_real_llm_config(),
        covered_domains=["orientation", "memory"],
    )
    repair_call = fake_client.calls[1]

    assert result["metadata"]["source"] == "qwen"
    assert result["target_domain"] == "language"
    assert "您好，我是小顾" not in result["question"]
    assert result["question"].count("？") == 1
    assert "质量修复重试" in result["metadata"]["reason"]
    assert "不通过原因: style_error: 下一问重复介绍小顾" in repair_call["messages"][1]["content"]


def test_generate_next_question_repairs_overconfident_praise_after_vague_answer(
    monkeypatch,
) -> None:
    bad_payload = {
        "target_domain": "memory",
        "question": "您说得很清楚。那您记得今天早上吃了什么吗？",
        "reason": "观察记忆线索",
        "source": "qwen",
    }
    repaired_payload = {
        "target_domain": "memory",
        "question": "没关系，我们慢慢来。那您还记得今天早上吃了什么吗？",
        "reason": "含糊回答后用更温和的方式观察记忆线索",
        "sample_answers": {
            "normal": "我早上喝了粥，还吃了一个鸡蛋。",
            "mild_decline": "好像喝了粥，别的有点想不起来。",
            "vague": "这个我不太记得了。",
        },
        "source": "qwen",
    }
    fake_client = _FakeChatClient(
        [
            json.dumps(bad_payload, ensure_ascii=False),
            json.dumps(repaired_payload, ensure_ascii=False),
        ]
    )
    monkeypatch.setattr(llm_client, "_create_openai_client", lambda _config: fake_client)

    result = generate_next_question(
        ["AI访谈问题：您知道今天是星期几吗？", "老人回答：我不太清楚，最近日子差不多。"],
        config=_real_llm_config(),
        covered_domains=["orientation"],
    )
    repair_call = fake_client.calls[1]

    assert result["metadata"]["source"] == "qwen"
    assert result["target_domain"] == "memory"
    assert "说得很清楚" not in result["question"]
    assert "没关系" in result["question"]
    assert "质量修复重试" in result["metadata"]["reason"]
    assert "含糊回答后不应使用过度肯定评价" in repair_call["messages"][1]["content"]


def test_vlm_client_returns_mock_clock_report_in_demo_mode() -> None:
    config = AppConfig(
        vlm_base_url="https://example.test/v1",
        vlm_api_key="test-key",
        vlm_model="qwen-vl-plus",
        demo_mode=True,
    )
    report = analyze_clock_image(
        image=_png_bytes(),
        filename="clock.png",
        config=config,
    )

    assert report["is_mock"] is True
    assert report["metadata"]["source"] == "mock"
    assert report["metadata"]["reason"] == "DEMO_MODE=true"
    assert report["risk_level"] in RISK_LEVELS
    assert report["disclaimer"] == DISCLAIMER
    assert report["clock_findings"]["visuospatial_evidence"]


def test_vlm_client_returns_mock_clock_report_without_api_key(monkeypatch) -> None:
    def fail_if_called(_config):
        raise AssertionError("real VLM client should not be created without API key")

    monkeypatch.setattr(vlm_client, "_create_openai_client", fail_if_called)
    config = AppConfig(
        vlm_base_url="https://example.test/v1",
        vlm_api_key="",
        vlm_model="qwen-vl-plus",
        demo_mode=False,
    )
    report = analyze_clock_image(filename="clock.png", config=AppConfig())

    assert report["is_mock"] is True
    assert report["metadata"]["source"] == "mock"
    assert analyze_clock_image(
        image=_png_bytes(),
        filename="clock.png",
        config=config,
    )["metadata"]["source"] == "mock"
    assert report["risk_level"] in RISK_LEVELS
    assert report["disclaimer"] == DISCLAIMER
    assert report["clock_findings"]["visuospatial_evidence"]


def test_vlm_client_real_clock_uses_qwen_vl_compatible_chat_settings(
    monkeypatch,
) -> None:
    fake_client = _FakeChatClient(
        json.dumps(_valid_clock_payload(), ensure_ascii=False)
    )
    monkeypatch.setattr(vlm_client, "_create_openai_client", lambda _config: fake_client)

    report = analyze_clock_image(
        image=_png_bytes(),
        filename="clock.png",
        config=_real_vlm_config(),
    )
    call = fake_client.calls[0]

    assert report["is_mock"] is False
    assert report["metadata"]["source"] == "qwen-vl"
    assert report["metadata"]["model"] == "qwen-vl-plus"
    assert report["risk_level"] == "medium"
    assert call["model"] == "qwen-vl-plus"
    assert call["temperature"] == vlm_client.VLM_TEMPERATURE
    assert call["max_tokens"] == vlm_client.VLM_MAX_TOKENS
    assert call["response_format"] == {"type": "json_object"}
    assert _messages_contain_json(call["messages"])
    assert _messages_contain_data_url(call["messages"], "data:image/png;base64,")
    assert "target_time is 11:10" in _message_text(call["messages"])
    assert report["cdt_features"]["target_time_match"] is False


def test_vlm_client_bad_json_returns_fallback(monkeypatch) -> None:
    fake_client = _FakeChatClient("not json")
    monkeypatch.setattr(vlm_client, "_create_openai_client", lambda _config: fake_client)

    report = analyze_clock_image(
        image=_png_bytes(),
        filename="clock.png",
        config=_real_vlm_config(),
    )

    assert report["metadata"]["source"] == "fallback"
    assert report["metadata"]["model"] == "qwen-vl-plus"
    assert report["metadata"]["reason"] == "json_error: 模型返回内容不是有效 JSON"
    assert report["risk_level"] == "unknown"


def test_vlm_client_api_exception_returns_fallback(monkeypatch) -> None:
    fake_client = _FakeChatClient(error=RuntimeError("simulated VLM failure"))
    monkeypatch.setattr(vlm_client, "_create_openai_client", lambda _config: fake_client)

    report = analyze_clock_image(
        image=_png_bytes(),
        filename="clock.png",
        config=_real_vlm_config(),
    )

    assert report["metadata"]["source"] == "fallback"
    assert report["metadata"]["reason"] == "api_error: VLM 调用失败，已回退到安全结果"


def test_vlm_client_data_url_uses_png_or_jpeg_mime_type() -> None:
    png_url = vlm_client._image_to_data_url(_png_bytes(), filename="clock.png")
    jpeg_url = vlm_client._image_to_data_url(_jpeg_bytes(), filename="clock.jpg")

    assert png_url.startswith("data:image/png;base64,")
    assert jpeg_url.startswith("data:image/jpeg;base64,")


def test_qwen_vl_connection_without_path_does_not_call_api(capsys) -> None:
    exit_code = check_qwen_vl_connection.main([])
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "Usage:" in output


def test_asr_client_returns_mock_transcription_without_api_key() -> None:
    result = transcribe_audio(filename="sample.wav", config=AppConfig())

    assert result["is_mock"] is True
    assert result["source_filename"] == "sample.wav"
    assert result["text"]
    assert result["metadata"]["source"] == "mock"
    assert result["metadata"]["reason"] == "DEMO_MODE=true"
    assert result["disclaimer"] == DISCLAIMER


def test_asr_client_demo_mode_does_not_create_real_client(monkeypatch) -> None:
    def fail_if_called(_config):
        raise AssertionError("real ASR client should not be created in demo mode")

    monkeypatch.setattr(asr_client, "_create_openai_client", fail_if_called)

    config = AppConfig(
        asr_base_url="https://example.test/v1",
        asr_api_key="test-key",
        asr_model="qwen-asr",
        demo_mode=True,
    )
    result = transcribe_audio(b"RIFF....WAVE", filename="answer.wav", config=config)

    assert result["is_mock"] is True
    assert result["metadata"]["source"] == "mock"
    assert result["metadata"]["model"] == "qwen-asr"
    assert result["metadata"]["reason"] == "DEMO_MODE=true"


def test_asr_client_missing_config_does_not_create_real_client(monkeypatch) -> None:
    def fail_if_called(_config):
        raise AssertionError("real ASR client should not be created without config")

    monkeypatch.setattr(asr_client, "_create_openai_client", fail_if_called)

    config = AppConfig(
        asr_base_url="https://example.test/v1",
        asr_api_key="",
        asr_model="qwen-asr",
        demo_mode=False,
    )
    result = transcribe_audio(b"RIFF....WAVE", filename="answer.wav", config=config)

    assert result["is_mock"] is True
    assert result["metadata"]["source"] == "mock"
    assert result["metadata"]["reason"] == "ASR 配置不完整"


def test_asr_client_qwen3_flash_uses_chat_input_audio_settings(monkeypatch) -> None:
    fake_client = _FakeAudioClient("今天早上我吃了粥。")
    monkeypatch.setattr(asr_client, "_create_openai_client", lambda _config: fake_client)

    result = transcribe_audio(
        b"RIFF....WAVE",
        filename="answer.wav",
        config=_real_asr_config(),
    )
    call = fake_client.calls[0]
    rendered = json.dumps(result, ensure_ascii=False)

    assert result["is_mock"] is False
    assert result["metadata"]["source"] == "asr-api"
    assert result["metadata"]["model"] == "qwen3-asr-flash"
    assert result["text"] == "今天早上我吃了粥。"
    assert call["model"] == "qwen3-asr-flash"
    assert call["stream"] is False
    assert call["extra_body"] == {"asr_options": {"enable_itn": False}}
    assert call["messages"][0]["role"] == "user"
    content = call["messages"][0]["content"][0]
    assert content["type"] == "input_audio"
    assert content["input_audio"]["data"].startswith("data:audio/wav;base64,")
    assert "sk-secret" not in rendered


def test_asr_client_api_exception_returns_safe_fallback(monkeypatch) -> None:
    fake_client = _FakeAudioClient(error=RuntimeError("temporary failure sk-secret-asr"))
    monkeypatch.setattr(asr_client, "_create_openai_client", lambda _config: fake_client)

    result = transcribe_audio(
        b"ID3fake",
        filename="answer.mp3",
        config=_real_asr_config(),
    )
    rendered = json.dumps(result, ensure_ascii=False)

    assert result["text"] == ""
    assert result["is_mock"] is False
    assert result["metadata"]["source"] == "fallback"
    assert result["metadata"]["reason"] == "api_error: ASR 调用失败，未生成可靠转写"
    assert "[redacted-api-key]" in result["metadata"]["error"]
    assert "sk-secret-asr" not in rendered


def test_asr_client_missing_audio_with_real_config_returns_fallback(monkeypatch) -> None:
    def fail_if_called(_config):
        raise AssertionError("real ASR client should not be created without audio")

    monkeypatch.setattr(asr_client, "_create_openai_client", fail_if_called)

    result = transcribe_audio(filename="empty.wav", config=_real_asr_config())

    assert result["text"] == ""
    assert result["metadata"]["source"] == "fallback"
    assert result["metadata"]["reason"] == "未提供音频，无法转写"


def test_asr_connection_without_path_does_not_call_api(capsys) -> None:
    exit_code = check_asr_connection.main([])
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "Usage:" in output


def test_asr_connection_missing_audio_path_is_not_reported_as_config_error(capsys) -> None:
    exit_code = check_asr_connection.main(["missing-answer.wav"])
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "Audio file not found" in output
    assert "not an ASR configuration error" in output


def test_tts_client_returns_mock_without_tts_config(monkeypatch) -> None:
    def fail_if_called(_url, _payload, _config):
        raise AssertionError("real TTS request should not be made without config")

    monkeypatch.setattr(tts_client, "_post_json", fail_if_called)

    config = AppConfig(
        tts_base_url="",
        tts_api_key="",
        tts_model="qwen-tts",
        demo_mode=False,
    )
    result = synthesize_speech("请您把这串数字倒着说一遍：7-2-5。", config=config)

    assert result["audio_bytes"] is None
    assert result["mime_type"] == "audio/mpeg"
    assert result["metadata"]["source"] == "mock"
    assert result["metadata"]["model"] == "qwen-tts"
    assert result["metadata"]["voice"] == "Cherry"
    assert result["metadata"]["reason"] == "TTS 配置不完整"


def test_tts_client_voice_demo_mode_does_not_call_api(monkeypatch) -> None:
    def fail_if_called(_url, _payload, _config):
        raise AssertionError("real TTS request should not be made in voice demo mode")

    monkeypatch.setattr(tts_client, "_post_json", fail_if_called)

    config = _real_tts_config()
    config = AppConfig(
        tts_base_url=config.tts_base_url,
        tts_api_key=config.tts_api_key,
        tts_model=config.tts_model,
        tts_voice_assistant=config.tts_voice_assistant,
        tts_format=config.tts_format,
        demo_mode=False,
        voice_demo_mode=True,
    )
    result = synthesize_speech("请您慢慢回答。", config=config)

    assert result["metadata"]["source"] == "mock"
    assert result["metadata"]["reason"] == "VOICE_DEMO_MODE=true"


def test_tts_client_demo_mode_serves_bundled_demo_voice(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tts_client, "DEMO_VOICE_DIR", tmp_path)

    config = AppConfig(demo_mode=True)
    text = "您好，今天我们轻松聊几句。"
    bundled_path = tts_client.demo_voice_path_for(
        tts_client.prepare_text_for_tts(text),
        config,
        config.tts_voice_assistant,
        config.tts_model_assistant,
    )
    # qwen-tts 实际常返回 RIFF/WAVE，即便文件名按 mp3 摘要命名。
    bundled_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVEpre-baked")

    result = synthesize_speech(text, config=config)

    assert result["audio_bytes"] == b"RIFF\x00\x00\x00\x00WAVEpre-baked"
    assert result["metadata"]["source"] == "static_audio"
    assert result["metadata"]["cached"] is True
    # mime 按真实内容嗅探为 wav，而不是文件名暗示的 mp3。
    assert result["mime_type"] == "audio/wav"


def test_tts_client_demo_mode_without_bundled_voice_still_mocks(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tts_client, "DEMO_VOICE_DIR", tmp_path)

    result = synthesize_speech("一段没有预置音频的文本。", config=AppConfig(demo_mode=True))

    assert result["audio_bytes"] is None
    assert result["metadata"]["source"] == "mock"
    assert result["metadata"]["reason"] == "DEMO_MODE=true"


def test_tts_client_fake_success_returns_audio_bytes(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tts_client, "DEFAULT_TTS_CACHE_DIR", tmp_path)
    audio_bytes = b"fake mp3 bytes"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")

    def fake_post_json(url, payload, _config):
        assert url == "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
        assert payload["model"] == "qwen-tts"
        assert payload["input"]["voice"] == "Cherry"
        assert "请您" in payload["input"]["text"]
        assert "7，2，5" in payload["input"]["text"]
        assert "7-2-5" not in payload["input"]["text"]
        return {
            "status_code": 200,
            "output": {
                "audio": {
                    "data": encoded_audio,
                }
            },
        }

    monkeypatch.setattr(tts_client, "_post_json", fake_post_json)

    result = synthesize_speech(
        "请您把这串数字倒着说一遍：7-2-5。",
        config=_real_tts_config(),
    )

    assert result["audio_bytes"] == audio_bytes
    assert result["mime_type"] == "audio/mpeg"
    assert result["metadata"]["source"] == "tts"
    assert result["metadata"]["model"] == "qwen-tts"
    assert result["metadata"]["voice"] == "Cherry"
    assert result["metadata"]["reason"] == ""
    assert result["metadata"]["cached"] is False
    assert result["metadata"]["cache_path"].startswith("tts_")


def test_tts_prepare_text_separates_digit_sequences_without_changing_display_text() -> None:
    original = "请您把这串数字倒着说一遍：3-8-1。等待 10-20 秒。"

    prepared = tts_client.prepare_text_for_tts(original)

    assert prepared == "请您把这串数字倒着说一遍：3，8，1。等待 10，20 秒。"
    assert original == "请您把这串数字倒着说一遍：3-8-1。等待 10-20 秒。"


def test_tts_client_fake_url_success_downloads_audio(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tts_client, "DEFAULT_TTS_CACHE_DIR", tmp_path)

    def fake_post_json(_url, _payload, _config):
        return {
            "status_code": 200,
            "output": {
                "audio": {
                    "url": "https://example.test/tts_check.wav",
                }
            },
        }

    def fake_download_audio_url(url):
        assert url == "https://example.test/tts_check.wav"
        return b"RIFF....WAVE", "audio/wav"

    monkeypatch.setattr(tts_client, "_post_json", fake_post_json)
    monkeypatch.setattr(tts_client, "_download_audio_url", fake_download_audio_url)

    result = synthesize_speech("请您慢慢回答。", config=_real_tts_config())

    assert result["audio_bytes"] == b"RIFF....WAVE"
    assert result["mime_type"] == "audio/wav"
    assert result["metadata"]["source"] == "tts"
    assert result["metadata"]["cached"] is False


def test_tts_client_fake_url_can_skip_python_download(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tts_client, "DEFAULT_TTS_CACHE_DIR", tmp_path)

    def fake_post_json(_url, _payload, _config):
        return {
            "status_code": 200,
            "output": {
                "audio": {
                    "url": "https://example.test/tts_check.wav",
                }
            },
        }

    def fail_if_downloaded(_url):
        raise AssertionError("remote audio URL should be handed to the browser")

    monkeypatch.setattr(tts_client, "_post_json", fake_post_json)
    monkeypatch.setattr(tts_client, "_download_audio_url", fail_if_downloaded)

    result = synthesize_speech(
        "请您慢慢回答。",
        config=_real_tts_config(),
        prefer_remote_url=True,
    )

    assert result["audio_bytes"] is None
    assert result["audio_url"] == "https://example.test/tts_check.wav"
    assert result["mime_type"] == "audio/wav"
    assert result["metadata"]["source"] == "tts_url"
    assert result["metadata"]["cached"] is False
    assert "cache_path" not in result["metadata"]


def test_tts_client_cosyvoice_model_uses_speech_synthesizer_endpoint(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tts_client, "DEFAULT_TTS_CACHE_DIR", tmp_path)
    call = {}

    def fake_post_json(url, payload, _config):
        call["url"] = url
        call["payload"] = payload
        return {
            "status_code": 200,
            "output": {
                "audio": {
                    "url": "https://example.test/cosyvoice_patient.mp3",
                }
            },
        }

    def fake_download_audio_url(url):
        assert url == "https://example.test/cosyvoice_patient.mp3"
        return b"cosyvoice mp3 bytes", "audio/mpeg"

    monkeypatch.setattr(tts_client, "_post_json", fake_post_json)
    monkeypatch.setattr(tts_client, "_download_audio_url", fake_download_audio_url)

    result = synthesize_speech(
        "今天早上我看了手机上的日期。",
        model="cosyvoice-v3-flash",
        voice="longlaoyi_v3",
        config=_real_tts_config(),
    )

    assert call["url"] == "https://dashscope.aliyuncs.com/api/v1/services/audio/tts/SpeechSynthesizer"
    assert call["payload"]["model"] == "cosyvoice-v3-flash"
    assert call["payload"]["input"]["voice"] == "longlaoyi_v3"
    assert call["payload"]["input"]["format"] == "mp3"
    assert result["audio_bytes"] == b"cosyvoice mp3 bytes"
    assert result["metadata"]["source"] == "tts"
    assert result["metadata"]["model"] == "cosyvoice-v3-flash"
    assert result["metadata"]["voice"] == "longlaoyi_v3"


def test_tts_client_fake_exception_returns_safe_fallback(monkeypatch) -> None:
    def fake_post_json(_url, _payload, _config):
        raise RuntimeError("temporary failure sk-secret-tts")

    monkeypatch.setattr(tts_client, "_post_json", fake_post_json)

    result = synthesize_speech("请您慢慢回答。", config=_real_tts_config())
    rendered = json.dumps(result, ensure_ascii=False)

    assert result["audio_bytes"] is None
    assert result["metadata"]["source"] == "fallback"
    assert result["metadata"]["reason"] == "api_error: TTS 调用失败，未生成真实音频"
    assert "[redacted-api-key]" in result["metadata"]["error"]
    assert "sk-secret-tts" not in rendered


def test_tts_client_cosyvoice_failure_returns_clear_fallback_reason(monkeypatch) -> None:
    def fake_post_json(_url, _payload, _config):
        raise RuntimeError("voice not supported sk-secret-tts")

    monkeypatch.setattr(tts_client, "_post_json", fake_post_json)

    result = synthesize_speech(
        "今天早上我看了手机上的日期。",
        model="cosyvoice-v3-flash",
        voice="longlaoyi_v3",
        config=_real_tts_config(),
        use_cache=False,
    )
    rendered = json.dumps(result, ensure_ascii=False)

    assert result["audio_bytes"] is None
    assert result["metadata"]["source"] == "fallback"
    assert "voice longlaoyi_v3 may require CosyVoice model/API" in result["metadata"]["reason"]
    assert "[redacted-api-key]" in result["metadata"]["error"]
    assert "sk-secret-tts" not in rendered


def test_tts_client_hits_cache_on_second_same_text(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tts_client, "DEFAULT_TTS_CACHE_DIR", tmp_path)
    calls = {"count": 0}
    audio_bytes = b"cached after first call"
    encoded_audio = base64.b64encode(audio_bytes).decode("ascii")

    def fake_post_json(_url, _payload, _config):
        calls["count"] += 1
        return {"status_code": 200, "output": {"audio": {"data": encoded_audio}}}

    monkeypatch.setattr(tts_client, "_post_json", fake_post_json)

    config = _real_tts_config()
    first = synthesize_speech("请您慢慢回答。", config=config)
    second = synthesize_speech("请您慢慢回答。", config=config)

    assert calls["count"] == 1
    assert first["metadata"]["source"] == "tts"
    assert first["metadata"]["cached"] is False
    assert second["metadata"]["source"] == "tts_cache"
    assert second["metadata"]["cached"] is True
    assert second["audio_bytes"] == audio_bytes
    assert second["metadata"]["cache_path"] == first["metadata"]["cache_path"]


def test_tts_cache_key_changes_for_different_text(tmp_path) -> None:
    config = _real_tts_config()
    first_path = tts_client._cache_path_for("第一个问题。", config, "Cherry")
    second_path = tts_client._cache_path_for("第二个问题。", config, "Cherry")

    assert first_path.name != second_path.name


def test_tts_cache_key_changes_for_different_voice(tmp_path) -> None:
    config = _real_tts_config()
    assistant_path = tts_client._cache_path_for("同一句演示文本。", config, "Cherry")
    patient_path = tts_client._cache_path_for("同一句演示文本。", config, "DemoPatient")

    assert assistant_path.name != patient_path.name


def test_tts_cache_key_changes_for_different_model(tmp_path) -> None:
    config = _real_tts_config()
    qwen_path = tts_client._cache_path_for(
        "同一句演示文本。",
        config,
        "longlaoyi_v3",
        model="qwen-tts",
    )
    cosyvoice_path = tts_client._cache_path_for(
        "同一句演示文本。",
        config,
        "longlaoyi_v3",
        model="cosyvoice-v3-flash",
    )

    assert qwen_path.name != cosyvoice_path.name


def test_tts_client_no_cache_ignores_existing_cache(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(tts_client, "DEFAULT_TTS_CACHE_DIR", tmp_path)
    config = _real_tts_config()
    text = "请您慢慢回答。"
    cache_path = tts_client._cache_path_for(text, config, "Cherry")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(b"old cached bytes")

    fresh_audio = b"fresh no-cache bytes"
    encoded_audio = base64.b64encode(fresh_audio).decode("ascii")

    def fake_post_json(_url, _payload, _config):
        return {"status_code": 200, "output": {"audio": {"data": encoded_audio}}}

    monkeypatch.setattr(tts_client, "_post_json", fake_post_json)

    result = synthesize_speech(text, config=config, use_cache=False)

    assert result["audio_bytes"] == fresh_audio
    assert result["metadata"]["source"] == "tts"
    assert result["metadata"]["cached"] is False
    assert "cache_path" not in result["metadata"]
    assert cache_path.read_bytes() == b"old cached bytes"


def test_tts_result_structure_is_stable() -> None:
    result = synthesize_speech("请您慢慢回答。", config=AppConfig())

    assert set(result) == {"audio_bytes", "mime_type", "metadata"}
    assert {"source", "model", "voice", "reason", "cached"} <= set(result["metadata"])


def test_tts_connection_script_uses_mock_without_network(monkeypatch, capsys) -> None:
    monkeypatch.setattr(check_tts_connection, "load_config", lambda: AppConfig())

    exit_code = check_tts_connection.main([])
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "VOICE_DEMO_MODE=false" in output
    assert "QWEN_API_KEY configured=False" in output
    assert "TTS 为非实时合成，首次生成可能需要数秒到十几秒。" in output
    assert "如果单条诊断成功但页面批量失败，优先尝试降低并发或重试失败项。" in output
    assert "role=assistant" in output
    assert "final_model=qwen-tts" in output
    assert "source=mock" in output
    assert "cached=false" in output
    assert "audio_size=0" in output


def test_tts_connection_script_supports_patient_role_without_network(monkeypatch, capsys) -> None:
    monkeypatch.setattr(check_tts_connection, "load_config", lambda: AppConfig())

    exit_code = check_tts_connection.main(["--role", "patient", "今天早上我看了手机上的日期。"])
    output = capsys.readouterr().out

    assert exit_code == 2
    assert "role=patient" in output
    assert "final_model=cosyvoice-v3-flash" in output
    assert "final_voice=longlaoyi_v3" in output
    assert "model=cosyvoice-v3-flash" in output
    assert "voice=longlaoyi_v3" in output
    assert "source=mock" in output


def test_embedding_client_returns_stable_mock_vector_without_api_key() -> None:
    config = AppConfig()
    first = embed_text("测试文本", config=config, dimensions=8)
    second = embed_text("测试文本", config=config, dimensions=8)
    other = embed_text("另一段文本", config=config, dimensions=8)

    assert first == second
    assert first != other
    assert len(first) == 8
    assert all(-1 <= value <= 1 for value in first)


def test_prompt_files_exist_and_request_json_output() -> None:
    prompt_dir = PROJECT_ROOT / "core" / "prompts"
    prompt_files = [
        "dialog_eval.md",
        "clock_eval.md",
        "trend_report.md",
        "family_report.md",
        "next_question.md",
    ]

    for filename in prompt_files:
        content = (prompt_dir / filename).read_text(encoding="utf-8")
        assert "JSON" in content
        assert "医学诊断" in content or filename == "next_question.md"


def test_next_question_prompt_keeps_json_schema_and_xiaogu_style() -> None:
    content = (PROJECT_ROOT / "core" / "prompts" / "next_question.md").read_text(
        encoding="utf-8"
    )

    assert "小顾" in content
    assert "温柔的年轻晚辈" in content
    assert "小顾陪您慢慢来" in content
    assert "年轻家人" in content
    assert "只输出 JSON" in content
    assert "question" in content
    assert "target_domain" in content
    assert "reason" in content
    assert "sample_answers" in content
    assert "source" in content
    assert "尽量不超过 70 个中文字符" in content
    assert "只有第一轮" in content
    assert "每轮只问一个核心问题" in content
    assert "不要复制、改写或拼接" in content
    assert "我听明白了" in content
    assert "含糊、不确定、记不清" in content
    assert "禁止说“您说得很清楚”" in content
    assert "没关系，慢慢来" in content
    assert "从家的左边还是右边去公园" in content
    assert "避免固定使用“苹果、钥匙、报纸”" in content
    assert "不要每次都使用钥匙、水杯、眼镜" in content
    assert "不要输出 Markdown" in content


def _real_llm_config() -> AppConfig:
    return AppConfig(
        llm_base_url="https://example.test/v1",
        llm_api_key="test-key",
        llm_model="qwen-plus",
        demo_mode=False,
    )


def _real_vlm_config() -> AppConfig:
    return AppConfig(
        vlm_base_url="https://example.test/v1",
        vlm_api_key="test-key",
        vlm_model="qwen-vl-plus",
        demo_mode=False,
    )


def _real_asr_config() -> AppConfig:
    return AppConfig(
        asr_base_url="https://example.test/v1",
        asr_api_key="sk-secret-asr",
        asr_model="qwen3-asr-flash",
        demo_mode=False,
    )


def _real_tts_config() -> AppConfig:
    return AppConfig(
        qwen_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        qwen_api_key="sk-secret-tts",
        tts_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        tts_api_key="sk-secret-tts",
        tts_model="qwen-tts",
        tts_voice_assistant="Cherry",
        tts_format="mp3",
        demo_mode=False,
    )


def _valid_dialogue_payload(
    risk_level: str = "low",
    evidence: Optional[list] = None,
) -> dict:
    return {
        "domain_scores": {domain: 0.8 for domain in COGNITIVE_DOMAINS},
        "evidence": evidence
        if evidence is not None
        else [
            {
                "domain": "memory",
                "source": "dialog",
                "text": "能复述刚才提到的早餐内容。",
            }
        ],
        "risk_level": risk_level,
        "explanation": "对话内容较连贯，仅作为风险提示参考。",
        "disclaimer": DISCLAIMER,
    }


def _valid_clock_payload() -> dict:
    return {
        "domain_scores": {
            "visuospatial": 0.55,
            "executive_function": 0.6,
        },
        "evidence": [
            "数字集中在右侧。",
            "指针方向不够准确。",
        ],
        "clock_findings": {
            "number_placement": "数字集中在右侧，间距不均。",
            "hand_accuracy": "指针方向不够准确。",
            "visuospatial_evidence": [
                "数字集中在右侧。",
                "圆形轮廓略不规则。",
            ],
        },
        "cdt_features": {
            "numbers_complete": True,
            "number_order_correct": True,
            "number_spacing": "crowded",
            "number_distribution": "right_shifted",
            "hands_present": True,
            "target_time_match": False,
            "center_anchor_clear": True,
        },
        "risk_level": "中风险",
        "explanation": "画钟图中存在视觉空间和执行功能相关风险信号。",
        "disclaimer": DISCLAIMER,
    }


def _png_bytes() -> bytes:
    return b"\x89PNG\r\n\x1a\n" + b"clock"


def _jpeg_bytes() -> bytes:
    return b"\xff\xd8\xff" + b"clock"


class _FakeChatClient:
    def __init__(self, content="", error: Optional[Exception] = None) -> None:
        self.contents = list(content) if isinstance(content, list) else [content]
        if not self.contents:
            self.contents = [""]
        self.error = error
        self.calls: list[dict] = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        content_index = min(len(self.calls) - 1, len(self.contents) - 1)

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self.contents[content_index]),
                )
            ]
        )


class _FakeAudioClient:
    def __init__(self, text: str = "", error: Optional[Exception] = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict] = []
        self.audio = SimpleNamespace(transcriptions=self)
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if "messages" in kwargs:
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=self.text),
                    )
                ]
            )
        return SimpleNamespace(text=self.text)


def _clear_model_env(monkeypatch) -> None:
    for key in [
        "QWEN_BASE_URL",
        "QWEN_API_KEY",
        "LLM_BASE_URL",
        "LLM_API_KEY",
        "LLM_MODEL",
        "VLM_BASE_URL",
        "VLM_API_KEY",
        "VLM_MODEL",
        "ASR_BASE_URL",
        "ASR_API_KEY",
        "ASR_MODEL",
        "TTS_BASE_URL",
        "TTS_API_KEY",
        "TTS_MODEL",
        "TTS_MODEL_ASSISTANT",
        "TTS_MODEL_PATIENT_DEMO",
        "TTS_VOICE_ASSISTANT",
        "TTS_VOICE_PATIENT_DEMO",
        "TTS_FORMAT",
        "DEMO_MODE",
        "VOICE_DEMO_MODE",
        "USE_CACHE",
        "STAFF_PASSWORD",
    ]:
        monkeypatch.setenv(key, "")


def _assert_fallback_report(report: dict, expected_reason: str) -> None:
    fallback = fallback_result()
    for key, value in fallback.items():
        assert report[key] == value

    assert report["metadata"]["source"] == "fallback"
    assert report["metadata"]["model"] == "qwen-plus"
    assert report["metadata"]["reason"] == expected_reason


def _messages_contain_json(messages: list[dict]) -> bool:
    combined_content = "\n".join(_message_text(message) for message in messages)
    return "json" in combined_content.lower()


def _messages_contain_data_url(messages: list[dict], prefix: str) -> bool:
    return prefix in "\n".join(_message_text(message) for message in messages)


def _message_text(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return "\n".join(_message_text(item) for item in value.values())
    if isinstance(value, list):
        return "\n".join(_message_text(item) for item in value)
    return ""
