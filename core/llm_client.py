from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from core.config import AppConfig, load_config
from core.mock_data import (
    INTERVIEW_COMPLETED_MESSAGE,
    all_dialog_domains_covered,
    get_dialog_example_answers,
    get_next_preset_interview_question,
    infer_dialog_question_type,
    normalize_dialog_domains,
)
from core.report import build_family_brief, generate_mock_dialog_report, summarize_trend
from core.schemas import (
    COGNITIVE_DOMAINS,
    DISCLAIMER,
    calibrate_dialogue_result,
    fallback_result,
    validate_dialogue_assessment,
)


PROMPT_DIR = Path(__file__).resolve().parent / "prompts"
LLM_TEMPERATURE = 0.2
LLM_MAX_TOKENS = 900
NEXT_QUESTION_MAX_TOKENS = 450
LLM_TIMEOUT_SECONDS = 20.0
DEFAULT_DIALOG_EVAL_PROMPT = (
    "Return only a valid JSON object. Do not output Markdown, code fences, "
    "or extra explanation. The JSON object must include domain_scores, "
    "evidence, risk_level, explanation, and disclaimer."
)
STALE_FIXED_QUESTION_MATERIALS = (
    ("苹果", "钥匙", "报纸"),
    ("钥匙", "水杯", "眼镜"),
)
REPETITIVE_NEXT_QUESTION_OPENERS = (
    "我听明白了",
    "我大概听明白了",
    "没关系，慢慢来",
    "慢慢来就好",
    "我们慢慢说就好",
    "小顾陪您慢慢来",
)
UNCERTAIN_ANSWER_OPENERS = (
    "先不着急",
    "答不上来也可以",
    "我们换个轻松一点的问题",
    "按您想到的说就好",
)
NEUTRAL_ANSWER_OPENERS = (
    "好的，我记下来了",
    "明白了",
    "咱们接着聊一点生活里的事",
    "这个我先记下来",
)


def evaluate_dialogue(
    messages: list[str], config: Optional[AppConfig] = None
) -> dict[str, Any]:
    active_config = config or load_config()
    if active_config.demo_mode:
        return _with_metadata(
            generate_mock_dialog_report(messages),
            source="mock",
            model=active_config.llm_model,
            reason="DEMO_MODE=true",
        )

    if not _should_use_real_llm(active_config):
        return _with_metadata(
            generate_mock_dialog_report(messages),
            source="mock",
            model=active_config.llm_model,
            reason="LLM 配置不完整",
        )

    try:
        content = _request_dialogue_evaluation(messages, active_config)
    except Exception:
        return _fallback_with_metadata(
            active_config,
            reason="api_error: LLM 调用失败，已回退到安全结果",
        )

    try:
        payload = _parse_json_object(content)
    except json.JSONDecodeError:
        return _fallback_with_metadata(
            active_config,
            reason="json_error: 模型返回内容不是有效 JSON",
        )
    except ValueError:
        return _fallback_with_metadata(
            active_config,
            reason="json_error: 模型返回 JSON 不是对象",
        )

    try:
        return _with_metadata(
            calibrate_dialogue_result(
                validate_dialogue_assessment(_normalize_dialogue_payload(payload))
            ),
            source="qwen",
            model=active_config.llm_model,
        )
    except ValueError:
        return _fallback_with_metadata(
            active_config,
            reason="schema_error: 模型返回 JSON 未通过 schema 校验",
        )


def generate_trend_report(
    sessions: list[dict[str, Any]], config: Optional[AppConfig] = None
) -> dict[str, Any]:
    _ = config or load_config()
    return summarize_trend(sessions)


def generate_family_report(
    sessions: list[dict[str, Any]], config: Optional[AppConfig] = None
) -> dict[str, Any]:
    _ = config or load_config()
    if not sessions:
        return {
            "trend_label": "unknown",
            "summary": "没有可用于生成家属简报的模拟 session。",
            "family_reminders": [
                "请先完成至少一次 mock 会话，或加载 demo fixture 数据。"
            ],
            "disclaimer": DISCLAIMER,
            "is_mock": True,
        }

    report = build_family_brief(sessions)
    report["is_mock"] = True
    return report


def generate_next_question(
    messages: list[str],
    config: Optional[AppConfig] = None,
    covered_domains: Optional[list[str]] = None,
) -> dict[str, Any]:
    active_config = config or load_config()
    covered = normalize_dialog_domains(covered_domains or [])

    if all_dialog_domains_covered(covered):
        return _next_question_result(
            question=INTERVIEW_COMPLETED_MESSAGE,
            target_domain="",
            source="mock",
            model=active_config.llm_model,
            reason="主要认知域已覆盖",
            is_mock=True,
            completed=True,
            sample_answers={},
        )

    if active_config.demo_mode:
        return _mock_next_question(active_config, covered, reason="DEMO_MODE=true")

    if not _should_use_real_llm(active_config):
        return _mock_next_question(active_config, covered, reason="LLM 配置不完整")

    try:
        content = _request_next_question(messages, active_config, covered)
    except Exception:
        return _next_question_fallback(active_config, "api_error: 下一问生成失败")

    try:
        payload = _parse_json_object(content)
    except json.JSONDecodeError:
        return _next_question_fallback(active_config, "json_error: 下一问返回内容不是有效 JSON")
    except ValueError:
        return _next_question_fallback(active_config, "json_error: 下一问返回 JSON 不是对象")

    result, rejection_reason, rejected_question = _next_question_from_payload(
        payload,
        active_config,
        covered,
        messages,
    )
    if result is not None:
        return result

    try:
        repaired_content = _request_next_question(
            messages,
            active_config,
            covered,
            repair_reason=rejection_reason,
            rejected_question=rejected_question,
        )
    except Exception:
        return _next_question_fallback(
            active_config,
            f"quality_retry_failed: {rejection_reason}; repair_api_error: 下一问修复重试失败",
        )

    try:
        repaired_payload = _parse_json_object(repaired_content)
    except json.JSONDecodeError:
        return _next_question_fallback(
            active_config,
            f"quality_retry_failed: {rejection_reason}; repair_json_error: 修复重试不是有效 JSON",
        )
    except ValueError:
        return _next_question_fallback(
            active_config,
            f"quality_retry_failed: {rejection_reason}; repair_json_error: 修复重试 JSON 不是对象",
        )

    repaired_result, repaired_rejection_reason, _ = _next_question_from_payload(
        repaired_payload,
        active_config,
        covered,
        messages,
        repair_reason=rejection_reason,
    )
    if repaired_result is not None:
        return repaired_result

    return _next_question_fallback(
        active_config,
        f"quality_retry_failed: {repaired_rejection_reason}",
    )


def _should_use_real_llm(config: AppConfig) -> bool:
    return (
        not config.demo_mode
        and bool(config.llm_base_url.strip())
        and bool(config.llm_api_key.strip())
        and bool(config.llm_model.strip())
    )


def _request_dialogue_evaluation(messages: list[str], config: AppConfig) -> str:
    prompt = _read_prompt("dialog_eval.md")
    client = _create_openai_client(config)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": _build_dialogue_user_content(messages)},
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=LLM_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return _extract_message_content(response)


def _request_next_question(
    messages: list[str],
    config: AppConfig,
    covered_domains: list[str],
    repair_reason: str = "",
    rejected_question: str = "",
) -> str:
    prompt = _read_prompt("next_question.md")
    client = _create_openai_client(config)
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=[
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": _build_next_question_user_content(
                    messages,
                    covered_domains,
                    repair_reason=repair_reason,
                    rejected_question=rejected_question,
                ),
            },
        ],
        temperature=LLM_TEMPERATURE,
        max_tokens=NEXT_QUESTION_MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return _extract_message_content(response)


def _create_openai_client(config: AppConfig) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        timeout=LLM_TIMEOUT_SECONDS,
    )


def _read_prompt(filename: str) -> str:
    try:
        prompt = (PROMPT_DIR / filename).read_text(encoding="utf-8").strip()
    except OSError:
        return DEFAULT_DIALOG_EVAL_PROMPT

    return prompt or DEFAULT_DIALOG_EVAL_PROMPT


def _build_dialogue_user_content(messages: list[str]) -> str:
    clean_messages = [message.strip() for message in messages if message.strip()]
    if not clean_messages:
        return "本次没有可用对话文本。请返回 unknown 风险等级，并仅输出 JSON。"

    lines = [f"{index}. {message}" for index, message in enumerate(clean_messages, 1)]
    return (
        "请根据以下老人端对话文本完成认知风险提示。"
        "只返回符合系统 prompt 中 schema 的 JSON 对象。\n\n"
        + "\n".join(lines)
    )


def _build_next_question_user_content(
    messages: list[str],
    covered_domains: list[str],
    repair_reason: str = "",
    rejected_question: str = "",
) -> str:
    clean_messages = [message.strip() for message in messages if message.strip()]
    remaining_domains = [
        domain for domain in COGNITIVE_DOMAINS if domain not in set(covered_domains)
    ]
    lines = [f"{index}. {message}" for index, message in enumerate(clean_messages, 1)]
    dialogue_text = "\n".join(lines) if lines else "暂无已完成问答。"
    last_answer = _last_elder_answer(clean_messages)
    content = (
        "请以“小顾”的口吻，根据当前访谈历史生成下一轮自然问题。Return only JSON.\n"
        f"已覆盖认知域: {', '.join(covered_domains) or 'none'}\n"
        f"优先选择未覆盖认知域: {', '.join(remaining_domains) or 'none'}\n"
        f"老人上一轮回答: {last_answer or '暂无'}\n"
        "请优先结合老人上一轮回答的内容、遗漏或表达特点来生成下一问。\n"
        "question 先具体回应上一轮回答，再自然追问；通常 1-2 句，尽量不超过 70 个中文字符。\n"
        "小顾的人设像家里温柔的年轻晚辈：亲切、有耐心，愿意陪老人轻松说，可以自然轻柔鼓励。\n"
        "不要连续复用同一句承接语；尤其避免多轮重复“我听明白了”“没关系，慢慢来”“慢慢来就好”。\n"
        "小顾对长者说话时统一使用“您”；老人示例回答可保留自然口语里的“你/我”。\n"
        "如果需要安抚，可以轮换使用“先不着急”“按您想到的说就好”“答不上来也可以”等简短说法。\n"
        "如果老人上一轮回答含糊、不确定、记不清、说不上来、弄混或答非所问，禁止说“您说得很清楚”“记得很清楚”“很准确”“数得很好”等肯定表现的话；"
        "这时请用低压力承接，不要评价老人表现，也不要固定套用同一句口癖。\n"
        "只有第一轮、暂无已完成问答时，才可以介绍“您好，我是小顾”；已有对话历史后不要重复自我介绍。\n"
        "每轮只问一个核心问题，不要连续问多个问题，也不要把多个历史问题拼接成一串。\n"
        "不要复制、改写或拼接之前 AI 已经问过的问题原文；下一问应承接老人上一轮回答。\n"
        "语气温和陪伴，不像考试、审问或后台表单；不过度夸张、幼态或哄小孩，也不要居高临下评价老人。\n"
        "不要机械复用固定记词材料；记忆词语应根据当前对话临时选择，避免总是苹果、钥匙、报纸。\n"
        "视觉空间问题可以使用熟悉路线、房间之间怎么走、物品大概位置或日常出行顺序，不要总是钥匙、水杯、眼镜。\n"
        "视觉空间问题不要生硬二选一问“从家的左边还是右边去公园”。\n"
        "不要重复已覆盖认知域。输出字段必须包含 target_domain、question、reason、sample_answers、source。\n"
        "sample_answers 必须包含 normal、mild_decline、vague 三类回答，并且必须与 question 语义匹配。\n\n"
        "问题必须是一轮可回答的自包含问题，语气低压力、生活化，适合老人访谈。"
        "禁止抽象数学题、抽象几何题、画图题或考试式题目，例如正方形、三角形、几何拆分。"
        "视觉空间问题应使用路线、左右位置、日常物品摆放等生活情境。"
        "注意力/倒背数字题必须直接包含数字，记忆题必须直接给出要记住的词。"
        "如果是倒背数字题，sample_answers.normal 必须是正确反向数字。"
        "如果是位置关系或路线题，sample_answers 必须围绕左右、位置或路线回答。"
        "不要让系统稍后再补充材料。\n\n"
        + dialogue_text
    )
    if repair_reason:
        content += (
            "\n\n上一次输出未通过本地质量校验，请根据原因重写一次。\n"
            f"不通过原因: {repair_reason}\n"
            f"上一次问题: {rejected_question or '无可用问题文本'}\n"
            "请保留同一 JSON schema，优先选择未覆盖认知域，重新生成一个更自然、"
            "更像小顾陪长者聊天的问题。不要解释修复过程。"
        )
    return content


def _last_elder_answer(messages: list[str]) -> str:
    for message in reversed(messages):
        if "老人回答：" in message:
            return message.split("老人回答：", 1)[1].strip()
    return ""


def _extract_message_content(response: Any) -> str:
    content = response.choices[0].message.content
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM response content is empty")
    return content.strip()


def _parse_json_object(content: str) -> dict[str, Any]:
    candidate = _strip_json_fence(content)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(candidate[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("LLM response JSON must be an object")
    return parsed


def _strip_json_fence(content: str) -> str:
    text = content.strip()
    if not text.startswith("```"):
        return text

    lines = text.splitlines()
    if lines and lines[0].strip().startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def _normalize_dialogue_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    normalized["risk_level"] = _normalize_risk_level(payload.get("risk_level"))
    normalized["domain_scores"] = _normalize_domain_scores(
        payload.get("domain_scores")
    )
    normalized["evidence"] = _normalize_evidence(payload.get("evidence"))

    disclaimer = payload.get("disclaimer")
    if not isinstance(disclaimer, str) or "不构成医学诊断" not in disclaimer:
        normalized["disclaimer"] = DISCLAIMER

    return normalized


def _normalize_risk_level(value: Any) -> str:
    if not isinstance(value, str):
        return "unknown"

    normalized = value.strip().lower()
    risk_map = {
        "low": "low",
        "low risk": "low",
        "低": "low",
        "低风险": "low",
        "medium": "medium",
        "medium risk": "medium",
        "moderate": "medium",
        "中": "medium",
        "中等": "medium",
        "中风险": "medium",
        "中等风险": "medium",
        "high": "high",
        "high risk": "high",
        "高": "high",
        "高风险": "high",
        "unknown": "unknown",
        "未知": "unknown",
        "不确定": "unknown",
        "无法判断": "unknown",
    }
    return risk_map.get(normalized, "unknown")


def _normalize_domain_scores(value: Any) -> dict[str, Optional[float]]:
    scores = value if isinstance(value, dict) else {}
    return {domain: _normalize_score(scores.get(domain)) for domain in COGNITIVE_DOMAINS}


def _normalize_score(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None

    try:
        score = float(value)
    except (TypeError, ValueError):
        return None

    if not 0 <= score <= 1:
        return None
    return score


def _normalize_evidence(value: Any) -> list[dict[str, str]]:
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, list):
        items = value
    else:
        return []

    evidence: list[dict[str, str]] = []
    for item in items:
        normalized = _normalize_evidence_item(item)
        if normalized is not None:
            evidence.append(normalized)
    return evidence


def _normalize_evidence_item(value: Any) -> Optional[dict[str, str]]:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        return {
            "domain": _infer_evidence_domain(text),
            "source": "dialog",
            "text": text,
        }

    if not isinstance(value, dict):
        return None

    raw_text = value.get("text") or value.get("evidence") or value.get("content")
    if not isinstance(raw_text, str) or not raw_text.strip():
        return None

    text = raw_text.strip()
    raw_domain = value.get("domain")
    domain = raw_domain if raw_domain in COGNITIVE_DOMAINS else _infer_evidence_domain(text)
    return {
        "domain": domain,
        "source": "dialog",
        "text": text,
    }


def _infer_evidence_domain(text: str) -> str:
    keyword_domains = [
        ("orientation", ("时间", "日期", "星期", "地点", "今天", "昨天")),
        ("memory", ("记得", "回忆", "复述", "刚才", "早餐", "早饭")),
        ("attention", ("注意", "计算", "倒数", "专注")),
        ("executive_function", ("安排", "计划", "步骤", "执行", "照着做")),
        ("visuospatial", ("左", "右", "方向", "空间", "位置")),
    ]
    for domain, keywords in keyword_domains:
        if any(keyword in text for keyword in keywords):
            return domain
    return "language"


def _fallback_with_metadata(config: AppConfig, reason: str) -> dict[str, Any]:
    return _with_metadata(
        fallback_result(),
        source="fallback",
        model=config.llm_model,
        reason=reason,
    )


def _mock_next_question(
    config: AppConfig,
    covered_domains: list[str],
    reason: str,
) -> dict[str, Any]:
    preset = get_next_preset_interview_question(covered_domains)
    if preset is None:
        return _next_question_result(
            question=INTERVIEW_COMPLETED_MESSAGE,
            target_domain="",
            source="mock",
            model=config.llm_model,
            reason="主要认知域已覆盖",
            is_mock=True,
            completed=True,
            sample_answers={},
        )
    sample_answers = get_dialog_example_answers(
        preset["question"],
        target_domain=preset["domain"],
    )
    return _next_question_result(
        question=preset["question"],
        target_domain=preset["domain"],
        source="mock",
        model=config.llm_model,
        reason=reason,
        is_mock=True,
        completed=False,
        sample_answers=sample_answers,
    )


def _next_question_fallback(config: AppConfig, reason: str) -> dict[str, Any]:
    return _next_question_result(
        question="",
        target_domain="",
        source="fallback",
        model=config.llm_model,
        reason=reason,
        is_mock=False,
        completed=False,
        sample_answers={},
    )


def _next_question_from_payload(
    payload: dict[str, Any],
    config: AppConfig,
    covered: list[str],
    messages: list[str],
    repair_reason: str = "",
) -> tuple[Optional[dict[str, Any]], str, str]:
    question = payload.get("question")
    rejected_question = question.strip() if isinstance(question, str) else ""
    if not isinstance(question, str) or not question.strip():
        return None, "schema_error: 下一问缺少 question", rejected_question
    question = _polish_next_question_style(question.strip(), messages)

    target_domain = payload.get("target_domain")
    if target_domain not in COGNITIVE_DOMAINS:
        target_domain = infer_dialog_question_type(question)

    if target_domain in covered:
        return None, "schema_error: 下一问重复已覆盖认知域", question

    if target_domain not in COGNITIVE_DOMAINS:
        return None, "schema_error: 下一问认知域无效", question

    if not _is_self_contained_question(question, target_domain):
        return None, "schema_error: 下一问不是自包含问题", question

    if _uses_stale_fixed_question_material(question):
        return None, "schema_error: 下一问复用了过时固定题材", question

    if not _is_natural_elder_question(question, target_domain):
        return None, "schema_error: 下一问不够自然友好", question

    conversation_rejection = _conversation_quality_rejection_reason(question, messages)
    if conversation_rejection:
        return None, conversation_rejection, question

    payload_source = payload.get("source", "qwen")
    if payload_source not in (None, "", "qwen"):
        return None, "schema_error: 下一问 source 字段无效", question

    sample_answers = get_dialog_example_answers(
        question,
        target_domain=target_domain,
        sample_answers=payload.get("sample_answers"),
    )
    reason = str(payload.get("reason", "")).strip()
    if repair_reason:
        reason = f"质量修复重试：{repair_reason}" + (f"；{reason}" if reason else "")

    return (
        _next_question_result(
            question=question,
            target_domain=target_domain,
            source="qwen",
            model=config.llm_model,
            reason=reason,
            is_mock=False,
            completed=False,
            sample_answers=sample_answers,
        ),
        "",
        question,
    )


def _is_self_contained_question(question: str, target_domain: str) -> bool:
    bad_phrases = (
        "我念一串",
        "我稍后念",
        "等一下我",
        "稍后我会念",
        "我会念",
        "我们来玩",
        "先听我念",
    )
    if any(phrase in question for phrase in bad_phrases):
        return False

    unsuitable_keywords = (
        "正方形",
        "小正方形",
        "三角形",
        "几何",
        "图形",
        "画一个",
        "分成四个",
        "数学题",
        "证明",
    )
    if any(keyword in question for keyword in unsuitable_keywords):
        return False

    if target_domain == "attention":
        asks_digit_task = any(keyword in question for keyword in ("数字", "倒着", "倒背", "倒序"))
        if asks_digit_task and not any(character.isdigit() for character in question):
            return False

    if target_domain == "memory":
        asks_remember_task = any(keyword in question for keyword in ("请记住", "记住这", "稍后我会再问"))
        if asks_remember_task and not any(separator in question for separator in ("：", ":", "、", ",")):
            return False

    if target_domain == "visuospatial":
        life_keywords = (
            "左",
            "右",
            "位置",
            "旁边",
            "前面",
            "后面",
            "经过",
            "客厅",
            "厨房",
            "卧室",
            "钥匙",
            "水杯",
            "眼镜",
            "门口",
            "路线",
        )
        if not any(keyword in question for keyword in life_keywords):
            return False

    return True


def _uses_stale_fixed_question_material(question: str) -> bool:
    normalized = question.replace("，", "、").replace(",", "、")
    return any(
        all(material in normalized for material in materials)
        for materials in STALE_FIXED_QUESTION_MATERIALS
    )


def _is_natural_elder_question(question: str, target_domain: str) -> bool:
    if target_domain != "visuospatial":
        return True

    awkward_route_patterns = (
        ("从家", "左边还是右边"),
        ("家的左边", "公园"),
        ("家的右边", "公园"),
        ("是从", "左边还是右边"),
        ("左边还是右边", "走过去"),
        ("左边还是右边", "散步"),
    )
    return not any(
        all(part in question for part in pattern)
        for pattern in awkward_route_patterns
    )


def _conversation_quality_rejection_reason(question: str, messages: list[str]) -> str:
    clean_messages = [message.strip() for message in messages if message.strip()]
    if not clean_messages:
        return ""

    if "您好，我是小顾" in question:
        return "style_error: 下一问重复介绍小顾"

    if question.count("？") >= 2:
        return "style_error: 下一问包含多个核心问题"

    last_answer = _last_elder_answer(clean_messages)
    if _answer_looks_uncertain_or_vague(last_answer) and _uses_overconfident_praise(question):
        return "style_error: 含糊回答后不应使用过度肯定评价"

    previous_questions = _previous_ai_questions(clean_messages)
    for previous_question in previous_questions:
        if _reuses_previous_question(question, previous_question):
            return "style_error: 下一问拼接了历史问题原文"

    return ""


def _polish_next_question_style(question: str, messages: list[str]) -> str:
    clean_question = _normalize_elder_address_in_question(str(question or "").strip())
    if not clean_question:
        return clean_question

    clean_messages = [message.strip() for message in messages if message.strip()]
    previous_questions_text = " ".join(_previous_ai_questions(clean_messages))
    last_answer = _last_elder_answer(clean_messages)
    for opener in REPETITIVE_NEXT_QUESTION_OPENERS:
        if not clean_question.startswith(opener):
            continue
        should_replace = opener in previous_questions_text or _answer_looks_uncertain_or_vague(last_answer)
        if should_replace:
            replacement = _next_question_opener_for_context(clean_messages, last_answer)
            clean_question = _replace_leading_opener(clean_question, opener, replacement)
        break

    return clean_question


def _normalize_elder_address_in_question(question: str) -> str:
    clean_question = str(question or "").strip()
    if not clean_question:
        return clean_question
    placeholder = "__COGNIGUARD_YOUMEN__"
    return (
        clean_question.replace("你好", "您好")
        .replace("你们", placeholder)
        .replace("你", "您")
        .replace(placeholder, "你们")
    )


def _next_question_opener_for_context(messages: list[str], last_answer: str) -> str:
    choices = (
        UNCERTAIN_ANSWER_OPENERS
        if _answer_looks_uncertain_or_vague(last_answer)
        else NEUTRAL_ANSWER_OPENERS
    )
    index = len(_previous_ai_questions(messages)) % len(choices)
    return choices[index]


def _replace_leading_opener(question: str, opener: str, replacement: str) -> str:
    remainder = question[len(opener):].lstrip("，,。；;：: ")
    if not remainder:
        return replacement
    punctuation = "。" if not replacement.endswith(("。", "？", "！")) else ""
    return f"{replacement}{punctuation}{remainder}"


def _answer_looks_uncertain_or_vague(answer: str) -> bool:
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
        "记不清",
        "弄混",
        "混了",
        "分不清",
        "不会",
        "差不多",
        "看情况",
        "随便",
    )
    clean_answer = str(answer or "").strip()
    return any(keyword in clean_answer for keyword in vague_keywords)


def _uses_overconfident_praise(question: str) -> bool:
    praise_phrases = (
        "说得很清楚",
        "记得很清楚",
        "回答得很清楚",
        "数得很好",
        "很准确",
        "非常准确",
        "真好",
    )
    clean_question = str(question or "").strip()
    return any(phrase in clean_question for phrase in praise_phrases)


def _previous_ai_questions(messages: list[str]) -> list[str]:
    questions = []
    for message in messages:
        if "AI访谈问题：" in message:
            questions.append(message.split("AI访谈问题：", 1)[1].strip())
    return [question for question in questions if question]


def _reuses_previous_question(question: str, previous_question: str) -> bool:
    if len(previous_question) >= 12 and previous_question in question:
        return True

    fragments = [
        fragment.strip()
        for fragment in previous_question.replace("？", "?\n").splitlines()
        if len(fragment.strip()) >= 10
    ]
    return any(fragment in question for fragment in fragments)


def _next_question_result(
    *,
    question: str,
    target_domain: str,
    source: str,
    model: str,
    reason: str,
    is_mock: bool,
    completed: bool,
    sample_answers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    metadata = {
        "source": source,
        "model": model.strip() or "未配置",
    }
    if reason:
        metadata["reason"] = reason
    return {
        "question": question,
        "target_domain": target_domain,
        "is_mock": is_mock,
        "completed": completed,
        "sample_answers": sample_answers or {},
        "metadata": metadata,
        "disclaimer": DISCLAIMER,
    }


def _with_metadata(
    report: dict[str, Any],
    source: str,
    model: str = "",
    reason: str = "",
) -> dict[str, Any]:
    metadata = {
        "source": source,
        "model": model.strip() or "未配置",
    }
    if reason:
        metadata["reason"] = reason

    report["metadata"] = metadata
    return report
