from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import ENV_PATH  # noqa: E402
from core.schemas import COGNITIVE_DOMAINS  # noqa: E402


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps help usable in bare envs.
    load_dotenv = None


DEFAULT_OUTPUT_PATH = PROJECT_ROOT / "data" / "eval" / "deepseek_dialog_cases.json"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
LABELS = ("normal", "mild_decline", "obvious_issue")
EXPECTED_RISK_BY_LABEL = {
    "normal": "low",
    "mild_decline": "medium",
    "obvious_issue": "high",
}


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    if load_dotenv is not None:
        load_dotenv(ENV_PATH, override=False)

    output_path = _resolve_project_path(args.output, DEFAULT_OUTPUT_PATH)
    if output_path.exists() and not args.force:
        print(
            f"Output already exists: {output_path}. "
            "Use --force to overwrite it after you have reviewed the old file."
        )
        return 2

    base_url = _env_or_default("DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    model = args.model or _env_or_default("DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
    _print_safe_config_status(base_url=base_url, api_key=api_key, model=model)
    if not api_key.strip():
        print(
            "DEEPSEEK_API_KEY is not configured. Add it to local .env and rerun; "
            "the key must not be committed."
        )
        return 2

    cases = generate_cases(
        labels=args.labels,
        per_label=args.per_label,
        batch_size=args.batch_size,
        base_url=base_url,
        api_key=api_key,
        model=model,
        seed=args.seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Generated {len(cases)} DeepSeek dialogue evaluation cases: {output_path}")
    return 0


def generate_cases(
    *,
    labels: list[str],
    per_label: int,
    batch_size: int,
    base_url: str,
    api_key: str,
    model: str,
    seed: int,
) -> list[dict[str, Any]]:
    all_cases: list[dict[str, Any]] = []
    for label in labels:
        remaining = per_label
        label_cases: list[dict[str, Any]] = []
        while remaining > 0:
            current_batch_size = min(batch_size, remaining)
            start_index = len(label_cases) + 1
            prompt = build_generation_prompt(
                label=label,
                count=current_batch_size,
                start_index=start_index,
                seed=seed,
            )
            raw_cases = _request_cases_from_deepseek(
                prompt=prompt,
                base_url=base_url,
                api_key=api_key,
                model=model,
            )
            label_cases.extend(
                validate_generated_cases(
                    raw_cases,
                    label=label,
                    start_index=start_index,
                )
            )
            remaining -= current_batch_size

        for index, case in enumerate(label_cases, start=1):
            case["case_id"] = f"ds_dialog_{label}_{index:03d}"
            all_cases.append(case)
    return all_cases


def build_generation_prompt(
    *,
    label: str,
    count: int,
    start_index: int = 1,
    seed: int = 20260607,
) -> str:
    expected_risk = EXPECTED_RISK_BY_LABEL[label]
    domains = ", ".join(COGNITIVE_DOMAINS)
    if label == "normal":
        label_rule = (
            "老人回答整体清楚稳定，时间/记忆/语言/执行/注意/视觉空间信号基本正常；"
            "expected_low_domains 必须为空数组。"
        )
    elif label == "mild_decline":
        label_rule = (
            "老人回答有轻度不稳定或迟疑，但不是完全混乱；至少 2 个认知域应进入 "
            "expected_low_domains，常见为 memory、attention、executive_function 或 visuospatial。"
        )
    else:
        label_rule = (
            "老人回答出现明显风险信号，例如日期地点混乱、延迟回忆失败、步骤安排困难、"
            "左右路线混乱或语言明显贫乏；至少 3 个认知域应进入 expected_low_domains。"
        )

    return f"""
你是 CogniGuard 技术原型的评测集生成器。请生成 {count} 条中文模拟老人访谈评测用例，
从编号 {start_index} 开始构思，随机种子提示为 {seed}。这些数据只用于本地 Demo 的
规则一致性评测，不是真实医疗数据，也不能包含真实身份信息。

统一要求：
- 只输出 JSON 对象，不要 Markdown，不要解释；对象必须只有一个 cases 字段，cases 是用例数组。
- 每条用例必须包含字段：
  case_id, label, is_mock, dialogue_turns, expected_risk_level, expected_low_domains, notes。
- label 必须固定为 "{label}"。
- is_mock 必须为 true。
- expected_risk_level 必须为 "{expected_risk}"。
- expected_low_domains 只能使用这些英文 key：{domains}。
- dialogue_turns 必须有 6 轮，每轮包含 assistant 和 user。
- assistant 必须是“小顾”风格的生活化问题，不要像考试，不要使用临床诊断措辞。
- user 必须是老人自然回答，不能全是模板句，不能每条都说同一句“我不知道”。
- 6 轮要尽量覆盖 orientation、memory、language、executive_function、attention、visuospatial。
- notes 必须包含“模拟数据，非临床数据”。

本批标签规则：
{label_rule}

请让不同用例的问题和回答有明显差异，覆盖噪声、含糊回答、部分正确、跑题、迟疑等情况，
但不要出现手机号、身份证、真实地址、真实病历、药物推荐或确诊语言。
""".strip()


def validate_generated_cases(
    raw_cases: Any,
    *,
    label: str,
    start_index: int = 1,
) -> list[dict[str, Any]]:
    if not isinstance(raw_cases, list):
        raise ValueError("DeepSeek response must be a JSON array")

    validated: list[dict[str, Any]] = []
    expected_risk = EXPECTED_RISK_BY_LABEL[label]
    for offset, raw_case in enumerate(raw_cases):
        if not isinstance(raw_case, dict):
            raise ValueError("Each generated case must be an object")
        raw_turns = raw_case.get("dialogue_turns")
        if not isinstance(raw_turns, list) or len(raw_turns) < 5:
            raise ValueError("Each generated case must contain at least 5 dialogue turns")

        turns: list[dict[str, str]] = []
        for turn in raw_turns[:6]:
            if not isinstance(turn, dict):
                raise ValueError("Each dialogue turn must be an object")
            assistant = str(turn.get("assistant", "")).strip()
            user = str(turn.get("user", "")).strip()
            if not assistant or not user:
                raise ValueError("Each dialogue turn must include assistant and user")
            turns.append({"assistant": assistant, "user": user})

        expected_low_domains = raw_case.get("expected_low_domains", [])
        if not isinstance(expected_low_domains, list):
            raise ValueError("expected_low_domains must be a list")
        normalized_domains = [
            str(domain).strip()
            for domain in expected_low_domains
            if str(domain).strip() in COGNITIVE_DOMAINS
        ]
        if label == "normal":
            normalized_domains = []
        elif not normalized_domains:
            raise ValueError(f"{label} case must include expected_low_domains")

        notes = str(raw_case.get("notes", "")).strip()
        if "模拟数据" not in notes or "非临床数据" not in notes:
            notes = (notes + "；" if notes else "") + "模拟数据，非临床数据。"

        case_index = start_index + offset
        validated.append(
            {
                "case_id": f"ds_dialog_{label}_{case_index:03d}",
                "label": label,
                "is_mock": True,
                "dialogue_turns": turns,
                "expected_risk_level": expected_risk,
                "expected_low_domains": normalized_domains,
                "notes": notes,
            }
        )
    return validated


def parse_json_array(content: str) -> list[Any]:
    text = content.strip()
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    if fenced_match:
        text = fenced_match.group(1).strip()
    payload = json.loads(text)
    if not isinstance(payload, list):
        raise ValueError("response JSON must be an array")
    return payload


def _request_cases_from_deepseek(
    *,
    prompt: str,
    base_url: str,
    api_key: str,
    model: str,
) -> list[Any]:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60.0, max_retries=1)
    last_error = ""
    active_prompt = prompt
    for attempt in range(1, 4):
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "你只输出可解析 JSON 对象，不输出 Markdown 或额外解释。",
                },
                {"role": "user", "content": active_prompt},
            ],
            "temperature": 0.7 if attempt == 1 else 0.35,
            "max_tokens": 5000,
        }
        if attempt == 1:
            request_kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content or ""
        try:
            payload = _parse_cases_payload(content)
        except Exception as error:
            last_error = f"{type(error).__name__}: {str(error)[:120]}"
            active_prompt = _repair_prompt(prompt)
            continue
        return payload

    raise ValueError(f"DeepSeek response was not parseable after retries: {last_error}")


def _parse_cases_payload(content: str) -> list[Any]:
    text = content.strip()
    if not text:
        raise ValueError("empty response content")
    fenced_match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.S)
    if fenced_match:
        text = fenced_match.group(1).strip()
    payload = json.loads(text)
    if isinstance(payload, dict) and isinstance(payload.get("cases"), list):
        return payload["cases"]
    if isinstance(payload, list):
        return payload
    raise ValueError("response must be a JSON array or an object with cases")


def _repair_prompt(original_prompt: str) -> str:
    return (
        original_prompt
        + "\n\n上一次输出无法解析。请重新输出，必须是严格 JSON 对象："
        + '{"cases":[...]}。不要 Markdown，不要代码块，不要解释，不要省略字段。'
    )


def _parse_args(argv: Optional[list[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate mock dialogue evaluation cases with DeepSeek. This script may "
            "call real APIs and consume tokens."
        )
    )
    parser.add_argument(
        "--per-label",
        type=int,
        default=30,
        help="Cases per label. Defaults to 30, for 90 balanced dialogue cases.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Cases per DeepSeek request. Defaults to 10.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        choices=LABELS,
        default=list(LABELS),
        help="Labels to generate. Defaults to all three labels.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_PATH.relative_to(PROJECT_ROOT)),
        help="Output JSON path inside the project. Defaults to data/eval/deepseek_dialog_cases.json.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="DeepSeek model. Defaults to DEEPSEEK_MODEL or deepseek-v4-flash.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260607,
        help="Seed hint included in the generation prompt.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    args = parser.parse_args(argv)
    if args.per_label < 1:
        parser.error("--per-label must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return args


def _resolve_project_path(value: str, default: Path) -> Path:
    if value is None or str(value).strip() == "":
        return default
    raw_path = Path(value)
    resolved = raw_path.resolve() if raw_path.is_absolute() else (PROJECT_ROOT / raw_path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"path must stay inside project: {value}") from error
    return resolved


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def _print_safe_config_status(*, base_url: str, api_key: str, model: str) -> None:
    print(f"DEEPSEEK_BASE_URL configured={bool(base_url.strip())}")
    print(f"DEEPSEEK_MODEL={model or '未配置'}")
    print(f"DEEPSEEK_API_KEY configured={bool(api_key.strip())}")


if __name__ == "__main__":
    raise SystemExit(main())
