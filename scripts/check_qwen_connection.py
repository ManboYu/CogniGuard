from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import AppConfig, load_config  # noqa: E402


TEMPERATURE = 0
MAX_TOKENS = 50
TIMEOUT_SECONDS = 20.0


def main() -> int:
    config = load_config()
    _print_config_status(config)

    if not _llm_config_complete(config):
        print("Qwen API connection skipped: LLM config is incomplete.")
        return 2

    try:
        content = _call_qwen(config)
    except Exception as error:
        print("Qwen API connection failed")
        print(f"Exception type: {type(error).__name__}")
        print(f"Safe error message: {_safe_error_message(error, config)}")
        return 1

    print("Qwen API connection succeeded")
    print("Model response:")
    print(content)
    return 0


def _print_config_status(config: AppConfig) -> None:
    print(f"DEMO_MODE={str(config.demo_mode).lower()}")
    print(f"LLM_BASE_URL configured={bool(config.llm_base_url.strip())}")
    print(f"LLM_MODEL={config.llm_model or '未配置'}")
    print(f"LLM_API_KEY configured={bool(config.llm_api_key.strip())}")


def _llm_config_complete(config: AppConfig) -> bool:
    return all(
        [
            config.llm_base_url.strip(),
            config.llm_api_key.strip(),
            config.llm_model.strip(),
        ]
    )


def _call_qwen(config: AppConfig) -> str:
    from openai import OpenAI

    client = OpenAI(
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        timeout=TIMEOUT_SECONDS,
    )
    response = client.chat.completions.create(
        model=config.llm_model,
        messages=_build_check_messages(),
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={"type": "json_object"},
    )
    return _extract_message_content(response)


def _build_check_messages() -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                'Return a JSON object exactly like {"ok": true}. '
                "Do not output Markdown or extra text."
            ),
        },
        {
            "role": "user",
            "content": 'Return JSON now: {"ok": true}',
        },
    ]


def _extract_message_content(response: Any) -> str:
    content = response.choices[0].message.content
    return content.strip() if isinstance(content, str) else str(content)


def _safe_error_message(error: Exception, config: AppConfig) -> str:
    message = str(error)
    api_key = config.llm_api_key.strip()
    if api_key:
        message = message.replace(api_key, "[redacted-api-key]")
    return message


if __name__ == "__main__":
    raise SystemExit(main())
