from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps config importable in bare envs.
    load_dotenv = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


@dataclass(frozen=True)
class AppConfig:
    qwen_base_url: str = ""
    qwen_api_key: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""
    vlm_base_url: str = ""
    vlm_api_key: str = ""
    vlm_model: str = ""
    embed_base_url: str = ""
    embed_api_key: str = ""
    embed_model: str = ""
    asr_base_url: str = ""
    asr_api_key: str = ""
    asr_model: str = ""
    tts_base_url: str = ""
    tts_api_key: str = ""
    tts_model: str = "qwen-tts"
    tts_model_assistant: str = "qwen-tts"
    tts_model_patient_demo: str = "cosyvoice-v3-flash"
    tts_voice_assistant: str = "Cherry"
    tts_voice_patient_demo: str = "longlaoyi_v3"
    tts_format: str = "mp3"
    demo_mode: bool = True
    voice_demo_mode: bool = False
    use_cache: bool = True
    staff_password: str = "8888"


def is_llm_config_complete(config: AppConfig) -> bool:
    return all(
        [
            config.llm_base_url.strip(),
            config.llm_api_key.strip(),
            config.llm_model.strip(),
        ]
    )


def is_vlm_config_complete(config: AppConfig) -> bool:
    return all(
        [
            config.vlm_base_url.strip(),
            config.vlm_api_key.strip(),
            config.vlm_model.strip(),
        ]
    )


def is_asr_config_complete(config: AppConfig) -> bool:
    return all(
        [
            config.asr_base_url.strip(),
            config.asr_api_key.strip(),
            config.asr_model.strip(),
        ]
    )


def is_tts_config_complete(config: AppConfig) -> bool:
    return all(
        [
            config.tts_base_url.strip(),
            config.tts_api_key.strip(),
            config.tts_model.strip(),
            config.tts_voice_assistant.strip(),
        ]
    )


def build_runtime_status(config: AppConfig) -> dict[str, str]:
    api_ready = not config.demo_mode and (
        is_llm_config_complete(config) or is_vlm_config_complete(config)
    )
    return {
        "运行模式": "真实 API" if api_ready else "Demo Mock",
        "LLM 模型": config.llm_model.strip() or "未配置",
        "VLM 模型": config.vlm_model.strip() or "未配置",
    }


def load_config() -> AppConfig:
    if load_dotenv is not None:
        load_dotenv(ENV_PATH, override=False)

    qwen_base_url = os.getenv("QWEN_BASE_URL", "")
    qwen_api_key = os.getenv("QWEN_API_KEY", "")

    tts_model = _env_or_default("TTS_MODEL", "qwen-tts")

    return AppConfig(
        qwen_base_url=qwen_base_url,
        qwen_api_key=qwen_api_key,
        llm_base_url=_env_or_fallback("LLM_BASE_URL", qwen_base_url),
        llm_api_key=_env_or_fallback("LLM_API_KEY", qwen_api_key),
        llm_model=os.getenv("LLM_MODEL", ""),
        vlm_base_url=_env_or_fallback("VLM_BASE_URL", qwen_base_url),
        vlm_api_key=_env_or_fallback("VLM_API_KEY", qwen_api_key),
        vlm_model=os.getenv("VLM_MODEL", ""),
        embed_base_url=os.getenv("EMBED_BASE_URL", ""),
        embed_api_key=os.getenv("EMBED_API_KEY", ""),
        embed_model=os.getenv("EMBED_MODEL", ""),
        asr_base_url=_env_or_fallback("ASR_BASE_URL", qwen_base_url),
        asr_api_key=_env_or_fallback("ASR_API_KEY", qwen_api_key),
        asr_model=os.getenv("ASR_MODEL", ""),
        tts_base_url=_env_or_fallback("TTS_BASE_URL", qwen_base_url),
        tts_api_key=_env_or_fallback("TTS_API_KEY", qwen_api_key),
        tts_model=tts_model,
        tts_model_assistant=_env_or_default("TTS_MODEL_ASSISTANT", tts_model),
        tts_model_patient_demo=_env_or_default("TTS_MODEL_PATIENT_DEMO", "cosyvoice-v3-flash"),
        tts_voice_assistant=_env_or_default("TTS_VOICE_ASSISTANT", "Cherry"),
        tts_voice_patient_demo=_env_or_default("TTS_VOICE_PATIENT_DEMO", "longlaoyi_v3"),
        tts_format=_env_or_default("TTS_FORMAT", "mp3"),
        demo_mode=_env_bool("DEMO_MODE", default=True),
        voice_demo_mode=_env_bool("VOICE_DEMO_MODE", default=False),
        use_cache=_env_bool("USE_CACHE", default=True),
        staff_password=_env_or_default("STAFF_PASSWORD", "8888"),
    )


def _env_or_fallback(name: str, fallback: str) -> str:
    value = os.getenv(name, "")
    return value if value.strip() else fallback


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None or raw_value.strip() == "":
        return default

    return raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}
