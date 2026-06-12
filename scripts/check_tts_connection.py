from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import AppConfig, load_config  # noqa: E402
from core.tts_client import synthesize_speech  # noqa: E402


DEFAULT_TEXT = "请您把这串数字倒着说一遍：7-2-5。"


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    use_cache = not args.no_cache
    text = " ".join(args.text).strip() or DEFAULT_TEXT
    config = load_config()
    selected_model, selected_voice = _resolve_role_tts(config, args.role, args.model, args.voice)

    _print_config_status(config)
    print("TTS 为非实时合成，首次生成可能需要数秒到十几秒。")
    print("如果单条诊断成功但页面批量失败，优先尝试降低并发或重试失败项。")
    print(f"role={args.role}")
    print(f"final_model={selected_model or '未配置'}")
    print(f"final_voice={selected_voice or '未配置'}")

    result = synthesize_speech(
        text,
        model=selected_model,
        voice=selected_voice,
        config=config,
        use_cache=use_cache,
    )
    metadata = result.get("metadata", {})
    audio_bytes = result.get("audio_bytes")
    audio_size = len(audio_bytes) if isinstance(audio_bytes, (bytes, bytearray)) else 0

    print(f"source={metadata.get('source', 'unknown')}")
    print(f"model={metadata.get('model', selected_model or '未配置')}")
    print(f"voice={metadata.get('voice', selected_voice or '未配置')}")
    print(f"cached={str(bool(metadata.get('cached'))).lower()}")
    print(f"audio_size={audio_size}")
    if metadata.get("cache_path"):
        print(f"cache_path={metadata.get('cache_path')}")

    if audio_size:
        if use_cache:
            output_path = _save_check_audio(
                bytes(audio_bytes),
                config,
                str(result.get("mime_type", "")),
            )
            print(f"saved_audio={output_path}")
        else:
            print("saved_audio=disabled_by_no_cache")
        return 0

    print("No real audio was generated.")
    print(f"reason={metadata.get('reason', '')}")
    return 2 if metadata.get("source") == "mock" else 1


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check CogniGuard TTS connection without printing API keys.",
    )
    parser.add_argument(
        "--role",
        choices=["assistant", "patient"],
        default="assistant",
        help="Use assistant system voice or classroom patient demo voice.",
    )
    parser.add_argument("--model", default="", help="Override TTS model for this check.")
    parser.add_argument("--voice", default="", help="Override TTS voice for this check.")
    parser.add_argument("--no-cache", action="store_true", help="Bypass local TTS cache.")
    parser.add_argument("text", nargs="*", help="Text to synthesize.")
    return parser.parse_args(argv)


def _resolve_role_tts(
    config: AppConfig,
    role: str,
    model_override: str = "",
    voice_override: str = "",
) -> tuple[str, str]:
    if role == "patient":
        default_model = config.tts_model_patient_demo or "cosyvoice-v3-flash"
        default_voice = config.tts_voice_patient_demo or "longlaoyi_v3"
    else:
        default_model = config.tts_model_assistant or config.tts_model or "qwen-tts"
        default_voice = config.tts_voice_assistant or "Cherry"
    model = model_override.strip() or default_model.strip()
    voice = voice_override.strip() or default_voice.strip()
    return model, voice


def _print_config_status(config: AppConfig) -> None:
    print(f"DEMO_MODE={str(config.demo_mode).lower()}")
    print(f"VOICE_DEMO_MODE={str(config.voice_demo_mode).lower()}")
    print(f"QWEN_BASE_URL configured={bool(config.qwen_base_url.strip())}")
    print(f"QWEN_API_KEY configured={bool(config.qwen_api_key.strip())}")
    print(f"TTS_BASE_URL configured={bool(config.tts_base_url.strip())}")
    print(f"TTS_API_KEY configured={bool(config.tts_api_key.strip())}")
    print(f"TTS_MODEL={config.tts_model or '未配置'}")
    print(f"TTS_MODEL_ASSISTANT={config.tts_model_assistant or '未配置'}")
    print(f"TTS_MODEL_PATIENT_DEMO={config.tts_model_patient_demo or '未配置'}")
    print(f"TTS_VOICE_ASSISTANT={config.tts_voice_assistant or '未配置'}")
    print(f"TTS_VOICE_PATIENT_DEMO={config.tts_voice_patient_demo or '未配置'}")


def _save_check_audio(audio_bytes: bytes, config: AppConfig, mime_type: str = "") -> Path:
    output_dir = PROJECT_ROOT / "data" / "voice_cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = _audio_suffix(mime_type or config.tts_format)
    output_path = output_dir / f"tts_check{suffix}"
    output_path.write_bytes(audio_bytes)
    return output_path


def _audio_suffix(audio_format: str) -> str:
    normalized = audio_format.strip().lower().lstrip(".")
    if normalized in {"audio/wav", "audio/x-wav"}:
        return ".wav"
    if normalized in {"audio/ogg"}:
        return ".ogg"
    if normalized in {"audio/mp4", "audio/aac"}:
        return ".m4a"
    if normalized in {"wav", "ogg", "m4a", "mp4"}:
        return f".{normalized}"
    return ".mp3"


if __name__ == "__main__":
    raise SystemExit(main())
