from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.asr_client import transcribe_audio  # noqa: E402
from core.config import AppConfig, load_config  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    config = load_config()
    _print_config_status(config)

    if not args:
        print("Usage: python scripts\\check_asr_connection.py path\\to\\answer.wav")
        return 2

    audio_path = Path(args[0])
    if not audio_path.exists() or not audio_path.is_file():
        print(f"Audio file not found: {audio_path}")
        print("Please replace the path with a real local wav/mp3 audio file before checking ASR.")
        print("This means the input file path is invalid; it is not an ASR configuration error.")
        return 2

    audio_bytes = audio_path.read_bytes()
    result = transcribe_audio(
        audio=audio_bytes,
        filename=audio_path.name,
        config=config,
    )
    metadata = result.get("metadata", {})
    source = metadata.get("source", "unknown")

    if source == "asr-api":
        print("ASR API connection succeeded")
        print(f"text={result.get('text', '')}")
        return 0

    if source == "mock":
        print("ASR API connection returned mock result.")
        print("Please check DEMO_MODE and ASR_MODEL, plus QWEN or dedicated ASR config.")
        print(f"reason={metadata.get('reason', '')}")
        return 2

    print("ASR API connection failed")
    print(f"reason={metadata.get('reason', '')}")
    if metadata.get("error"):
        print(f"safe_error={metadata.get('error')}")
    return 1


def _print_config_status(config: AppConfig) -> None:
    print(f"DEMO_MODE={str(config.demo_mode).lower()}")
    print(f"QWEN_BASE_URL configured={bool(config.qwen_base_url.strip())}")
    print(f"QWEN_API_KEY configured={bool(config.qwen_api_key.strip())}")
    print(f"ASR_BASE_URL configured={bool(config.asr_base_url.strip())}")
    print(f"ASR_MODEL={config.asr_model or '未配置'}")
    print(f"ASR_API_KEY configured={bool(config.asr_api_key.strip())}")


if __name__ == "__main__":
    raise SystemExit(main())
