from __future__ import annotations

import base64
from typing import Any, Optional

from core.config import AppConfig, load_config
from core.schemas import DISCLAIMER


ASR_TIMEOUT_SECONDS = 30.0


def transcribe_audio(
    audio: Optional[Any] = None,
    filename: Optional[str] = None,
    config: Optional[AppConfig] = None,
) -> dict[str, Any]:
    active_config = config or load_config()
    source_name = filename or getattr(audio, "name", None) or "not_saved"

    if active_config.demo_mode:
        return _mock_transcription(
            source_name,
            active_config,
            reason="DEMO_MODE=true",
        )

    if not _should_use_real_asr(active_config):
        return _mock_transcription(
            source_name,
            active_config,
            reason="ASR 配置不完整",
        )

    audio_bytes = _read_audio_bytes(audio)
    if not audio_bytes:
        return _fallback_transcription(
            source_name,
            active_config,
            reason="未提供音频，无法转写",
        )

    try:
        text = _request_transcription(audio_bytes, source_name, audio, active_config)
    except Exception as error:
        return _fallback_transcription(
            source_name,
            active_config,
            reason="api_error: ASR 调用失败，未生成可靠转写",
            error=_safe_error_text(error, active_config),
        )

    clean_text = text.strip()
    if not clean_text:
        return _fallback_transcription(
            source_name,
            active_config,
            reason="empty_result: ASR 未返回有效文本",
        )

    return _with_metadata(
        {
            "text": clean_text,
            "language": "zh",
            "confidence": None,
            "source_filename": source_name,
            "is_mock": False,
            "disclaimer": DISCLAIMER,
        },
        source="asr-api",
        model=active_config.asr_model,
    )


def _should_use_real_asr(config: AppConfig) -> bool:
    return (
        not config.demo_mode
        and bool(config.asr_base_url.strip())
        and bool(config.asr_api_key.strip())
        and bool(config.asr_model.strip())
    )


def _request_transcription(
    audio_bytes: bytes,
    filename: str,
    audio: Optional[Any],
    config: AppConfig,
) -> str:
    client = _create_openai_client(config)
    mime_type = _detect_audio_mime_type(audio_bytes, audio=audio, filename=filename)
    if _uses_qwen3_asr_flash(config.asr_model):
        return _request_qwen3_asr_flash(client, audio_bytes, mime_type, config)

    response = client.audio.transcriptions.create(
        model=config.asr_model,
        file=(filename or "answer.wav", audio_bytes, mime_type),
    )
    return _extract_transcription_text(response)


def _uses_qwen3_asr_flash(model: str) -> bool:
    return model.strip().lower() in {"qwen3-asr-flash", "qwen3-asr-flash-us"}


def _request_qwen3_asr_flash(
    client: Any,
    audio_bytes: bytes,
    mime_type: str,
    config: AppConfig,
) -> str:
    response = client.chat.completions.create(
        model=config.asr_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": _audio_to_data_url(audio_bytes, mime_type),
                        },
                    }
                ],
            }
        ],
        stream=False,
        extra_body={
            "asr_options": {
                "enable_itn": False,
            }
        },
    )
    return _extract_chat_message_text(response)


def _create_openai_client(config: AppConfig) -> Any:
    from openai import OpenAI

    return OpenAI(
        api_key=config.asr_api_key,
        base_url=config.asr_base_url,
        timeout=ASR_TIMEOUT_SECONDS,
    )


def _read_audio_bytes(audio: Optional[Any]) -> bytes:
    if audio is None:
        return b""
    if isinstance(audio, bytes):
        return audio
    if isinstance(audio, bytearray):
        return bytes(audio)
    if hasattr(audio, "getvalue"):
        value = audio.getvalue()
        return bytes(value) if isinstance(value, (bytes, bytearray)) else b""
    if hasattr(audio, "read"):
        value = audio.read()
        return bytes(value) if isinstance(value, (bytes, bytearray)) else b""
    return b""


def _detect_audio_mime_type(
    audio_bytes: bytes,
    audio: Optional[Any] = None,
    filename: Optional[str] = None,
) -> str:
    uploaded_type = getattr(audio, "type", None)
    if isinstance(uploaded_type, str) and uploaded_type.startswith("audio/"):
        return uploaded_type

    lower_name = (filename or getattr(audio, "name", "") or "").lower()
    extension_mimes = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".m4a": "audio/mp4",
        ".mp4": "audio/mp4",
        ".webm": "audio/webm",
        ".ogg": "audio/ogg",
    }
    for extension, mime_type in extension_mimes.items():
        if lower_name.endswith(extension):
            return mime_type

    if audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:16]:
        return "audio/wav"
    if audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb"):
        return "audio/mpeg"
    if audio_bytes.startswith(b"OggS"):
        return "audio/ogg"
    if audio_bytes.startswith(b"\x1aE\xdf\xa3"):
        return "audio/webm"
    return "audio/wav"


def _audio_to_data_url(audio_bytes: bytes, mime_type: str) -> str:
    encoded = base64.b64encode(audio_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_transcription_text(response: Any) -> str:
    if isinstance(response, dict):
        text = response.get("text", "")
        return text if isinstance(text, str) else str(text)

    text = getattr(response, "text", "")
    return text if isinstance(text, str) else str(text)


def _extract_chat_message_text(response: Any) -> str:
    content = response.choices[0].message.content
    return content if isinstance(content, str) else str(content)


def _mock_transcription(
    filename: str,
    config: AppConfig,
    reason: str,
) -> dict[str, Any]:
    return _with_metadata(
        {
            "text": "这是模拟 ASR 转写文本，可替换为老人实际回答。",
            "language": "zh",
            "confidence": None,
            "source_filename": filename,
            "is_mock": True,
            "disclaimer": DISCLAIMER,
        },
        source="mock",
        model=config.asr_model,
        reason=reason,
    )


def _fallback_transcription(
    filename: str,
    config: AppConfig,
    reason: str,
    error: str = "",
) -> dict[str, Any]:
    result = _with_metadata(
        {
            "text": "",
            "language": "zh",
            "confidence": None,
            "source_filename": filename,
            "is_mock": False,
            "disclaimer": DISCLAIMER,
        },
        source="fallback",
        model=config.asr_model,
        reason=reason,
    )
    if error:
        result["metadata"]["error"] = error
    return result


def _with_metadata(
    result: dict[str, Any],
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
    result["metadata"] = metadata
    return result


def _safe_error_text(error: Exception, config: AppConfig) -> str:
    message = str(error)
    api_key = config.asr_api_key.strip()
    if api_key:
        message = message.replace(api_key, "[redacted-api-key]")
    return message
