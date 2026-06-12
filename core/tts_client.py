from __future__ import annotations

import base64
import hashlib
import json
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

from core.config import AppConfig, PROJECT_ROOT, load_config


TTS_TIMEOUT_SECONDS = 30.0
DEFAULT_TTS_MODEL = "qwen-tts"
DEFAULT_TTS_VOICE = "Cherry"
DEFAULT_TTS_CACHE_DIR = PROJECT_ROOT / "data" / "voice_cache"
# 项目内预置演示语音目录（提交进仓库）：DEMO_MODE/VOICE_DEMO_MODE 下命中即返回真实音频，
# 让默认演示在零运行时 API 的情况下也能出声。由 scripts/generate_demo_voice.py 离线生成。
DEMO_VOICE_DIR = PROJECT_ROOT / "assets" / "demo_voice"
_DIGIT_SEQUENCE_SEPARATOR_PATTERN = re.compile(r"\s*[-‐‑‒–—―−－]\s*")
_DIGIT_SEQUENCE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_])\d+(?:\s*[-‐‑‒–—―−－]\s*\d+)+(?![A-Za-z0-9_])"
)


def prepare_text_for_tts(text: str) -> str:
    clean_text = str(text or "").strip()
    if not clean_text:
        return ""
    return _DIGIT_SEQUENCE_PATTERN.sub(_separate_digit_sequence_for_tts, clean_text)


def _separate_digit_sequence_for_tts(match: re.Match[str]) -> str:
    return _DIGIT_SEQUENCE_SEPARATOR_PATTERN.sub("，", match.group(0))


def synthesize_speech(
    text: str,
    voice: Optional[str] = None,
    model: Optional[str] = None,
    config: Optional[AppConfig] = None,
    use_cache: bool = True,
    prefer_remote_url: bool = False,
) -> dict[str, Any]:
    active_config = config or load_config()
    clean_text = prepare_text_for_tts(text)
    selected_model = (
        model
        or active_config.tts_model_assistant
        or active_config.tts_model
        or DEFAULT_TTS_MODEL
    ).strip()
    selected_model = selected_model or DEFAULT_TTS_MODEL
    selected_voice = (voice or active_config.tts_voice_assistant or DEFAULT_TTS_VOICE).strip()
    selected_voice = selected_voice or DEFAULT_TTS_VOICE

    if not clean_text:
        return _fallback_result(
            active_config,
            selected_model,
            selected_voice,
            reason="empty_text: 未提供可合成的文本",
        )

    if active_config.demo_mode:
        bundled = _load_demo_voice(clean_text, active_config, selected_voice, selected_model)
        if bundled is not None:
            return bundled
        return _mock_result(
            active_config,
            selected_model,
            selected_voice,
            reason="DEMO_MODE=true",
        )

    if active_config.voice_demo_mode:
        bundled = _load_demo_voice(clean_text, active_config, selected_voice, selected_model)
        if bundled is not None:
            return bundled
        return _mock_result(
            active_config,
            selected_model,
            selected_voice,
            reason="VOICE_DEMO_MODE=true",
        )

    if not _should_use_real_tts(active_config, selected_model):
        return _mock_result(
            active_config,
            selected_model,
            selected_voice,
            reason="TTS 配置不完整",
        )

    cache_path = _cache_path_for(
        clean_text,
        active_config,
        selected_voice,
        model=selected_model,
    )
    if use_cache:
        cached_audio = _read_cached_audio(cache_path)
        if cached_audio:
            return _result(
                audio_bytes=cached_audio,
                mime_type=_mime_type_for_format(active_config.tts_format),
                source="tts_cache",
                model=selected_model,
                voice=selected_voice,
                cached=True,
                cache_path=_safe_cache_path(cache_path),
            )

    try:
        response = _request_speech(
            clean_text,
            selected_voice,
            selected_model,
            active_config,
        )
        if prefer_remote_url:
            audio_url = _find_audio_url(response)
            if audio_url:
                return _result(
                    audio_bytes=None,
                    audio_url=audio_url,
                    mime_type=_mime_type_from_url(audio_url),
                    source="tts_url",
                    model=selected_model,
                    voice=selected_voice,
                    cached=False,
                )
        audio_bytes, mime_type = _extract_audio(response, active_config)
    except Exception as error:
        return _fallback_result(
            active_config,
            selected_model,
            selected_voice,
            reason=_api_failure_reason(selected_model, selected_voice),
            error=_safe_error_text(error, active_config),
        )

    if not audio_bytes:
        return _fallback_result(
            active_config,
            selected_model,
            selected_voice,
            reason=_empty_result_reason(selected_model, selected_voice),
        )

    safe_cache_path = ""
    cache_write_reason = ""
    if use_cache:
        try:
            _write_cached_audio(cache_path, audio_bytes)
            safe_cache_path = _safe_cache_path(cache_path)
        except OSError:
            cache_write_reason = "cache_write_failed: TTS 音频已生成，但缓存写入失败"

    return _result(
        audio_bytes=audio_bytes,
        mime_type=mime_type or _mime_type_for_format(active_config.tts_format),
        source="tts",
        model=selected_model,
        voice=selected_voice,
        reason=cache_write_reason,
        cached=False,
        cache_path=safe_cache_path,
    )


def _should_use_real_tts(config: AppConfig, model: str) -> bool:
    return (
        not config.demo_mode
        and not config.voice_demo_mode
        and bool(config.tts_base_url.strip())
        and bool(config.tts_api_key.strip())
        and bool(model.strip())
    )


def _request_speech(text: str, voice: str, model: str, config: AppConfig) -> dict[str, Any]:
    if _is_cosyvoice_model(model):
        return _request_cosyvoice_speech(text, voice, model, config)
    return _request_qwen_tts_speech(text, voice, model, config)


def _request_qwen_tts_speech(text: str, voice: str, model: str, config: AppConfig) -> dict[str, Any]:
    endpoint = _tts_generation_url(config.tts_base_url)
    if not endpoint:
        raise ValueError("TTS endpoint is not configured")

    payload = {
        "model": model,
        "input": {
            "text": text,
            "voice": voice,
            "language_type": "Chinese",
        },
    }
    response = _post_json(endpoint, payload, config)
    status_code = response.get("status_code")
    if status_code not in (None, 200):
        code = str(response.get("code", "")).strip()
        message = str(response.get("message", "")).strip()
        raise RuntimeError(f"TTS API returned status_code={status_code}, code={code}, message={message}")
    return response


def _request_cosyvoice_speech(text: str, voice: str, model: str, config: AppConfig) -> dict[str, Any]:
    endpoint = _cosyvoice_generation_url(config.tts_base_url)
    if not endpoint:
        raise ValueError("TTS endpoint is not configured")

    payload = {
        "model": model,
        "input": {
            "text": text,
            "voice": voice,
            "format": _normalized_audio_format(config.tts_format),
        },
    }
    response = _post_json(endpoint, payload, config)
    status_code = response.get("status_code")
    if status_code not in (None, 200):
        code = str(response.get("code", "")).strip()
        message = str(response.get("message", "")).strip()
        raise RuntimeError(
            f"CosyVoice API returned status_code={status_code}, code={code}, message={message}"
        )
    return response


def _voice_filename(
    text: str,
    config: AppConfig,
    voice: str,
    model: Optional[str] = None,
) -> str:
    selected_model = (model or config.tts_model or DEFAULT_TTS_MODEL).strip()
    payload = {
        "text": text,
        "model": selected_model or DEFAULT_TTS_MODEL,
        "voice": voice.strip(),
        "format": _normalized_audio_format(config.tts_format),
    }
    digest = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return f"tts_{digest}.{payload['format']}"


def _cache_path_for(
    text: str,
    config: AppConfig,
    voice: str,
    model: Optional[str] = None,
) -> Path:
    return DEFAULT_TTS_CACHE_DIR / _voice_filename(text, config, voice, model)


def demo_voice_path_for(
    text: str,
    config: AppConfig,
    voice: str,
    model: Optional[str] = None,
) -> Path:
    """项目内预置演示语音的路径（与缓存同样的 text+model+voice+format 摘要）。"""
    return DEMO_VOICE_DIR / _voice_filename(text, config, voice, model)


def _load_demo_voice(
    text: str,
    config: AppConfig,
    voice: str,
    model: str,
) -> Optional[dict[str, Any]]:
    demo_path = demo_voice_path_for(text, config, voice, model)
    audio_bytes = _read_cached_audio(demo_path)
    if not audio_bytes:
        return None
    return _result(
        audio_bytes=audio_bytes,
        # 文件名摘要按 mp3 命名（保证查找命中），但实际字节可能是 WAV（qwen-tts 常返回
        # RIFF/WAVE）。按真实内容返回 mime，避免严格浏览器（Safari/iOS）或 data-URL 拒播。
        mime_type=_sniff_audio_mime(audio_bytes, config.tts_format),
        source="static_audio",
        model=model,
        voice=voice,
        reason="demo_voice: 项目内预置演示语音",
        cached=True,
        cache_path=_safe_cache_path(demo_path),
    )


def _sniff_audio_mime(audio_bytes: bytes, fallback_format: str) -> str:
    head = bytes(audio_bytes[:12])
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio/wav"
    if head[:4] == b"OggS":
        return "audio/ogg"
    if head[:3] == b"ID3" or head[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "audio/mpeg"
    if head[:4] in (b"ftyp", b"\x00\x00\x00\x18", b"\x00\x00\x00\x20") or head[4:8] == b"ftyp":
        return "audio/mp4"
    return _mime_type_for_format(fallback_format)


def _read_cached_audio(cache_path: Path) -> Optional[bytes]:
    try:
        audio_bytes = cache_path.read_bytes()
    except OSError:
        return None
    return audio_bytes or None


def _write_cached_audio(cache_path: Path, audio_bytes: bytes) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(audio_bytes)


def _safe_cache_path(cache_path: Path) -> str:
    try:
        return cache_path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return cache_path.name


def _tts_generation_url(base_url: str) -> str:
    clean_base = base_url.strip().rstrip("/")
    if not clean_base:
        return ""
    if clean_base.endswith("/services/aigc/multimodal-generation/generation"):
        return clean_base
    if clean_base.endswith("/compatible-mode/v1"):
        return clean_base[: -len("/compatible-mode/v1")] + "/api/v1/services/aigc/multimodal-generation/generation"
    if clean_base.endswith("/api/v1"):
        return clean_base + "/services/aigc/multimodal-generation/generation"
    return clean_base + "/services/aigc/multimodal-generation/generation"


def _cosyvoice_generation_url(base_url: str) -> str:
    clean_base = base_url.strip().rstrip("/")
    if not clean_base:
        return ""
    if clean_base.endswith("/services/audio/tts/SpeechSynthesizer"):
        return clean_base
    if clean_base.endswith("/compatible-mode/v1"):
        return clean_base[: -len("/compatible-mode/v1")] + "/api/v1/services/audio/tts/SpeechSynthesizer"
    if clean_base.endswith("/api/v1"):
        return clean_base + "/services/audio/tts/SpeechSynthesizer"
    return clean_base + "/services/audio/tts/SpeechSynthesizer"


def _post_json(url: str, payload: dict[str, Any], config: AppConfig) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.tts_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=TTS_TIMEOUT_SECONDS) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {error.code}: {body}") from error

    parsed = json.loads(body)
    if not isinstance(parsed, dict):
        raise ValueError("TTS response JSON must be an object")
    return parsed


def _extract_audio(response: dict[str, Any], config: AppConfig) -> tuple[Optional[bytes], str]:
    base64_audio = _find_audio_data(response)
    if base64_audio:
        return _decode_base64_audio(base64_audio), _mime_type_for_format(config.tts_format)

    audio_url = _find_audio_url(response)
    if audio_url:
        return _download_audio_url(audio_url)

    return None, _mime_type_for_format(config.tts_format)


def _find_audio_data(value: Any) -> str:
    for item in _walk_values(value):
        if not isinstance(item, dict):
            continue
        for key in ("data", "audio_data", "audio", "base64"):
            candidate = item.get(key)
            if isinstance(candidate, str) and _looks_like_base64(candidate):
                return candidate.strip()
    return ""


def _find_audio_url(value: Any) -> str:
    for item in _walk_values(value):
        if not isinstance(item, dict):
            continue
        for key in ("url", "audio_url", "audioUrl"):
            candidate = item.get(key)
            if isinstance(candidate, str) and candidate.strip().lower().startswith(("http://", "https://")):
                return candidate.strip()
    return ""


def _walk_values(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from _walk_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_values(child)


def _looks_like_base64(value: str) -> bool:
    candidate = value.strip()
    if not candidate:
        return False
    if candidate.startswith("data:") and ";base64," in candidate:
        candidate = candidate.split(";base64,", 1)[1]
    try:
        base64.b64decode(candidate, validate=True)
    except Exception:
        return False
    return True


def _decode_base64_audio(value: str) -> bytes:
    candidate = value.strip()
    if candidate.startswith("data:") and ";base64," in candidate:
        candidate = candidate.split(";base64,", 1)[1]
    return base64.b64decode(candidate)


def _download_audio_url(url: str) -> tuple[bytes, str]:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "CogniGuard-TTS-Check/1.0"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=TTS_TIMEOUT_SECONDS) as response:
        audio_bytes = response.read()
        headers = getattr(response, "headers", None)
        if hasattr(headers, "get_content_type"):
            mime_type = headers.get_content_type()
        else:
            mime_type = str(headers.get("Content-Type", "") if headers else "").split(";", 1)[0]
    return audio_bytes, mime_type or _mime_type_from_url(url)


def _mime_type_for_format(audio_format: str) -> str:
    normalized = _normalized_audio_format(audio_format)
    if normalized in {"mp3", "mpeg"}:
        return "audio/mpeg"
    if normalized == "wav":
        return "audio/wav"
    if normalized == "ogg":
        return "audio/ogg"
    if normalized in {"m4a", "mp4", "aac"}:
        return "audio/mp4"
    return "audio/mpeg"


def _normalized_audio_format(audio_format: str) -> str:
    normalized = audio_format.strip().lower().lstrip(".")
    if normalized in {"wav", "ogg", "m4a", "mp4", "aac"}:
        return normalized
    return "mp3"


def _mime_type_from_url(url: str) -> str:
    path = urlparse(url).path.lower()
    if path.endswith(".wav"):
        return "audio/wav"
    if path.endswith(".ogg"):
        return "audio/ogg"
    if path.endswith((".m4a", ".mp4", ".aac")):
        return "audio/mp4"
    return "audio/mpeg"


def _is_cosyvoice_model(model: str) -> bool:
    return "cosyvoice" in model.strip().lower()


def _needs_cosyvoice_hint(model: str, voice: str) -> bool:
    normalized_voice = voice.strip().lower()
    return _is_cosyvoice_model(model) or normalized_voice.endswith("_v3") or normalized_voice.startswith("longlaoyi")


def _api_failure_reason(model: str, voice: str) -> str:
    if _needs_cosyvoice_hint(model, voice):
        return (
            "api_error: TTS 调用失败，未生成真实音频；"
            f"voice {voice} may require CosyVoice model/API or account access"
        )
    return "api_error: TTS 调用失败，未生成真实音频"


def _empty_result_reason(model: str, voice: str) -> str:
    if _needs_cosyvoice_hint(model, voice):
        return (
            "empty_result: TTS 未返回有效音频；"
            f"voice {voice} may require CosyVoice model/API or account access"
        )
    return "empty_result: TTS 未返回有效音频"


def _mock_result(config: AppConfig, model: str, voice: str, reason: str) -> dict[str, Any]:
    return _result(
        audio_bytes=None,
        mime_type=_mime_type_for_format(config.tts_format),
        source="mock",
        model=model,
        voice=voice,
        reason=reason,
    )


def _fallback_result(
    config: AppConfig,
    model: str,
    voice: str,
    reason: str,
    error: str = "",
) -> dict[str, Any]:
    result = _result(
        audio_bytes=None,
        mime_type=_mime_type_for_format(config.tts_format),
        source="fallback",
        model=model,
        voice=voice,
        reason=reason,
    )
    if error:
        result["metadata"]["error"] = error
    return result


def _result(
    *,
    audio_bytes: Optional[bytes],
    mime_type: str,
    source: str,
    model: str,
    voice: str,
    reason: str = "",
    cached: bool = False,
    cache_path: str = "",
    audio_url: str = "",
) -> dict[str, Any]:
    metadata = {
        "source": source,
        "model": model.strip() or "未配置",
        "voice": voice.strip() or DEFAULT_TTS_VOICE,
        "reason": reason,
        "cached": cached,
    }
    if cache_path:
        metadata["cache_path"] = cache_path
    result = {
        "audio_bytes": audio_bytes,
        "mime_type": mime_type,
        "metadata": metadata,
    }
    if audio_url:
        result["audio_url"] = audio_url
    return result


def _safe_error_text(error: Exception, config: AppConfig) -> str:
    message = str(error)
    api_key = config.tts_api_key.strip()
    if api_key:
        message = message.replace(api_key, "[redacted-api-key]")
    return message
