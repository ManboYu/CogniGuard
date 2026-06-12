"""离线生成「项目内预置演示语音」，让默认演示（DEMO_MODE=true）也能出声。

用途
----
DEMO_MODE/VOICE_DEMO_MODE 下，core/tts_client.synthesize_speech 会先查
assets/demo_voice/ 里有没有对应文本的预置音频，命中就直接返回真实音频，
因此部署后无需任何运行时 API 也能听到问题朗读。

本脚本用真实 Qwen-TTS（系统音色）和 CosyVoice（模拟老人音色）一次性把
固定演示文案合成成 mp3，按 text+model+voice+format 摘要命名写入
assets/demo_voice/，再提交进仓库即可。

前置条件
--------
- 本地 .env 配好 QWEN_API_KEY / QWEN_BASE_URL（或 TTS_BASE_URL / TTS_API_KEY）。
- 会真实调用 API、消耗 token；不要提交 .env 或 API Key。
- 系统音色（qwen-tts / Cherry）一般都能成功；老人音色（cosyvoice-v3-flash /
  longlaoyi_v3）需要账号支持，失败会跳过并提示，系统语音仍可用。

用法
----
    python scripts/generate_demo_voice.py            # 生成全部
    python scripts/generate_demo_voice.py --assistant-only   # 只生成系统问题语音
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Windows 控制台默认 GBK，预览里的中文/特殊字符会编码报错；统一切到 UTF-8。
for _stream_name in ("stdout", "stderr"):
    _stream = getattr(sys, _stream_name, None)
    _reconfigure = getattr(_stream, "reconfigure", None)
    if callable(_reconfigure):
        try:
            _reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass

from core.config import AppConfig, load_config  # noqa: E402
from core.mock_data import (  # noqa: E402
    CLASSROOM_DEMO_LEVEL_LABELS,
    PRESET_INTERVIEW_QUESTIONS,
    build_classroom_demo_interview,
)
from core.tts_client import (  # noqa: E402
    DEMO_VOICE_DIR,
    demo_voice_path_for,
    prepare_text_for_tts,
    synthesize_speech,
)


def _collect_targets(assistant_only: bool) -> list[tuple[str, str, str, str]]:
    """返回去重后的 (text, model, voice, source_label) 列表，使用 deploy 默认音色。"""
    defaults = AppConfig()
    assistant_model = defaults.tts_model_assistant
    assistant_voice = defaults.tts_voice_assistant
    patient_model = defaults.tts_model_patient_demo
    patient_voice = defaults.tts_voice_patient_demo

    targets: list[tuple[str, str, str, str]] = []

    # 长者简易版 / 快捷访谈评估页的系统问题（预设访谈题）。
    for item in PRESET_INTERVIEW_QUESTIONS:
        targets.append((str(item["question"]), assistant_model, assistant_voice, "system"))

    # 演示模式页三种认知水平的系统问题 / 模拟老人回答 / 画钟提示。
    for level in CLASSROOM_DEMO_LEVEL_LABELS:
        for turn in build_classroom_demo_interview(level):
            targets.append((str(turn.get("system_question", "")), assistant_model, assistant_voice, "system"))
            if not assistant_only:
                targets.append((str(turn.get("patient_answer", "")), patient_model, patient_voice, "patient"))
            clock_prompt = str(turn.get("clock_trigger_elder_message", "")).strip()
            if clock_prompt:
                targets.append((clock_prompt, assistant_model, assistant_voice, "system"))

    # 去重：相同 (prepared_text, model, voice) 只合成一次。
    seen: set[tuple[str, str, str]] = set()
    unique: list[tuple[str, str, str, str]] = []
    for text, model, voice, label in targets:
        prepared = prepare_text_for_tts(text)
        if not prepared:
            continue
        key = (prepared, model, voice)
        if key in seen:
            continue
        seen.add(key)
        unique.append((text, model, voice, label))
    return unique


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv if argv is not None else sys.argv[1:])

    user_config = load_config()
    # 真实合成：强制关掉 demo/voice_demo，并固定 deploy 默认 format（mp3），
    # 保证文件名摘要与运行时一致。
    synth_config = dataclasses.replace(
        user_config,
        demo_mode=False,
        voice_demo_mode=False,
        tts_format=AppConfig().tts_format,
    )
    if not synth_config.tts_base_url.strip() or not synth_config.tts_api_key.strip():
        print("缺少 TTS 配置：请在本地 .env 配好 QWEN_BASE_URL/QWEN_API_KEY（或 TTS_BASE_URL/TTS_API_KEY）后重试。")
        print("（脚本会真实调用 API 并消耗 token；不要提交 .env 或 API Key。）")
        return 1

    # 文件名摘要使用 deploy 默认配置（无 .env 时的运行时配置），确保命中。
    key_config = AppConfig()
    targets = _collect_targets(args.assistant_only)
    print(f"准备生成 {len(targets)} 条预置演示语音 → {DEMO_VOICE_DIR.relative_to(PROJECT_ROOT).as_posix()}/")

    DEMO_VOICE_DIR.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, str]] = []
    success = 0
    failure = 0

    for index, (text, model, voice, label) in enumerate(targets, start=1):
        result = synthesize_speech(
            text,
            model=model,
            voice=voice,
            config=synth_config,
            use_cache=False,
        )
        metadata = result.get("metadata", {})
        source = str(metadata.get("source", ""))
        audio_bytes = result.get("audio_bytes")
        if isinstance(audio_bytes, (bytes, bytearray)) and audio_bytes and source == "tts":
            path = demo_voice_path_for(prepare_text_for_tts(text), key_config, voice, model)
            path.write_bytes(bytes(audio_bytes))
            manifest.append(
                {
                    "text": text,
                    "model": model,
                    "voice": voice,
                    "role": label,
                    "file": path.name,
                    "bytes": str(len(audio_bytes)),
                }
            )
            success += 1
            print(f"[{index}/{len(targets)}] OK   {label:7s} {path.name}  「{_preview(text)}」")
        else:
            failure += 1
            reason = str(metadata.get("reason", "")) or "未知原因"
            print(f"[{index}/{len(targets)}] FAIL {label:7s} source={source or 'none'} reason={reason}  「{_preview(text)}」")

    manifest_path = DEMO_VOICE_DIR / "manifest.json"
    manifest_path.write_text(
        json.dumps(sorted(manifest, key=lambda row: row["file"]), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("-" * 60)
    print(f"完成：成功 {success} 条，失败 {failure} 条。清单：{manifest_path.relative_to(PROJECT_ROOT).as_posix()}")
    if failure:
        print("失败多为老人音色 longlaoyi_v3 需账号权限；系统音色成功即可让默认演示出声。")
    print("接下来：git add assets/demo_voice/ 并提交，部署后默认演示即有声音。")
    return 0


def _preview(text: str, limit: int = 18) -> str:
    clean = str(text or "").strip().replace("\n", " ")
    return clean if len(clean) <= limit else clean[:limit] + "…"


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="离线生成项目内预置演示语音")
    parser.add_argument(
        "--assistant-only",
        action="store_true",
        help="只生成系统问题语音（qwen-tts / Cherry），跳过模拟老人音色",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
