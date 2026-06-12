from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.config import AppConfig, load_config  # noqa: E402
from core.vlm_client import analyze_clock_image  # noqa: E402


def main(argv: Optional[list[str]] = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    config = load_config()
    _print_config_status(config)

    if not args:
        print("Usage: python scripts\\check_qwen_vl_connection.py path\\to\\clock.png")
        return 2

    image_path = Path(args[0])
    if not image_path.exists() or not image_path.is_file():
        print(f"Image file not found: {image_path}")
        return 2

    image_bytes = image_path.read_bytes()
    report = analyze_clock_image(
        image=image_bytes,
        filename=image_path.name,
        config=config,
    )
    metadata = report.get("metadata", {})
    source = metadata.get("source", "unknown")

    if source == "qwen-vl":
        print("Qwen-VL API connection succeeded")
        print(f"risk_level={report.get('risk_level', 'unknown')}")
        print(f"explanation={report.get('explanation', '')}")
        return 0

    if source == "mock":
        print("Qwen-VL API connection returned mock result.")
        print("Please check DEMO_MODE and VLM_BASE_URL/VLM_API_KEY/VLM_MODEL.")
        print(f"reason={metadata.get('reason', '')}")
        return 2

    print("Qwen-VL API connection failed")
    print(f"reason={metadata.get('reason', '')}")
    print(f"validation_errors={metadata.get('validation_errors', [])}")
    return 1


def _print_config_status(config: AppConfig) -> None:
    print(f"DEMO_MODE={str(config.demo_mode).lower()}")
    print(f"VLM_BASE_URL configured={bool(config.vlm_base_url.strip())}")
    print(f"VLM_MODEL={config.vlm_model or '未配置'}")
    print(f"VLM_API_KEY configured={bool(config.vlm_api_key.strip())}")


if __name__ == "__main__":
    raise SystemExit(main())
