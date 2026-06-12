from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "demo" / "fixtures" / "clock_samples"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "clock_samples_12"
EVAL_CASES_PATH = PROJECT_ROOT / "data" / "eval" / "clock_cases_12.json"
EVAL_30_OUTPUT_DIR = PROJECT_ROOT / "data" / "eval" / "clock_samples_30"
EVAL_30_CASES_PATH = PROJECT_ROOT / "data" / "eval" / "clock_cases_30.json"
TARGET_TIME = "11:10"
CDT_FEATURE_KEYS = (
    "numbers_complete",
    "number_order_correct",
    "number_spacing",
    "number_distribution",
    "hands_present",
    "target_time_match",
    "center_anchor_clear",
)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print(
            "Pillow is not available. Install project dependencies or run this "
            "script in an environment that includes Pillow."
        )
        return 2

    if args.eval_set_12 or args.eval_set_30:
        eval_size = 30 if args.eval_set_30 else 12
        default_output_dir = EVAL_30_OUTPUT_DIR if args.eval_set_30 else EVAL_OUTPUT_DIR
        default_cases_path = EVAL_30_CASES_PATH if args.eval_set_30 else EVAL_CASES_PATH
        output_dir = _resolve_project_path(args.output_dir, default_output_dir)
        cases_path = _resolve_project_path(args.cases_output, default_cases_path)
        cases = generate_eval_clock_set(
            Image=Image,
            ImageDraw=ImageDraw,
            ImageFont=ImageFont,
            eval_size=eval_size,
            output_dir=output_dir,
            cases_path=cases_path,
            force=args.force,
        )
        print(f"Generated {len(cases)} labeled clock eval cases in {output_dir}")
        print(f"Case labels written to {cases_path}")
        return 0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _draw_clock(
        Image,
        ImageDraw,
        OUTPUT_DIR / "normal_clock.png",
        number_mode="normal",
        hands_mode="normal",
    )
    _draw_clock(
        Image,
        ImageDraw,
        OUTPUT_DIR / "spatial_shift_clock.png",
        number_mode="shift_right",
        hands_mode="normal",
    )
    _draw_clock(
        Image,
        ImageDraw,
        OUTPUT_DIR / "wrong_hands_clock.png",
        number_mode="normal",
        hands_mode="wrong",
    )
    print(f"Generated demo clock samples in {OUTPUT_DIR}")
    return 0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic clock drawing samples for CogniGuard."
    )
    parser.add_argument(
        "--eval-set-12",
        action="store_true",
        help="Generate a 12-image labeled CDT evaluation set.",
    )
    parser.add_argument(
        "--eval-set-30",
        action="store_true",
        help="Generate a 30-image labeled CDT evaluation set.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Directory for --eval-set-12 images. Defaults to "
            "data/eval/clock_samples_12 or data/eval/clock_samples_30. "
            "Path must stay inside project."
        ),
    )
    parser.add_argument(
        "--cases-output",
        default=None,
        help=(
            "JSON output path for --eval-set-12 labels. Defaults to "
            "data/eval/clock_cases_12.json or data/eval/clock_cases_30.json. "
            "Path must stay inside project."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite generated eval-set files if they already exist.",
    )
    args = parser.parse_args(argv)
    if args.eval_set_12 and args.eval_set_30:
        parser.error("--eval-set-12 and --eval-set-30 cannot be used together")
    return args


def _resolve_project_path(value: str | None, default: Path) -> Path:
    raw_path = default if value is None or not str(value).strip() else Path(value)
    resolved = raw_path.resolve() if raw_path.is_absolute() else (PROJECT_ROOT / raw_path).resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as error:
        raise ValueError(f"path must stay inside project: {value}") from error
    return resolved


def generate_eval_clock_set_12(
    *,
    Image: Any,
    ImageDraw: Any,
    ImageFont: Any,
    output_dir: Path = EVAL_OUTPUT_DIR,
    cases_path: Path = EVAL_CASES_PATH,
    force: bool = False,
) -> list[dict[str, Any]]:
    return generate_eval_clock_set(
        Image=Image,
        ImageDraw=ImageDraw,
        ImageFont=ImageFont,
        eval_size=12,
        output_dir=output_dir,
        cases_path=cases_path,
        force=force,
    )


def generate_eval_clock_set(
    *,
    Image: Any,
    ImageDraw: Any,
    ImageFont: Any,
    eval_size: int,
    output_dir: Path,
    cases_path: Path,
    force: bool = False,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases_path.parent.mkdir(parents=True, exist_ok=True)
    specs = _eval_clock_specs()[:eval_size]
    if len(specs) != eval_size:
        raise ValueError(f"unsupported eval_size: {eval_size}")
    _guard_no_existing_outputs(output_dir, cases_path, specs, force=force)

    font = _load_clock_font(ImageFont, size=30)
    small_font = _load_clock_font(ImageFont, size=24)
    cases: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        filename = f"{index:02d}_{spec['case_id']}.png"
        image_path = output_dir / filename
        _draw_eval_clock(
            Image=Image,
            ImageDraw=ImageDraw,
            image_path=image_path,
            spec=spec,
            font=font,
            small_font=small_font,
        )
        relative_image_path = image_path.relative_to(PROJECT_ROOT).as_posix()
        cases.append(
            {
                "case_id": spec["case_id"],
                "label": spec["label"],
                "image_path": relative_image_path,
                "target_time": TARGET_TIME,
                "is_mock": True,
                "expected_risk_level": spec["expected_risk_level"],
                "expected_low_domains": spec["expected_low_domains"],
                "expected_cdt_features": {
                    key: spec["expected_cdt_features"][key] for key in CDT_FEATURE_KEYS
                },
                "notes": spec["notes"],
            }
        )

    cases_path.write_text(
        json.dumps(cases, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return cases


def _guard_no_existing_outputs(
    output_dir: Path,
    cases_path: Path,
    specs: list[dict[str, Any]],
    *,
    force: bool,
) -> None:
    if force:
        return

    existing_paths = []
    for index, spec in enumerate(specs, start=1):
        candidate = output_dir / f"{index:02d}_{spec['case_id']}.png"
        if candidate.exists():
            existing_paths.append(candidate)
    if cases_path.exists():
        existing_paths.append(cases_path)

    if existing_paths:
        formatted = "\n".join(str(path) for path in existing_paths[:5])
        raise FileExistsError(
            "generated eval-set output already exists; use --force to overwrite:\n"
            f"{formatted}"
        )


def _eval_clock_specs() -> list[dict[str, Any]]:
    return [
        _clock_spec(
            case_id="clock12_normal_balanced_001",
            label="normal",
            expected_risk_level="low",
            expected_low_domains=[],
            number_layout="balanced",
            hand_mode="target",
            notes="模拟数据，非临床数据；数字、圆形轮廓、中心点和 11:10 指针均完整。",
        ),
        _clock_spec(
            case_id="clock12_normal_slight_wobble_002",
            label="normal",
            expected_risk_level="low",
            expected_low_domains=[],
            number_layout="slight_wobble",
            hand_mode="target",
            notes="模拟数据，非临床数据；轻微手绘抖动但结构完整，预期仍为低风险。",
        ),
        _clock_spec(
            case_id="clock12_normal_large_round_003",
            label="normal",
            expected_risk_level="low",
            expected_low_domains=[],
            number_layout="large_round",
            hand_mode="target",
            notes="模拟数据，非临床数据；大圆盘正常排布，目标时间清楚。",
        ),
        _clock_spec(
            case_id="clock12_normal_compact_004",
            label="normal",
            expected_risk_level="low",
            expected_low_domains=[],
            number_layout="compact_balanced",
            hand_mode="target",
            notes="模拟数据，非临床数据；数字稍靠内但均衡完整，指针符合目标时间。",
        ),
        _clock_spec(
            case_id="clock12_mild_right_shift_005",
            label="mild_decline",
            expected_risk_level="medium",
            expected_low_domains=["visuospatial"],
            number_layout="right_shifted",
            hand_mode="target",
            notes="模拟数据，非临床数据；数字整体向右偏移，主要考察视觉空间识别。",
        ),
        _clock_spec(
            case_id="clock12_mild_left_shift_006",
            label="mild_decline",
            expected_risk_level="medium",
            expected_low_domains=["visuospatial"],
            number_layout="left_shifted",
            hand_mode="target",
            notes="模拟数据，非临床数据；数字整体向左偏移，目标时间本身正确。",
        ),
        _clock_spec(
            case_id="clock12_mild_crowded_007",
            label="mild_decline",
            expected_risk_level="medium",
            expected_low_domains=["visuospatial"],
            number_layout="crowded",
            hand_mode="target",
            notes="模拟数据，非临床数据；数字集中拥挤但顺序基本完整。",
        ),
        _clock_spec(
            case_id="clock12_mild_wrong_hands_008",
            label="mild_decline",
            expected_risk_level="medium",
            expected_low_domains=["executive_function"],
            number_layout="balanced",
            hand_mode="wrong_time",
            notes="模拟数据，非临床数据；数字布局正常但指针不符合 11:10。",
        ),
        _clock_spec(
            case_id="clock12_mild_irregular_spacing_009",
            label="mild_decline",
            expected_risk_level="medium",
            expected_low_domains=["visuospatial"],
            number_layout="irregular",
            hand_mode="target",
            notes="模拟数据，非临床数据；数字间距明显不均，考察空间组织特征。",
        ),
        _clock_spec(
            case_id="clock12_high_missing_numbers_010",
            label="obvious_issue",
            expected_risk_level="high",
            expected_low_domains=["visuospatial", "executive_function"],
            number_layout="missing_numbers",
            hand_mode="wrong_time",
            notes="模拟数据，非临床数据；多个数字缺失且指针错误，预期明显异常。",
        ),
        _clock_spec(
            case_id="clock12_high_no_hands_011",
            label="obvious_issue",
            expected_risk_level="high",
            expected_low_domains=["executive_function"],
            number_layout="balanced",
            hand_mode="none",
            notes="模拟数据，非临床数据；数字基本完整但没有有效指针，无法完成目标时间。",
        ),
        _clock_spec(
            case_id="clock12_high_disorganized_012",
            label="obvious_issue",
            expected_risk_level="high",
            expected_low_domains=["visuospatial", "executive_function"],
            number_layout="disorganized",
            hand_mode="none",
            center_anchor=False,
            notes="模拟数据，非临床数据；数字聚集且顺序混乱，没有中心锚点和有效指针。",
        ),
        _clock_spec(
            case_id="clock30_normal_balanced_repeat_013",
            label="normal",
            expected_risk_level="low",
            expected_low_domains=[],
            number_layout="balanced",
            hand_mode="target",
            notes="模拟数据，非临床数据；正常画钟补充样本，数字和指针完整。",
        ),
        _clock_spec(
            case_id="clock30_normal_slight_wobble_014",
            label="normal",
            expected_risk_level="low",
            expected_low_domains=[],
            number_layout="slight_wobble",
            hand_mode="target",
            notes="模拟数据，非临床数据；轻微手绘抖动但主要结构完整。",
        ),
        _clock_spec(
            case_id="clock30_normal_compact_015",
            label="normal",
            expected_risk_level="low",
            expected_low_domains=[],
            number_layout="compact_balanced",
            hand_mode="target",
            notes="模拟数据，非临床数据；数字略靠内但顺序和指针完整。",
        ),
        _clock_spec(
            case_id="clock30_normal_large_round_016",
            label="normal",
            expected_risk_level="low",
            expected_low_domains=[],
            number_layout="large_round",
            hand_mode="target",
            notes="模拟数据，非临床数据；大圆盘正常排布，作为正常补充样本。",
        ),
        _clock_spec(
            case_id="clock30_normal_balanced_repeat_017",
            label="normal",
            expected_risk_level="low",
            expected_low_domains=[],
            number_layout="balanced",
            hand_mode="target",
            notes="模拟数据，非临床数据；正常画钟补充样本，用于扩大核心结构字段基数。",
        ),
        _clock_spec(
            case_id="clock30_normal_compact_repeat_018",
            label="normal",
            expected_risk_level="low",
            expected_low_domains=[],
            number_layout="compact_balanced",
            hand_mode="target",
            notes="模拟数据，非临床数据；数字完整且中心锚点清楚，预期低风险。",
        ),
        _clock_spec(
            case_id="clock30_mild_wrong_hands_019",
            label="mild_decline",
            expected_risk_level="medium",
            expected_low_domains=["executive_function"],
            number_layout="balanced",
            hand_mode="wrong_time",
            notes="模拟数据，非临床数据；数字完整但目标时间错误，预期触发轻度预警。",
        ),
        _clock_spec(
            case_id="clock30_mild_wrong_hands_compact_020",
            label="mild_decline",
            expected_risk_level="medium",
            expected_low_domains=["executive_function"],
            number_layout="compact_balanced",
            hand_mode="wrong_time",
            notes="模拟数据，非临床数据；结构完整但指针不符合 11:10。",
        ),
        _clock_spec(
            case_id="clock30_mild_right_shift_021",
            label="mild_decline",
            expected_risk_level="medium",
            expected_low_domains=["visuospatial"],
            number_layout="right_shifted",
            hand_mode="target",
            notes="模拟数据，非临床数据；数字整体偏右，主要考察空间布局提示。",
        ),
        _clock_spec(
            case_id="clock30_mild_left_shift_022",
            label="mild_decline",
            expected_risk_level="medium",
            expected_low_domains=["visuospatial"],
            number_layout="left_shifted",
            hand_mode="target",
            notes="模拟数据，非临床数据；数字整体偏左，主要考察空间布局提示。",
        ),
        _clock_spec(
            case_id="clock30_mild_crowded_023",
            label="mild_decline",
            expected_risk_level="medium",
            expected_low_domains=["visuospatial"],
            number_layout="crowded",
            hand_mode="target",
            notes="模拟数据，非临床数据；数字拥挤集中但基础结构仍可辨认。",
        ),
        _clock_spec(
            case_id="clock30_mild_irregular_024",
            label="mild_decline",
            expected_risk_level="medium",
            expected_low_domains=["visuospatial"],
            number_layout="irregular",
            hand_mode="target",
            notes="模拟数据，非临床数据；数字间距不均，作为轻度异常补充样本。",
        ),
        _clock_spec(
            case_id="clock30_high_missing_numbers_025",
            label="obvious_issue",
            expected_risk_level="high",
            expected_low_domains=["visuospatial", "executive_function"],
            number_layout="missing_numbers",
            hand_mode="wrong_time",
            notes="模拟数据，非临床数据；数字缺失且指针错误，明显异常。",
        ),
        _clock_spec(
            case_id="clock30_high_missing_no_hands_026",
            label="obvious_issue",
            expected_risk_level="high",
            expected_low_domains=["visuospatial", "executive_function"],
            number_layout="missing_numbers",
            hand_mode="none",
            notes="模拟数据，非临床数据；数字缺失且没有有效指针，明显异常。",
        ),
        _clock_spec(
            case_id="clock30_high_no_hands_balanced_027",
            label="obvious_issue",
            expected_risk_level="high",
            expected_low_domains=["executive_function"],
            number_layout="balanced",
            hand_mode="none",
            notes="模拟数据，非临床数据；数字完整但没有指针，无法完成目标时间。",
        ),
        _clock_spec(
            case_id="clock30_high_no_hands_compact_028",
            label="obvious_issue",
            expected_risk_level="high",
            expected_low_domains=["executive_function"],
            number_layout="compact_balanced",
            hand_mode="none",
            notes="模拟数据，非临床数据；无有效指针，预期明显异常预警。",
        ),
        _clock_spec(
            case_id="clock30_high_disorganized_029",
            label="obvious_issue",
            expected_risk_level="high",
            expected_low_domains=["visuospatial", "executive_function"],
            number_layout="disorganized",
            hand_mode="none",
            notes="模拟数据，非临床数据；数字聚集且顺序混乱，没有有效指针。",
        ),
        _clock_spec(
            case_id="clock30_high_disorganized_wrong_hands_030",
            label="obvious_issue",
            expected_risk_level="high",
            expected_low_domains=["visuospatial", "executive_function"],
            number_layout="disorganized",
            hand_mode="wrong_time",
            notes="模拟数据，非临床数据；数字顺序混乱且指针错误，明显异常。",
        ),
    ]


def _clock_spec(
    *,
    case_id: str,
    label: str,
    expected_risk_level: str,
    expected_low_domains: list[str],
    number_layout: str,
    hand_mode: str,
    notes: str,
    center_anchor: bool = True,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "label": label,
        "expected_risk_level": expected_risk_level,
        "expected_low_domains": expected_low_domains,
        "number_layout": number_layout,
        "hand_mode": hand_mode,
        "center_anchor": center_anchor,
        "expected_cdt_features": _expected_cdt_features(
            number_layout=number_layout,
            hand_mode=hand_mode,
            center_anchor=center_anchor,
        ),
        "notes": notes,
    }


def _expected_cdt_features(
    *,
    number_layout: str,
    hand_mode: str,
    center_anchor: bool,
) -> dict[str, Any]:
    spacing_by_layout = {
        "balanced": "normal",
        "slight_wobble": "normal",
        "large_round": "normal",
        "compact_balanced": "normal",
        "right_shifted": "shifted",
        "left_shifted": "shifted",
        "crowded": "crowded",
        "irregular": "irregular",
        "missing_numbers": "irregular",
        "disorganized": "crowded",
    }
    distribution_by_layout = {
        "balanced": "balanced",
        "slight_wobble": "balanced",
        "large_round": "balanced",
        "compact_balanced": "balanced",
        "right_shifted": "right_shifted",
        "left_shifted": "left_shifted",
        "crowded": "clustered",
        "irregular": "balanced",
        "missing_numbers": "clustered",
        "disorganized": "clustered",
    }
    return {
        "numbers_complete": number_layout not in {"missing_numbers", "disorganized"},
        "number_order_correct": number_layout != "disorganized",
        "number_spacing": spacing_by_layout[number_layout],
        "number_distribution": distribution_by_layout[number_layout],
        "hands_present": hand_mode != "none",
        "target_time_match": hand_mode == "target",
        "center_anchor_clear": center_anchor,
    }


def _draw_eval_clock(
    *,
    Image: Any,
    ImageDraw: Any,
    image_path: Path,
    spec: dict[str, Any],
    font: Any,
    small_font: Any,
) -> None:
    size = 512
    center = (size // 2, size // 2)
    radius = 190
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    circle_box = (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )
    if spec["number_layout"] == "large_round":
        circle_box = (50, 50, 462, 462)
    draw.ellipse(circle_box, outline="black", width=5)

    _draw_eval_numbers(
        draw=draw,
        center=center,
        layout=spec["number_layout"],
        font=font,
        small_font=small_font,
    )
    _draw_eval_hands(
        draw=draw,
        center=center,
        mode=spec["hand_mode"],
    )
    if spec.get("center_anchor", True):
        draw.ellipse(
            (center[0] - 6, center[1] - 6, center[0] + 6, center[1] + 6),
            fill="black",
        )
    image.save(image_path)


def _draw_eval_numbers(
    *,
    draw: Any,
    center: tuple[int, int],
    layout: str,
    font: Any,
    small_font: Any,
) -> None:
    numbers = list(range(1, 13))
    if layout == "missing_numbers":
        numbers = [12, 1, 2, 3, 6, 9]
    if layout == "disorganized":
        numbers = [12, 5, 2, 8, 4, 11, 7, 3, 10]

    for index, number in enumerate(numbers):
        x, y = _number_position(number=number, index=index, center=center, layout=layout)
        active_font = small_font if layout in {"crowded", "disorganized"} else font
        _draw_centered_text(draw, (x, y), str(number), active_font)


def _number_position(
    *,
    number: int,
    index: int,
    center: tuple[int, int],
    layout: str,
) -> tuple[float, float]:
    angle = math.radians(number * 30 - 90)
    radius = 148
    x = center[0] + math.cos(angle) * radius
    y = center[1] + math.sin(angle) * radius

    if layout == "slight_wobble":
        offsets = {
            2: (6, -7),
            4: (-5, 7),
            7: (-8, 6),
            10: (7, -5),
        }
        dx, dy = offsets.get(number, (0, 0))
        x += dx
        y += dy
    elif layout == "large_round":
        x = center[0] + math.cos(angle) * 166
        y = center[1] + math.sin(angle) * 166
    elif layout == "compact_balanced":
        x = center[0] + math.cos(angle) * 118
        y = center[1] + math.sin(angle) * 118
    elif layout == "right_shifted":
        x = center[0] + 68 + math.cos(angle) * 108
        y = center[1] + math.sin(angle) * 136
    elif layout == "left_shifted":
        x = center[0] - 68 + math.cos(angle) * 108
        y = center[1] + math.sin(angle) * 136
    elif layout == "crowded":
        x = center[0] + 78 + math.cos(angle) * 54
        y = center[1] - 22 + math.sin(angle) * 62
    elif layout == "irregular":
        irregular_radii = {
            1: 164,
            2: 111,
            3: 151,
            4: 98,
            5: 168,
            6: 124,
            7: 158,
            8: 104,
            9: 173,
            10: 116,
            11: 149,
            12: 132,
        }
        active_radius = irregular_radii[number]
        x = center[0] + math.cos(angle) * active_radius
        y = center[1] + math.sin(angle) * active_radius
    elif layout == "missing_numbers":
        x = center[0] + 42 + math.cos(angle) * 88
        y = center[1] + 18 + math.sin(angle) * 112
    elif layout == "disorganized":
        cluster_positions = [
            (center[0] + 75, center[1] - 92),
            (center[0] + 110, center[1] - 55),
            (center[0] + 58, center[1] - 37),
            (center[0] + 125, center[1] - 8),
            (center[0] + 78, center[1] + 28),
            (center[0] + 137, center[1] + 45),
            (center[0] + 56, center[1] + 67),
            (center[0] + 118, center[1] + 88),
            (center[0] + 84, center[1] + 110),
        ]
        x, y = cluster_positions[index]

    return x, y


def _draw_eval_hands(
    *,
    draw: Any,
    center: tuple[int, int],
    mode: str,
) -> None:
    if mode == "none":
        return
    if mode == "wrong_time":
        _draw_hand(draw, center, angle_degrees=180, length=124, width=6)
        _draw_hand(draw, center, angle_degrees=90, length=82, width=9)
        return

    _draw_hand(draw, center, angle_degrees=60, length=130, width=6)
    _draw_hand(draw, center, angle_degrees=335, length=88, width=9)


def _draw_centered_text(draw: Any, center: tuple[float, float], text: str, font: Any) -> None:
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
    except AttributeError:
        width, height = draw.textsize(text, font=font)
    draw.text((center[0] - width / 2, center[1] - height / 2), text, fill="black", font=font)


def _load_clock_font(ImageFont: Any, *, size: int) -> Any:
    font_path = PROJECT_ROOT / "static" / "fonts" / "NotoSansSC-Medium.woff2"
    try:
        return ImageFont.truetype(str(font_path), size=size)
    except OSError:
        try:
            return ImageFont.truetype("arial.ttf", size=size)
        except OSError:
            return ImageFont.load_default()


def _draw_clock(Image, ImageDraw, output_path: Path, number_mode: str, hands_mode: str) -> None:
    size = 360
    center = (size // 2, size // 2)
    radius = 135
    image = Image.new("RGB", (size, size), "white")
    draw = ImageDraw.Draw(image)

    draw.ellipse(
        (
            center[0] - radius,
            center[1] - radius,
            center[0] + radius,
            center[1] + radius,
        ),
        outline="black",
        width=4,
    )

    for number in range(1, 13):
        angle = math.radians(number * 30 - 90)
        number_radius = 105
        x = center[0] + math.cos(angle) * number_radius
        y = center[1] + math.sin(angle) * number_radius
        if number_mode == "shift_right":
            number_radius = 72
            x = center[0] + 72 + math.cos(angle) * number_radius
            y = center[1] + math.sin(angle) * 98
            if number in {7, 8, 9, 10, 11}:
                x += 18
        draw.text((x - 8, y - 8), str(number), fill="black")

    if hands_mode == "wrong":
        _draw_hand(draw, center, angle_degrees=180, length=88, width=5)
        _draw_hand(draw, center, angle_degrees=90, length=58, width=7)
    else:
        _draw_hand(draw, center, angle_degrees=60, length=88, width=5)
        _draw_hand(draw, center, angle_degrees=335, length=62, width=7)

    draw.ellipse(
        (center[0] - 5, center[1] - 5, center[0] + 5, center[1] + 5),
        fill="black",
    )
    image.save(output_path)


def _draw_hand(draw, center: tuple[int, int], angle_degrees: float, length: int, width: int) -> None:
    angle = math.radians(angle_degrees - 90)
    end = (
        center[0] + math.cos(angle) * length,
        center[1] + math.sin(angle) * length,
    )
    draw.line((center, end), fill="black", width=width)


if __name__ == "__main__":
    raise SystemExit(main())
