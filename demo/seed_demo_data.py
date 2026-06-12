from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Any, Optional, Union


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.db import DEFAULT_DB_PATH, init_db, save_session  # noqa: E402
from core.mock_data import load_all_fixture_sessions  # noqa: E402
from core.schemas import COGNITIVE_DOMAINS, DISCLAIMER  # noqa: E402


CURRENT_DEMO_USER_ID = "zhang-nainai"
CURRENT_DEMO_USER_DISPLAY_NAME = "张奶奶"
RECENT_CURRENT_USER_SESSIONS = [
    {
        "session_id": "dialog-202606040918000000800",
        "created_at": "2026-06-04T09:18:00.000+08:00",
        "risk_level": "low",
        "components": ["dialogue"],
        "scores": {
            "orientation": 0.91,
            "memory": 0.83,
            "language": 0.89,
            "executive_function": 0.82,
            "attention": 0.80,
            "visuospatial": 0.78,
        },
        "dialogue_text": "上午快速访谈中能说清当天日期和早饭内容，三词复述基本完整，路线描述稍微停顿。",
        "explanation": "对话评估：近期第一轮近期测试，回答整体连贯，个别空间路线问题需要稍作思考，作为低风险演示记录保存。",
    },
    {
        "session_id": "clock-202606041606000000800",
        "created_at": "2026-06-04T16:06:00.000+08:00",
        "risk_level": "low",
        "components": ["clock"],
        "scores": {
            "orientation": None,
            "memory": None,
            "language": None,
            "executive_function": 0.86,
            "attention": None,
            "visuospatial": 0.88,
        },
        "clock_profile": "balanced",
        "explanation": "画钟测试：单独复核拍照和 VLM/mock 结构化链路，数字、顺序和指针整体清楚，作为近期低风险画钟记录保存。",
    },
    {
        "session_id": "assessment-202606051024000000800",
        "created_at": "2026-06-05T10:24:00.000+08:00",
        "risk_level": "low",
        "components": ["dialogue", "clock"],
        "scores": {
            "orientation": 0.93,
            "memory": 0.86,
            "language": 0.91,
            "executive_function": 0.87,
            "attention": 0.84,
            "visuospatial": 0.90,
        },
        "dialogue_text": "上午综合测试中能主动补充买菜和看日历的细节，倒背数字速度偏慢但顺序正确。",
        "clock_profile": "balanced",
        "explanation": "本次综合评估包含对话评估和画钟测试。对话和画钟都较稳定，适合展示系统可以把同一轮对话与画钟合并成一条综合记录。",
    },
    {
        "session_id": "dialog-202606052031000000800",
        "created_at": "2026-06-05T20:31:00.000+08:00",
        "risk_level": "medium",
        "components": ["dialogue"],
        "scores": {
            "orientation": 0.74,
            "memory": 0.62,
            "language": 0.80,
            "executive_function": 0.66,
            "attention": 0.59,
            "visuospatial": 0.70,
        },
        "dialogue_text": "晚间链路测试中，日期回答先说成昨天，倒数练习中间跳过一个数，提醒后能继续完成。",
        "explanation": "对话评估：晚间复测出现轻微波动，系统给出中等风险提示，建议结合当天状态和后续记录继续观察，不作为诊断结论。",
    },
    {
        "session_id": "assessment-202606060942000000800",
        "created_at": "2026-06-06T09:42:00.000+08:00",
        "risk_level": "low",
        "components": ["dialogue", "clock"],
        "scores": {
            "orientation": 0.89,
            "memory": 0.82,
            "language": 0.88,
            "executive_function": 0.84,
            "attention": 0.80,
            "visuospatial": 0.86,
        },
        "dialogue_text": "周末上午复测时能说清当天安排，三词回忆少停顿一次，做饭步骤能按先后说出来。",
        "clock_profile": "balanced",
        "explanation": "本次综合评估包含对话评估和画钟测试。较前一晚对话波动明显恢复，系统提示以低风险和继续观察为主。",
    },
    {
        "session_id": "clock-202606061555000000800",
        "created_at": "2026-06-06T15:55:00.000+08:00",
        "risk_level": "medium",
        "components": ["clock"],
        "scores": {
            "orientation": None,
            "memory": None,
            "language": None,
            "executive_function": 0.64,
            "attention": None,
            "visuospatial": 0.58,
        },
        "clock_profile": "right_shifted",
        "explanation": "画钟测试：下午补测时数字略向右侧集中，目标时间指针不够稳定，适合展示系统对画钟结构波动的提示能力。",
    },
    {
        "session_id": "assessment-202606071642000000800",
        "created_at": "2026-06-07T16:42:00.000+08:00",
        "risk_level": "low",
        "components": ["dialogue", "clock"],
        "scores": {
            "orientation": 0.90,
            "memory": 0.85,
            "language": 0.89,
            "executive_function": 0.86,
            "attention": 0.83,
            "visuospatial": 0.88,
        },
        "dialogue_text": "最近一次综合测试中能说明手机日期、午饭和当天安排，倒数稍慢但能自我修正。",
        "clock_profile": "balanced",
        "explanation": "本次综合评估包含对话评估和画钟测试。作为最近一次完整链路自测，结果整体稳定，系统建议继续保持定期低压力观察。",
    },
]


def seed_demo_data(db_path: Optional[Union[str, Path]] = None) -> int:
    path = init_db(db_path)
    count = 0

    for sessions in load_all_fixture_sessions().values():
        for session in sessions:
            save_session(session, db_path=path)
            count += 1

    count += seed_current_user_recent_sessions(path)
    return count


def seed_current_user_recent_sessions(
    db_path: Optional[Union[str, Path]] = None,
) -> int:
    path = init_db(db_path)
    with sqlite3.connect(path) as connection:
        connection.execute(
            "DELETE FROM sessions WHERE user_id = ?",
            (CURRENT_DEMO_USER_ID,),
        )

    count = 0
    for item in RECENT_CURRENT_USER_SESSIONS:
        save_session(_build_recent_current_user_record(item), db_path=path)
        count += 1
    return count


def _build_recent_current_user_record(item: dict[str, Any]) -> dict[str, Any]:
    scores = _normalize_scores(item["scores"])
    components = list(item["components"])
    evidence: list[dict[str, str]] = []
    record = {
        "session_id": item["session_id"],
        "assessment_id": item["session_id"],
        "participant_id": CURRENT_DEMO_USER_ID,
        "user_id": CURRENT_DEMO_USER_ID,
        "created_at": item["created_at"],
        "trajectory": "recent_practice",
        "components": components,
        "domain_scores": scores,
        "evidence": evidence,
        "risk_level": item["risk_level"],
        "explanation": item["explanation"],
        "disclaimer": DISCLAIMER,
        "is_mock": True,
        "is_simulated": True,
        "display_name": CURRENT_DEMO_USER_DISPLAY_NAME,
    }

    if "dialogue" in components:
        dialogue_evidence = _dialogue_evidence(item["dialogue_text"])
        evidence.extend(dialogue_evidence)
        record["dialogue_result"] = {
            "session_id": f"{item['session_id']}-dialogue",
            "participant_id": CURRENT_DEMO_USER_ID,
            "domain_scores": scores,
            "evidence": dialogue_evidence,
            "risk_level": item["risk_level"],
            "explanation": f"对话评估：{item['dialogue_text']}",
            "disclaimer": DISCLAIMER,
            "is_mock": True,
            "is_simulated": True,
            "metadata": {
                "source": "mock",
                "model": "recent-practice-dialog-demo",
                "reason": "默认演示对象近期测试记录",
            },
        }

    if "clock" in components:
        clock_result = _clock_result(item["clock_profile"], item["session_id"])
        evidence.extend(clock_result["evidence"])
        record["clock_result"] = clock_result
        record["cdt_features"] = clock_result["cdt_features"]

    return record


def _normalize_scores(raw_scores: dict[str, Any]) -> dict[str, Optional[float]]:
    return {
        domain: (
            None
            if raw_scores.get(domain) is None
            else round(float(raw_scores[domain]), 2)
        )
        for domain in COGNITIVE_DOMAINS
    }


def _dialogue_evidence(text: str) -> list[dict[str, str]]:
    return [
        {
            "domain": "memory",
            "source": "dialog",
            "text": text,
        },
        {
            "domain": "attention",
            "source": "dialog",
            "text": "访谈中保留日期、记忆、倒数和路线等生活化问题，作为演示证据。",
        },
    ]


def _clock_result(profile_name: str, session_id: str) -> dict[str, Any]:
    profiles = {
        "balanced": {
            "risk_level": "low",
            "scores": {"executive_function": 0.90, "visuospatial": 0.92},
            "findings": {
                "number_placement": "数字完整、顺序清楚，整体分布较均衡。",
                "hand_accuracy": "指针能表达 11:10，长短针方向比较清楚。",
            },
            "features": {
                "numbers_complete": True,
                "number_order_correct": True,
                "number_spacing": "normal",
                "number_distribution": "balanced",
                "hands_present": True,
                "target_time_match": True,
                "center_anchor_clear": True,
            },
            "evidence": [
                "数字围绕钟面分布较均衡。",
                "指针能表达目标时间 11:10。",
            ],
            "explanation": "画钟结构完整，未呈现明显视觉空间或执行步骤异常信号。",
        },
        "right_shifted": {
            "risk_level": "medium",
            "scores": {"executive_function": 0.62, "visuospatial": 0.56},
            "findings": {
                "number_placement": "数字大体完整，但略向右侧集中，左侧空间留白偏多。",
                "hand_accuracy": "指针能表达大致意图，但目标时间匹配不够稳定。",
            },
            "features": {
                "numbers_complete": True,
                "number_order_correct": True,
                "number_spacing": "irregular",
                "number_distribution": "right_shifted",
                "hands_present": True,
                "target_time_match": False,
                "center_anchor_clear": True,
            },
            "evidence": [
                "数字分布略向右侧集中。",
                "指针与目标时间 11:10 匹配不够稳定。",
            ],
            "explanation": "画钟出现轻度空间分布和指针匹配波动，建议结合对话记录继续观察。",
        },
        "irregular": {
            "risk_level": "medium",
            "scores": {"executive_function": 0.58, "visuospatial": 0.52},
            "findings": {
                "number_placement": "数字完整但间距不均，局部略拥挤。",
                "hand_accuracy": "指针存在，但长短针区分和目标时间匹配不够清楚。",
            },
            "features": {
                "numbers_complete": True,
                "number_order_correct": True,
                "number_spacing": "crowded",
                "number_distribution": "right_shifted",
                "hands_present": True,
                "target_time_match": False,
                "center_anchor_clear": True,
            },
            "evidence": [
                "数字间距不均，局部略拥挤。",
                "长短针区分和目标时间匹配不够清楚。",
            ],
            "explanation": "画钟结构可辨认，但空间布局和执行目标时间存在波动。",
        },
    }
    profile = profiles.get(profile_name, profiles["balanced"])
    scores = {domain: None for domain in COGNITIVE_DOMAINS}
    scores.update(profile["scores"])
    evidence = [
        {
            "domain": "visuospatial",
            "source": "clock",
            "text": text,
        }
        for text in profile["evidence"]
    ]
    return {
        "session_id": f"{session_id}-clock",
        "uploaded_filename": f"{profile_name}-recent-clock.png",
        "target_time": "11:10",
        "source": "mock",
        "model": "recent-practice-clock-demo",
        "domain_scores": scores,
        "evidence": evidence,
        "risk_level": profile["risk_level"],
        "clock_findings": profile["findings"],
        "cdt_features": profile["features"],
        "explanation": profile["explanation"],
        "disclaimer": DISCLAIMER,
        "is_mock": True,
        "is_simulated": True,
        "metadata": {
            "source": "mock",
            "model": "recent-practice-clock-demo",
            "reason": "默认演示对象近期画钟记录",
        },
    }


def main() -> None:
    count = seed_demo_data(DEFAULT_DB_PATH)
    print(f"Seeded {count} demo sessions into {DEFAULT_DB_PATH}")


if __name__ == "__main__":
    main()
