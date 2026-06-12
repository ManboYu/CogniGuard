from __future__ import annotations

import copy
import json
import sqlite3

import pytest

from core.db import (
    authenticate_user,
    get_recent_sessions,
    init_db,
    list_demo_users,
    list_sessions,
    save_session,
)
from core.memory import init_memory, query_recent_memories, save_session_memory
from core.mock_data import load_fixture_sessions
from core.report import generate_mock_dialog_report
from core.report import summarize_trend
from core.schemas import COGNITIVE_DOMAINS
from core.session_history import (
    CURRENT_USER_DISPLAY_NAME,
    CURRENT_USER_ID,
    build_clock_assessment_record,
    build_clock_session_record,
    build_dialog_assessment_record,
    build_dialog_session_record,
    find_assessment_record,
    get_current_user_profile,
    load_current_user_sessions,
    load_sessions_for_brief,
    store_current_user_profile,
)
from demo.seed_demo_data import seed_demo_data


def test_init_db_creates_sessions_table(tmp_path) -> None:
    db_path = tmp_path / "cogniguard-test.db"

    returned_path = init_db(db_path)

    assert returned_path == db_path
    assert db_path.exists()

    with sqlite3.connect(db_path) as connection:
        columns = {
            row[1]
            for row in connection.execute("PRAGMA table_info(sessions)").fetchall()
        }

    assert {
        "session_id",
        "user_id",
        "created_at",
        "risk_level",
        "domain_scores_json",
        "evidence_json",
        "explanation",
        "raw_json",
    } <= columns


def test_init_db_creates_demo_users_with_hashed_passwords(tmp_path) -> None:
    db_path = tmp_path / "users.db"

    init_db(db_path)
    users = list_demo_users(db_path=db_path)

    assert {user["username"] for user in users} == {"zhang", "wang", "li"}
    assert {user["display_name"] for user in users} == {"张奶奶", "王叔叔", "李阿姨"}

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            "SELECT username, password_hash FROM users ORDER BY username"
        ).fetchall()

    assert rows
    for username, password_hash in rows:
        assert username in {"zhang", "wang", "li"}
        assert password_hash != "123456"
        assert password_hash.startswith("pbkdf2_sha256$")


def test_authenticate_user_accepts_correct_demo_password(tmp_path) -> None:
    db_path = tmp_path / "login.db"

    user = authenticate_user("zhang", "123456", db_path=db_path)

    assert user is not None
    assert user["user_id"] == CURRENT_USER_ID
    assert user["display_name"] == CURRENT_USER_DISPLAY_NAME
    assert "password_hash" not in user


def test_authenticate_user_rejects_wrong_password(tmp_path) -> None:
    db_path = tmp_path / "login-fail.db"

    assert authenticate_user("zhang", "wrong-password", db_path=db_path) is None
    assert authenticate_user("unknown", "123456", db_path=db_path) is None


def test_current_user_profile_falls_back_to_default_when_not_logged_in() -> None:
    profile = get_current_user_profile({})

    assert profile["user_id"] == CURRENT_USER_ID
    assert profile["display_name"] == CURRENT_USER_DISPLAY_NAME
    assert profile["is_authenticated"] is False


def test_save_and_list_sessions_use_temp_database(tmp_path) -> None:
    db_path = tmp_path / "cogniguard-test.db"
    record = copy.deepcopy(load_fixture_sessions("normal")[0])

    saved = save_session(record, db_path=db_path)
    sessions = list_sessions(db_path=db_path)

    assert saved["user_id"] == record["participant_id"]
    assert len(sessions) == 1
    assert sessions[0]["session_id"] == record["session_id"]
    assert sessions[0]["user_id"] == record["participant_id"]
    assert sessions[0]["participant_id"] == record["participant_id"]
    assert sessions[0]["domain_scores"] == record["domain_scores"]
    assert sessions[0]["evidence"] == record["evidence"]

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT user_id, domain_scores_json, evidence_json, raw_json
            FROM sessions
            WHERE session_id = ?
            """,
            (record["session_id"],),
        ).fetchone()

    assert row[0] == record["participant_id"]
    assert set(json.loads(row[1])) == set(COGNITIVE_DOMAINS)
    assert json.loads(row[2]) == record["evidence"]
    assert json.loads(row[3])["session_id"] == record["session_id"]


def test_get_recent_sessions_filters_and_keeps_chronological_order(tmp_path) -> None:
    db_path = tmp_path / "cogniguard-test.db"
    normal_sessions = load_fixture_sessions("normal")
    mild_sessions = load_fixture_sessions("mild_decline")

    for record in normal_sessions + mild_sessions:
        save_session(copy.deepcopy(record), db_path=db_path)

    normal_user = "demo-person-normal"
    recent = get_recent_sessions(normal_user, limit=2, db_path=db_path)
    all_normal = list_sessions(user_id=normal_user, db_path=db_path)

    assert [session["session_id"] for session in recent] == [
        "normal-002",
        "normal-003",
    ]
    assert [session["session_id"] for session in all_normal] == [
        "normal-001",
        "normal-002",
        "normal-003",
    ]


def test_save_session_rejects_invalid_domain_scores(tmp_path) -> None:
    record = copy.deepcopy(load_fixture_sessions("normal")[0])
    record["domain_scores"].pop("memory")

    with pytest.raises(ValueError, match="domain_scores"):
        save_session(record, db_path=tmp_path / "cogniguard-test.db")


def test_brief_sessions_fallback_to_fixtures_when_db_empty(tmp_path) -> None:
    result = load_sessions_for_brief("normal", db_path=tmp_path / "empty.db")

    assert result["source"] == "fixtures"
    assert result["user_id"] == "demo-person-normal"
    assert [session["session_id"] for session in result["sessions"]] == [
        "normal-001",
        "normal-002",
        "normal-003",
    ]


def test_brief_sessions_read_from_sqlite_after_seed(tmp_path) -> None:
    db_path = tmp_path / "seeded.db"
    seed_demo_data(db_path=db_path)

    result = load_sessions_for_brief("mild_decline", db_path=db_path)

    assert result["source"] == "sqlite"
    assert result["user_id"] == "demo-person-mild-decline"
    assert [session["session_id"] for session in result["sessions"]] == [
        "mild-decline-001",
        "mild-decline-002",
        "mild-decline-003",
    ]


def test_dialog_save_flow_writes_sqlite_and_memory_without_crashing(tmp_path) -> None:
    db_path = tmp_path / "dialog.db"
    memory_store = init_memory(storage_dir=tmp_path / "memory", prefer_chroma=False)
    report = generate_mock_dialog_report(["today is Saturday", "breakfast was porridge"])
    record = build_dialog_session_record(
        report,
        user_id="demo-person-normal",
        created_at="2026-05-23T10:00:00.123+08:00",
    )

    saved_record = save_session(record, db_path=db_path)
    saved_memory = save_session_memory(record, store=memory_store)
    recent_sessions = get_recent_sessions(
        "demo-person-normal",
        limit=1,
        db_path=db_path,
    )
    recent_memories = query_recent_memories(
        "demo-person-normal",
        limit=1,
        store=memory_store,
    )

    assert saved_record["session_id"] == "dialog-202605231000001230800"
    assert saved_memory["backend"] == "json"
    assert recent_sessions[0]["session_id"] == saved_record["session_id"]
    assert recent_memories[0]["session_id"] == saved_record["session_id"]


def test_current_user_default_dialog_session_can_save_and_read(tmp_path) -> None:
    db_path = tmp_path / "current-user.db"
    report = generate_mock_dialog_report(["today is Saturday", "breakfast was porridge"])
    record = build_dialog_session_record(
        report,
        created_at="2026-05-23T10:30:00.123+08:00",
    )

    saved = save_session(record, db_path=db_path)
    current = load_current_user_sessions(db_path=db_path, limit=3)

    assert saved["user_id"] == CURRENT_USER_ID
    assert current["source"] == "sqlite_current_user"
    assert current["user_id"] == CURRENT_USER_ID
    assert current["display_name"] == CURRENT_USER_DISPLAY_NAME
    assert [session["session_id"] for session in current["sessions"]] == [
        "dialog-202605231030001230800"
    ]


def test_current_logged_in_user_profile_can_drive_main_flow_reads(tmp_path) -> None:
    db_path = tmp_path / "current-profile.db"
    session_state: dict[str, object] = {}
    login_user = authenticate_user("wang", "123456", db_path=db_path)
    assert login_user is not None
    profile = store_current_user_profile(session_state, login_user)

    report = generate_mock_dialog_report(["今天周三。", "早饭吃了馒头。"])
    record = build_dialog_session_record(
        report,
        user_id=profile["user_id"],
        created_at="2026-06-03T09:00:00.123+08:00",
    )
    save_session(record, db_path=db_path)

    resolved_profile = get_current_user_profile(session_state)
    current = load_current_user_sessions(
        db_path=db_path,
        limit=3,
        user_profile=resolved_profile,
    )

    assert resolved_profile["user_id"] == "wang-shushu"
    assert resolved_profile["display_name"] == "王叔叔"
    assert resolved_profile["is_authenticated"] is True
    assert current["user_id"] == "wang-shushu"
    assert current["display_name"] == "王叔叔"
    assert [session["session_id"] for session in current["sessions"]] == [
        "dialog-202606030900001230800"
    ]


def test_clock_save_flow_writes_sqlite_raw_clock_result(tmp_path) -> None:
    db_path = tmp_path / "clock.db"
    report = {
        "metadata": {"source": "qwen-vl", "model": "qwen-vl-max"},
        "risk_level": "medium",
        "domain_scores": {
            "orientation": None,
            "memory": None,
            "language": None,
            "executive_function": 0.6,
            "attention": None,
            "visuospatial": 0.5,
        },
        "evidence": [
            {
                "domain": "visuospatial",
                "source": "clock",
                "text": "数字集中在右侧。",
            }
        ],
        "explanation": "画钟结果提示需要关注视觉空间和执行功能线索。",
        "clock_findings": {
            "number_placement": "数字集中在右侧。",
            "hand_accuracy": "指针方向不准确。",
        },
        "cdt_features": {
            "number_distribution": "right_shifted",
            "number_spacing": "crowded",
            "target_time_match": False,
        },
    }

    record = build_clock_session_record(
        report,
        created_at="2026-05-23T11:00:00.123+08:00",
        target_time="11:10",
    )
    saved = save_session(record, db_path=db_path)
    recent = get_recent_sessions(CURRENT_USER_ID, limit=1, db_path=db_path)

    assert saved["session_id"] == "clock-202605231100001230800"
    assert saved["user_id"] == CURRENT_USER_ID
    assert recent[0]["clock_result"]["source"] == "qwen-vl"
    assert recent[0]["clock_result"]["model"] == "qwen-vl-max"
    assert recent[0]["clock_result"]["target_time"] == "11:10"
    assert recent[0]["clock_result"]["cdt_features"]["target_time_match"] is False
    assert summarize_trend(recent)["domain_changes"]["orientation"] is None

    with sqlite3.connect(db_path) as connection:
        raw_json = connection.execute(
            "SELECT raw_json FROM sessions WHERE session_id = ?",
            (saved["session_id"],),
        ).fetchone()[0]

    raw_record = json.loads(raw_json)
    assert "clock_result" in raw_record
    assert raw_record["clock_result"]["cdt_features"]["number_distribution"] == "right_shifted"


def test_dialogue_and_clock_merge_into_same_assessment_record(tmp_path) -> None:
    db_path = tmp_path / "assessment.db"
    dialog_report = generate_mock_dialog_report(
        ["today is Saturday", "breakfast was porridge"]
    )
    dialog_report["risk_level"] = "low"

    dialog_record = build_dialog_assessment_record(
        dialog_report,
        created_at="2026-05-23T12:00:00.123+08:00",
    )
    save_session(dialog_record, db_path=db_path)

    existing_record = find_assessment_record(
        dialog_record["assessment_id"],
        db_path=db_path,
    )
    clock_report = {
        "metadata": {"source": "qwen-vl", "model": "qwen-vl-max"},
        "risk_level": "medium",
        "domain_scores": {
            "orientation": None,
            "memory": None,
            "language": None,
            "executive_function": 0.6,
            "attention": None,
            "visuospatial": 0.5,
        },
        "evidence": [
            {
                "domain": "visuospatial",
                "source": "clock",
                "text": "Numbers are shifted to the right.",
            }
        ],
        "explanation": "Clock result suggests visuospatial and planning concerns.",
        "clock_findings": {
            "number_placement": "Numbers are shifted to the right.",
            "hand_accuracy": "Hands do not match the target time.",
        },
        "cdt_features": {
            "number_distribution": "right_shifted",
            "number_spacing": "crowded",
            "target_time_match": False,
        },
    }

    merged_record = build_clock_assessment_record(
        clock_report,
        assessment_id=dialog_record["assessment_id"],
        existing_record=existing_record,
        created_at="2026-05-23T12:05:00.123+08:00",
        target_time="11:10",
    )
    save_session(merged_record, db_path=db_path)

    recent = get_recent_sessions(CURRENT_USER_ID, limit=3, db_path=db_path)

    assert len(recent) == 1
    assert recent[0]["session_id"] == dialog_record["assessment_id"]
    assert recent[0]["assessment_id"] == dialog_record["assessment_id"]
    assert set(recent[0]["components"]) == {"dialogue", "clock"}
    assert recent[0]["risk_level"] == "medium"
    assert recent[0]["domain_scores"]["executive_function"] == 0.6
    assert recent[0]["domain_scores"]["visuospatial"] == 0.5
    assert "dialogue_result" in recent[0]
    assert "clock_result" in recent[0]
    assert recent[0]["clock_result"]["target_time"] == "11:10"
    assert recent[0]["clock_result"]["source"] == "qwen-vl"
    assert recent[0]["clock_result"]["cdt_features"]["target_time_match"] is False
