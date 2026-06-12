from __future__ import annotations

import copy

from core import memory
from core.mock_data import load_fixture_sessions


def test_build_session_summary_contains_key_fields() -> None:
    record = load_fixture_sessions("normal")[0]

    summary = memory.build_session_summary(record)

    assert record["session_id"] in summary
    assert record["participant_id"] in summary
    assert "risk_level: low" in summary
    assert "memory=" in summary
    assert "visuospatial=" in summary
    assert "evidence:" in summary
    assert record["explanation"] in summary


def test_save_session_memory_and_query_recent_with_json_store(tmp_path) -> None:
    store = memory.init_memory(storage_dir=tmp_path / "memory", prefer_chroma=False)
    sessions = load_fixture_sessions("normal")

    for record in sessions:
        saved = memory.save_session_memory(copy.deepcopy(record), store=store)
        assert saved["session_id"] == record["session_id"]
        assert saved["backend"] == "json"
        assert saved["summary"]
        assert "embedding" not in saved

    recent = memory.query_recent_memories(
        "demo-person-normal",
        limit=2,
        store=store,
    )

    assert [item["session_id"] for item in recent] == ["normal-002", "normal-003"]
    assert all(item["user_id"] == "demo-person-normal" for item in recent)


def test_chromadb_unavailable_falls_back_without_crashing(tmp_path, monkeypatch) -> None:
    def fail_chroma(_chroma_dir):
        raise RuntimeError("chromadb unavailable")

    monkeypatch.setattr(memory, "_create_chroma_store", fail_chroma)

    store = memory.init_memory(storage_dir=tmp_path / "chroma", prefer_chroma=True)
    record = copy.deepcopy(load_fixture_sessions("mild_decline")[0])

    saved = memory.save_session_memory(record, store=store)
    recent = memory.query_recent_memories(record["participant_id"], store=store)

    assert store.backend in {"json", "memory"}
    assert saved["session_id"] == record["session_id"]
    assert recent[0]["session_id"] == record["session_id"]


def test_query_recent_memories_returns_empty_on_store_failure(tmp_path, monkeypatch) -> None:
    store = memory.init_memory(storage_dir=tmp_path / "memory", prefer_chroma=False)

    def fail_read(_json_path):
        raise RuntimeError("broken memory store")

    monkeypatch.setattr(memory, "_read_json_items", fail_read)

    assert memory.query_recent_memories("demo-person-normal", store=store) == []
