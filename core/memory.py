from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union

from core.config import AppConfig
from core.db import DATA_DIR
from core.embedding import embed_text
from core.schemas import COGNITIVE_DOMAINS


DEFAULT_CHROMA_DIR = DATA_DIR / "chroma"
DEFAULT_MEMORY_DIR = DATA_DIR / "memory"
MEMORY_FILENAME = "session_memories.json"
CHROMA_COLLECTION_NAME = "session_memories"
EMBEDDING_DIMENSIONS = 8
MIN_CHROMA_SQLITE_VERSION = (3, 35, 0)


@dataclass
class MemoryStore:
    backend: str
    storage_dir: Path
    json_path: Path
    collection: Optional[Any] = None
    fallback_dir: Optional[Path] = None
    items: list[dict[str, Any]] = field(default_factory=list)


def init_memory(
    storage_dir: Optional[Union[str, Path]] = None,
    prefer_chroma: bool = True,
) -> MemoryStore:
    if prefer_chroma:
        chroma_dir = Path(storage_dir) if storage_dir is not None else DEFAULT_CHROMA_DIR
        try:
            return _create_chroma_store(chroma_dir)
        except Exception:
            fallback_dir = _fallback_dir_for(chroma_dir, storage_dir)
            return _create_json_store(fallback_dir)

    memory_dir = Path(storage_dir) if storage_dir is not None else DEFAULT_MEMORY_DIR
    return _create_json_store(memory_dir)


def save_session_memory(
    record: dict[str, Any],
    store: Optional[MemoryStore] = None,
    config: Optional[AppConfig] = None,
) -> dict[str, Any]:
    active_store = store or init_memory()
    item = _build_memory_item(record, config=config)

    if active_store.backend == "chroma":
        try:
            _save_chroma_item(active_store, item)
            return _without_embedding(item, backend="chroma")
        except Exception:
            active_store = _create_json_store(
                active_store.fallback_dir or DEFAULT_MEMORY_DIR
            )

    if active_store.backend == "json":
        try:
            _save_json_item(active_store.json_path, item)
            return _without_embedding(item, backend="json")
        except Exception:
            active_store = MemoryStore(
                backend="memory",
                storage_dir=active_store.storage_dir,
                json_path=active_store.json_path,
            )

    active_store.items = _replace_item(active_store.items, item)
    return _without_embedding(item, backend="memory")


def query_recent_memories(
    user_id: str,
    limit: int = 3,
    store: Optional[MemoryStore] = None,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    active_store = store or init_memory()

    try:
        if active_store.backend == "chroma":
            try:
                items = _query_chroma_items(active_store, user_id)
            except Exception:
                fallback_store = _create_json_store(
                    active_store.fallback_dir or DEFAULT_MEMORY_DIR
                )
                items = _read_json_items(fallback_store.json_path)
        elif active_store.backend == "json":
            items = _read_json_items(active_store.json_path)
        else:
            items = active_store.items
    except Exception:
        return []

    filtered = [item for item in items if item.get("user_id") == user_id]
    filtered.sort(key=lambda item: (item.get("created_at", ""), item.get("session_id", "")))
    recent = filtered[-int(limit) :]
    return [_without_embedding(item, backend=item.get("backend")) for item in recent]


def build_session_summary(record: dict[str, Any]) -> str:
    session_id = record.get("session_id", "")
    user_id = record.get("user_id") or record.get("participant_id") or ""
    created_at = record.get("created_at", "")
    risk_level = record.get("risk_level", "unknown")
    explanation = record.get("explanation", "")

    scores = record.get("domain_scores") or {}
    score_parts = []
    for domain in COGNITIVE_DOMAINS:
        value = scores.get(domain)
        score_parts.append(f"{domain}={_format_score(value)}")

    evidence_parts = []
    for item in record.get("evidence") or []:
        if not isinstance(item, dict):
            continue
        domain = item.get("domain", "unknown")
        source = item.get("source", "unknown")
        text = item.get("text", "")
        evidence_parts.append(f"{domain}/{source}: {text}")

    return "\n".join(
        [
            f"session_id: {session_id}",
            f"user_id: {user_id}",
            f"created_at: {created_at}",
            f"risk_level: {risk_level}",
            "domain_scores: " + "; ".join(score_parts),
            "evidence: " + " | ".join(evidence_parts),
            f"explanation: {explanation}",
        ]
    )


def _create_chroma_store(chroma_dir: Path) -> MemoryStore:
    _ensure_chroma_sqlite()
    import chromadb  # type: ignore

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))
    collection = client.get_or_create_collection(name=CHROMA_COLLECTION_NAME)
    fallback_dir = chroma_dir.parent / "memory"
    return MemoryStore(
        backend="chroma",
        storage_dir=chroma_dir,
        json_path=fallback_dir / MEMORY_FILENAME,
        collection=collection,
        fallback_dir=fallback_dir,
    )


def _ensure_chroma_sqlite() -> None:
    import sqlite3

    if _sqlite_version_tuple(sqlite3.sqlite_version) >= MIN_CHROMA_SQLITE_VERSION:
        return

    try:
        import pysqlite3  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "ChromaDB requires sqlite3 >= 3.35.0. Install pysqlite3-binary "
            "in the project virtual environment or upgrade system SQLite."
        ) from exc

    sys.modules["sqlite3"] = pysqlite3
    import sqlite3 as patched_sqlite3

    if _sqlite_version_tuple(patched_sqlite3.sqlite_version) < MIN_CHROMA_SQLITE_VERSION:
        raise RuntimeError(
            "pysqlite3 is installed, but its bundled SQLite is still too old "
            "for ChromaDB."
        )


def _sqlite_version_tuple(version: str) -> tuple[int, int, int]:
    parts = []
    for piece in str(version).split("."):
        number = ""
        for character in piece:
            if not character.isdigit():
                break
            number += character
        parts.append(int(number or 0))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])  # type: ignore[return-value]


def _create_json_store(memory_dir: Path) -> MemoryStore:
    try:
        memory_dir.mkdir(parents=True, exist_ok=True)
        json_path = memory_dir / MEMORY_FILENAME
        if not json_path.exists():
            json_path.write_text("[]\n", encoding="utf-8")
        return MemoryStore(backend="json", storage_dir=memory_dir, json_path=json_path)
    except Exception:
        return MemoryStore(
            backend="memory",
            storage_dir=memory_dir,
            json_path=memory_dir / MEMORY_FILENAME,
        )


def _fallback_dir_for(chroma_dir: Path, storage_dir: Optional[Union[str, Path]]) -> Path:
    if storage_dir is None:
        return DEFAULT_MEMORY_DIR
    return chroma_dir.parent / "memory"


def _build_memory_item(
    record: dict[str, Any],
    config: Optional[AppConfig] = None,
) -> dict[str, Any]:
    session_id = _required_text(record, "session_id")
    user_id = record.get("user_id") or record.get("participant_id")
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("session record must include user_id or participant_id")

    created_at = _required_text(record, "created_at")
    risk_level = record.get("risk_level", "unknown")
    if not isinstance(risk_level, str) or not risk_level.strip():
        risk_level = "unknown"

    summary = build_session_summary(record)
    metadata = {
        "session_id": session_id,
        "user_id": user_id,
        "created_at": created_at,
        "risk_level": risk_level,
        "is_mock": bool(record.get("is_mock", False)),
        "trajectory": record.get("trajectory", ""),
    }

    return {
        "session_id": session_id,
        "user_id": user_id,
        "created_at": created_at,
        "risk_level": risk_level,
        "summary": summary,
        "metadata": metadata,
        "embedding": embed_text(summary, config=config, dimensions=EMBEDDING_DIMENSIONS),
    }


def _save_chroma_item(store: MemoryStore, item: dict[str, Any]) -> None:
    if store.collection is None:
        raise RuntimeError("Chroma collection is not initialized")

    store.collection.upsert(
        ids=[item["session_id"]],
        documents=[item["summary"]],
        embeddings=[item["embedding"]],
        metadatas=[item["metadata"]],
    )


def _query_chroma_items(store: MemoryStore, user_id: str) -> list[dict[str, Any]]:
    if store.collection is None:
        raise RuntimeError("Chroma collection is not initialized")

    result = store.collection.get(
        where={"user_id": user_id},
        include=["documents", "metadatas"],
    )
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    items = []
    for index, session_id in enumerate(ids):
        metadata = metadatas[index] if index < len(metadatas) else {}
        summary = documents[index] if index < len(documents) else ""
        items.append(
            {
                "session_id": session_id,
                "user_id": metadata.get("user_id", user_id),
                "created_at": metadata.get("created_at", ""),
                "risk_level": metadata.get("risk_level", "unknown"),
                "summary": summary,
                "metadata": metadata,
            }
        )
    return items


def _save_json_item(json_path: Path, item: dict[str, Any]) -> None:
    items = _read_json_items(json_path)
    items = _replace_item(items, item)
    json_path.write_text(
        json.dumps(items, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json_items(json_path: Path) -> list[dict[str, Any]]:
    if not json_path.exists():
        return []

    with json_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _replace_item(
    items: list[dict[str, Any]], item: dict[str, Any]
) -> list[dict[str, Any]]:
    replaced = [existing for existing in items if existing.get("session_id") != item["session_id"]]
    replaced.append(item)
    return replaced


def _without_embedding(
    item: dict[str, Any], backend: Optional[str] = None
) -> dict[str, Any]:
    public_item = {key: value for key, value in item.items() if key != "embedding"}
    if backend:
        public_item["backend"] = backend
    return public_item


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"session record must include {key}")
    return value


def _format_score(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, (float, int)):
        return f"{float(value):.2f}"
    return str(value)
