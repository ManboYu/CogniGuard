from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional, Union

from core.schemas import COGNITIVE_DOMAINS, RISK_LEVELS, normalize_domain_scores


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_DB_PATH = DATA_DIR / "cogniguard.db"
PASSWORD_HASH_SCHEME = "pbkdf2_sha256"
PASSWORD_HASH_ITERATIONS = 120_000

DEMO_USERS = (
    {
        "user_id": "zhang-nainai",
        "username": "zhang",
        "password": "123456",
        "display_name": "张奶奶",
        "profile_type": "elder_demo",
    },
    {
        "user_id": "wang-shushu",
        "username": "wang",
        "password": "123456",
        "display_name": "王叔叔",
        "profile_type": "elder_demo",
    },
    {
        "user_id": "li-ayi",
        "username": "li",
        "password": "123456",
        "display_name": "李阿姨",
        "profile_type": "elder_demo",
    },
)


def init_db(db_path: Optional[Union[str, Path]] = None) -> Path:
    path = _resolve_db_path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with _connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                domain_scores_json TEXT NOT NULL,
                evidence_json TEXT NOT NULL,
                explanation TEXT NOT NULL,
                raw_json TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_sessions_user_created_at
            ON sessions (user_id, created_at DESC)
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL UNIQUE,
                password_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                profile_type TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_users_username
            ON users (username)
            """
        )
        _ensure_demo_users(connection)

    return path


def hash_password(password: str, salt: Optional[str] = None) -> str:
    normalized_password = str(password or "")
    password_salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256",
        normalized_password.encode("utf-8"),
        password_salt.encode("utf-8"),
        PASSWORD_HASH_ITERATIONS,
    ).hex()
    return (
        f"{PASSWORD_HASH_SCHEME}${PASSWORD_HASH_ITERATIONS}"
        f"${password_salt}${digest}"
    )


def verify_password(password: str, password_hash: str) -> bool:
    try:
        scheme, iterations_text, salt, expected_digest = str(password_hash).split("$", 3)
        iterations = int(iterations_text)
    except (TypeError, ValueError):
        return False
    if scheme != PASSWORD_HASH_SCHEME or not salt or not expected_digest:
        return False

    actual_digest = hashlib.pbkdf2_hmac(
        "sha256",
        str(password or "").encode("utf-8"),
        salt.encode("utf-8"),
        iterations,
    ).hex()
    return hmac.compare_digest(actual_digest, expected_digest)


def list_demo_users(
    db_path: Optional[Union[str, Path]] = None,
) -> list[dict[str, Any]]:
    path = init_db(db_path)
    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT user_id, username, display_name, profile_type, created_at
            FROM users
            ORDER BY username ASC
            """
        ).fetchall()
    users = [_row_to_user(row) for row in rows]
    demo_order = {user["username"]: index for index, user in enumerate(DEMO_USERS)}
    return sorted(users, key=lambda user: demo_order.get(user["username"], len(DEMO_USERS)))


def get_user_by_id(
    user_id: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[dict[str, Any]]:
    normalized_user_id = str(user_id or "").strip()
    if not normalized_user_id:
        return None

    path = init_db(db_path)
    with _connect(path) as connection:
        row = connection.execute(
            """
            SELECT user_id, username, display_name, profile_type, created_at
            FROM users
            WHERE user_id = ?
            """,
            (normalized_user_id,),
        ).fetchone()
    return _row_to_user(row) if row else None


def authenticate_user(
    username: str,
    password: str,
    db_path: Optional[Union[str, Path]] = None,
) -> Optional[dict[str, Any]]:
    normalized_username = str(username or "").strip()
    if not normalized_username:
        return None

    path = init_db(db_path)
    with _connect(path) as connection:
        row = connection.execute(
            """
            SELECT user_id, username, password_hash, display_name, profile_type, created_at
            FROM users
            WHERE username = ?
            """,
            (normalized_username,),
        ).fetchone()
    if row is None or not verify_password(password, row["password_hash"]):
        return None
    return _row_to_user(row)


def save_session(
    record: dict[str, Any], db_path: Optional[Union[str, Path]] = None
) -> dict[str, Any]:
    path = init_db(db_path)
    normalized = _normalize_session_record(record)

    with _connect(path) as connection:
        connection.execute(
            """
            INSERT INTO sessions (
                session_id,
                user_id,
                created_at,
                risk_level,
                domain_scores_json,
                evidence_json,
                explanation,
                raw_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                user_id = excluded.user_id,
                created_at = excluded.created_at,
                risk_level = excluded.risk_level,
                domain_scores_json = excluded.domain_scores_json,
                evidence_json = excluded.evidence_json,
                explanation = excluded.explanation,
                raw_json = excluded.raw_json
            """,
            (
                normalized["session_id"],
                normalized["user_id"],
                normalized["created_at"],
                normalized["risk_level"],
                _json_dumps(normalized["domain_scores"]),
                _json_dumps(normalized["evidence"]),
                normalized["explanation"],
                _json_dumps(normalized),
            ),
        )

    return normalized


def list_sessions(
    user_id: Optional[str] = None, db_path: Optional[Union[str, Path]] = None
) -> list[dict[str, Any]]:
    path = init_db(db_path)

    with _connect(path) as connection:
        if user_id is None:
            rows = connection.execute(
                """
                SELECT *
                FROM sessions
                ORDER BY created_at ASC, session_id ASC
                """
            ).fetchall()
        else:
            rows = connection.execute(
                """
                SELECT *
                FROM sessions
                WHERE user_id = ?
                ORDER BY created_at ASC, session_id ASC
                """,
                (user_id,),
            ).fetchall()

    return [_row_to_session(row) for row in rows]


def get_recent_sessions(
    user_id: str, limit: int = 3, db_path: Optional[Union[str, Path]] = None
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []

    path = init_db(db_path)

    with _connect(path) as connection:
        rows = connection.execute(
            """
            SELECT *
            FROM sessions
            WHERE user_id = ?
            ORDER BY created_at DESC, session_id DESC
            LIMIT ?
            """,
            (user_id, int(limit)),
        ).fetchall()

    return [_row_to_session(row) for row in reversed(rows)]


def _resolve_db_path(db_path: Optional[Union[str, Path]]) -> Path:
    if db_path is None:
        return DEFAULT_DB_PATH
    return Path(db_path)


def _connect(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


def _ensure_demo_users(connection: sqlite3.Connection) -> None:
    existing_usernames = {
        row["username"]
        for row in connection.execute("SELECT username FROM users").fetchall()
    }
    created_at = _now_iso()
    for user in DEMO_USERS:
        if user["username"] in existing_usernames:
            continue
        connection.execute(
            """
            INSERT INTO users (
                user_id,
                username,
                password_hash,
                display_name,
                profile_type,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                user["user_id"],
                user["username"],
                hash_password(user["password"]),
                user["display_name"],
                user["profile_type"],
                created_at,
            ),
        )


def _row_to_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "user_id": row["user_id"],
        "username": row["username"],
        "display_name": row["display_name"],
        "profile_type": row["profile_type"],
        "created_at": row["created_at"],
    }


def _now_iso() -> str:
    china_tz = timezone(timedelta(hours=8))
    return datetime.now(china_tz).isoformat(timespec="seconds")


def _normalize_session_record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise TypeError("session record must be a dict")

    session_id = _required_text(record, "session_id")
    user_id = record.get("user_id") or record.get("participant_id")
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("session record must include user_id or participant_id")

    created_at = _required_text(record, "created_at")
    risk_level = _required_text(record, "risk_level")
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"invalid risk_level: {risk_level}")

    domain_scores = record.get("domain_scores")
    if not isinstance(domain_scores, dict):
        raise ValueError("session record must include domain_scores")
    if set(domain_scores) != set(COGNITIVE_DOMAINS):
        raise ValueError("domain_scores must include exactly the known domains")

    evidence = record.get("evidence")
    if not isinstance(evidence, list):
        raise ValueError("session record must include evidence as a list")

    explanation = _required_text(record, "explanation")

    normalized = dict(record)
    normalized["session_id"] = session_id
    normalized["user_id"] = user_id
    normalized.setdefault("participant_id", user_id)
    normalized["created_at"] = created_at
    normalized["risk_level"] = risk_level
    normalized["domain_scores"] = normalize_domain_scores(domain_scores)
    normalized["evidence"] = evidence
    normalized["explanation"] = explanation
    return normalized


def _required_text(record: dict[str, Any], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"session record must include {key}")
    return value


def _row_to_session(row: sqlite3.Row) -> dict[str, Any]:
    record = json.loads(row["raw_json"])
    record["session_id"] = row["session_id"]
    record["user_id"] = row["user_id"]
    record.setdefault("participant_id", row["user_id"])
    record["created_at"] = row["created_at"]
    record["risk_level"] = row["risk_level"]
    record["domain_scores"] = json.loads(row["domain_scores_json"])
    record["evidence"] = json.loads(row["evidence_json"])
    record["explanation"] = row["explanation"]
    return record


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
