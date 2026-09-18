"""
echo.storage
------------
Local, on-device storage (SQLite) for ECHO.

Privacy stance: by default only extracted FEATURES, DEVIATION SCORES, and
optional QUESTIONNAIRE ANSWERS are persisted -- never raw audio. This file
is the single place that touches the database, so the "what does ECHO
retain" question always has one clear, auditable answer.
"""

from __future__ import annotations
import json
import sqlite3
import time
from contextlib import contextmanager

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL,
    session_type TEXT NOT NULL,       -- 'calibration' | 'assessment'
    created_at REAL NOT NULL,
    features_json TEXT NOT NULL,      -- {task_key: {feature: value}}
    deviation_json TEXT,              -- output of deviation.score_session, null for calibration
    ewma_score REAL,
    questionnaire_json TEXT           -- null unless the alert questionnaire was completed
);

CREATE TABLE IF NOT EXISTS baseline_meta (
    user_id TEXT PRIMARY KEY,
    baseline_path TEXT NOT NULL,
    updated_at REAL NOT NULL
);
"""


class Storage:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ------------------------------------------------------------
    def add_session(self, user_id: str, session_type: str, features_by_task: dict,
                     deviation_result: dict | None = None, ewma_score: float | None = None,
                     questionnaire: dict | None = None, created_at: float | None = None) -> int:
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO sessions
                   (user_id, session_type, created_at, features_json, deviation_json,
                    ewma_score, questionnaire_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    user_id, session_type, created_at or time.time(),
                    json.dumps(features_by_task),
                    json.dumps(deviation_result) if deviation_result is not None else None,
                    ewma_score,
                    json.dumps(questionnaire) if questionnaire is not None else None,
                ),
            )
            return cur.lastrowid

    def attach_questionnaire(self, session_id: int, responses: dict) -> None:
        with self._conn() as conn:
            conn.execute(
                "UPDATE sessions SET questionnaire_json = ? WHERE id = ?",
                (json.dumps(responses), session_id),
            )

    def get_sessions(self, user_id: str, session_type: str | None = None) -> list[dict]:
        with self._conn() as conn:
            if session_type:
                rows = conn.execute(
                    "SELECT * FROM sessions WHERE user_id = ? AND session_type = ? ORDER BY created_at ASC",
                    (user_id, session_type),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM sessions WHERE user_id = ? ORDER BY created_at ASC",
                    (user_id,),
                ).fetchall()
            return [self._row_to_dict(r) for r in rows]

    def latest_ewma(self, user_id: str) -> float | None:
        with self._conn() as conn:
            row = conn.execute(
                """SELECT ewma_score FROM sessions
                   WHERE user_id = ? AND ewma_score IS NOT NULL
                   ORDER BY created_at DESC LIMIT 1""",
                (user_id,),
            ).fetchone()
            return row["ewma_score"] if row else None

    def set_baseline_meta(self, user_id: str, baseline_path: str) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO baseline_meta (user_id, baseline_path, updated_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET baseline_path=excluded.baseline_path,
                                                       updated_at=excluded.updated_at""",
                (user_id, baseline_path, time.time()),
            )

    def get_baseline_path(self, user_id: str) -> str | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT baseline_path FROM baseline_meta WHERE user_id = ?", (user_id,)
            ).fetchone()
            return row["baseline_path"] if row else None

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict:
        d = dict(row)
        d["features_json"] = json.loads(d["features_json"])
        d["deviation_json"] = json.loads(d["deviation_json"]) if d["deviation_json"] else None
        d["questionnaire_json"] = json.loads(d["questionnaire_json"]) if d["questionnaire_json"] else None
        return d
