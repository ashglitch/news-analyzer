
import sqlite3
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / "verity.db"


def get_connection():
    connection = sqlite3.connect(str(DATABASE_PATH))
    connection.row_factory = sqlite3.Row
    return connection


def init_db():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                headline TEXT NOT NULL,
                article_text TEXT NOT NULL,
                level TEXT NOT NULL,
                score INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (
                    datetime('now', 'localtime')
                )
            )
        """)
        connection.commit()


def save_analysis(result):
    with get_connection() as connection:
        cursor = connection.execute(
            """
            INSERT INTO analyses (
                headline, article_text, level, score
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                result["headline"],
                result["article_text"][:5000],
                result["level"],
                result["score"],
            ),
        )
        connection.commit()
        return cursor.lastrowid


def get_history():
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT id, headline, level, score, created_at
            FROM analyses
            ORDER BY id DESC
            """
        ).fetchall()


def get_analysis(analysis_id):
    with get_connection() as connection:
        return connection.execute(
            """
            SELECT *
            FROM analyses
            WHERE id = ?
            """,
            (analysis_id,),
        ).fetchone()


def delete_analysis(analysis_id):
    with get_connection() as connection:
        connection.execute(
            "DELETE FROM analyses WHERE id = ?",
            (analysis_id,),
        )
        connection.commit()
