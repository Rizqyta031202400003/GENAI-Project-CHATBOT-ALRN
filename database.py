"""
database.py — SQLite
Tabel:
  - chat_history   : riwayat percakapan per sesi
  - unanswered_log : pertanyaan yang tidak ditemukan di knowledge base
  - feedback       : rating jawaban dari user
  - error_log      : log error runtime
"""
import sqlite3
import os
import logging
from datetime import datetime
from contextlib import contextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("database")

DB_PATH = os.path.join(os.path.dirname(__file__), "database.db")


@contextmanager
def get_connection():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        yield conn
        conn.commit()
    except sqlite3.OperationalError as e:
        logger.error(f"Database OperationalError: {e}")
        if conn: conn.rollback()
        raise
    except Exception as e:
        logger.error(f"Database error: {e}")
        if conn: conn.rollback()
        raise
    finally:
        if conn: conn.close()


def init_database():
    """Inisialisasi semua tabel. Dipanggil sekali saat startup."""
    try:
        with get_connection() as conn:
            c = conn.cursor()
            c.execute("""CREATE TABLE IF NOT EXISTS chat_history (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT    NOT NULL,
                role       TEXT    NOT NULL,
                message    TEXT    NOT NULL,
                timestamp  TEXT    NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS unanswered_log (
                id        INTEGER PRIMARY KEY AUTOINCREMENT,
                question  TEXT NOT NULL,
                timestamp TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS feedback (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                question   TEXT,
                answer     TEXT,
                rating     INTEGER,
                comment    TEXT,
                timestamp  TEXT NOT NULL
            )""")
            c.execute("""CREATE TABLE IF NOT EXISTS error_log (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                context    TEXT,
                error_type TEXT,
                message    TEXT,
                timestamp  TEXT NOT NULL
            )""")
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.critical(f"Gagal menginisialisasi database: {e}")
        raise


def save_chat(session_id: str, role: str, message: str) -> bool:
    """Simpan pesan chat ke chat_history."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO chat_history (session_id, role, message, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, role, message, datetime.now().isoformat())
            )
        return True
    except Exception as e:
        logger.error(f"save_chat gagal (session={session_id}): {e}")
        return False


def log_unanswered(question: str) -> bool:
    """Simpan pertanyaan yang tidak terjawab oleh RAG."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO unanswered_log (question, timestamp) VALUES (?, ?)",
                (question, datetime.now().isoformat())
            )
        return True
    except Exception as e:
        logger.error(f"log_unanswered gagal: {e}")
        return False


def save_feedback(session_id: str, question: str, answer: str,
                  rating: int, comment: str = "") -> bool:
    """Simpan feedback 👍/👎 dari user."""
    try:
        if rating not in (1, 2, 3, 4, 5):
            raise ValueError(f"Rating tidak valid: {rating}. Harus 1–5.")
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO feedback (session_id, question, answer, rating, comment, timestamp) VALUES (?,?,?,?,?,?)",
                (session_id, question, answer, rating, comment, datetime.now().isoformat())
            )
        return True
    except ValueError as e:
        logger.warning(f"save_feedback validasi gagal: {e}")
        return False
    except Exception as e:
        logger.error(f"save_feedback gagal: {e}")
        return False


def log_error(context: str, error: Exception) -> bool:
    """Catat error runtime ke error_log."""
    try:
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO error_log (context, error_type, message, timestamp) VALUES (?,?,?,?)",
                (context, type(error).__name__, str(error), datetime.now().isoformat())
            )
        return True
    except Exception as e:
        logger.error(f"log_error gagal: {e}")
        return False