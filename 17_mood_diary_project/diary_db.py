# 일기 DB 모듈
# - SQLite 일기 테이블 생성 / 저장 / 조회 / 수정 / 삭제

import os
import sqlite3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "diary.db")


# 1. 테이블 생성
def init_db():
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS diaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                diary_date TEXT NOT NULL,
                content TEXT NOT NULL,
                emotion TEXT NOT NULL,
                score REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now', 'localtime'))
            )
        """)

    except sqlite3.Error as e:
        print("DB 오류:", e)
        raise

    finally:
        if conn is not None:
            conn.close()


# 2. 일기 저장
def save_diary(diary_date, content, emotion, score):
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO diaries (diary_date, content, emotion, score) VALUES (?, ?, ?, ?)",
            (diary_date, content, emotion, score),
        )
        conn.commit()

    except sqlite3.Error as e:
        print("DB 오류:", e)
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()


# 3. 일기 조회 (emotion 이 없으면 전체 조회)
def load_diaries(emotion=None):
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        if not emotion:
            cur.execute("SELECT * FROM diaries")
        else:
            cur.execute("SELECT * FROM diaries WHERE emotion = ?", (emotion,))

        return cur.fetchall()

    except sqlite3.Error as e:
        print("DB 오류:", e)
        raise

    finally:
        if conn is not None:
            conn.close()


# 4. 일기 수정 (내용이 바뀌면 감정, 확률도 같이 바꿈)
def update_diary(diary_id, content, emotion, score):
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "UPDATE diaries SET content = ?, emotion = ?, score = ? WHERE id = ?",
            (content, emotion, score, diary_id),
        )
        conn.commit()

    except sqlite3.Error as e:
        print("DB 오류:", e)
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()


# 5. 일기 삭제
def delete_diary(diary_id):
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute("DELETE FROM diaries WHERE id = ?", (diary_id,))
        conn.commit()

    except sqlite3.Error as e:
        print("DB 오류:", e)
        if conn is not None:
            conn.rollback()
        raise

    finally:
        if conn is not None:
            conn.close()
