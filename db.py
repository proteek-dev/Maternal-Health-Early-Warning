import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "app.db"

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS patients (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    age INTEGER NOT NULL,
    systolic_bp INTEGER NOT NULL,
    diastolic_bp INTEGER NOT NULL,
    hemoglobin REAL,
    gestational_weeks INTEGER,
    previous_preeclampsia INTEGER DEFAULT 0,
    risk_score REAL,
    risk_label TEXT
);
CREATE TABLE IF NOT EXISTS symptoms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    location TEXT NOT NULL,
    date TEXT NOT NULL, -- YYYY-MM-DD
    fever INTEGER DEFAULT 0,
    cough INTEGER DEFAULT 0,
    diarrhea INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_symptoms_location_date ON symptoms(location, date);
CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    channel TEXT NOT NULL, -- 'email' or 'sms' or 'log'
    recipient TEXT NOT NULL,
    subject TEXT,
    message TEXT
);
"""

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    cur = conn.cursor()
    for stmt in SCHEMA.strip().split(';'):
        s = stmt.strip()
        if s:
            cur.execute(s)
    conn.commit()
    conn.close()
