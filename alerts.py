"""
Simple alert logging stub.
Replace with Twilio/SendGrid integrations later.
"""
from pathlib import Path
from .db import get_conn

def log_alert(channel: str, recipient: str, subject: str, message: str):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO alerts(channel, recipient, subject, message) VALUES(?,?,?,?)",
        (channel, recipient, subject, message)
    )
    conn.commit()
    conn.close()
