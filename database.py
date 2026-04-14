import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = "nids.db"

# ══════════════════════════════════════════════════════════
# DATABASE INITIALIZATION
# ══════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    c    = conn.cursor()

    # Users table
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    UNIQUE NOT NULL,
            email      TEXT    UNIQUE NOT NULL,
            password   TEXT    NOT NULL,
            role       TEXT    DEFAULT 'user',
            created_at TEXT    DEFAULT (datetime('now'))
        )
    """)

    # Scans table
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id      INTEGER NOT NULL,
            username     TEXT    NOT NULL,
            scan_type    TEXT    NOT NULL,
            total        INTEGER DEFAULT 0,
            attacks      INTEGER DEFAULT 0,
            normal       INTEGER DEFAULT 0,
            risk_percent REAL    DEFAULT 0.0,
            timestamp    TEXT    DEFAULT (datetime('now')),
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)

    # Create default admin account
    admin_pass = hash_password("admin123")
    c.execute("""
        INSERT OR IGNORE INTO users (username, email, password, role)
        VALUES (?, ?, ?, ?)
    """, ("admin", "admin@nids.ai", admin_pass, "admin"))

    conn.commit()
    conn.close()

# ══════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def get_conn():
    return sqlite3.connect(DB_PATH)

# ══════════════════════════════════════════════════════════
# USER FUNCTIONS
# ══════════════════════════════════════════════════════════
def register_user(username, email, password):
    try:
        conn = get_conn()
        c    = conn.cursor()
        c.execute(
            "INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
            (username.strip(), email.strip(), hash_password(password))
        )
        conn.commit()
        conn.close()
        return True, "Account created successfully!"
    except sqlite3.IntegrityError:
        return False, "Username or email already exists!"

def login_user(username, password):
    conn = get_conn()
    c    = conn.cursor()
    c.execute(
        "SELECT id, username, email, role FROM users WHERE username=? AND password=?",
        (username.strip(), hash_password(password))
    )
    user = c.fetchone()
    conn.close()
    if user:
        return True, {"id": user[0], "username": user[1],
                      "email": user[2], "role": user[3]}
    return False, "Invalid username or password!"

def get_all_users():
    conn = get_conn()
    c    = conn.cursor()
    c.execute("SELECT id, username, email, role, created_at FROM users ORDER BY id DESC")
    users = c.fetchall()
    conn.close()
    return users

# ══════════════════════════════════════════════════════════
# SCAN FUNCTIONS
# ══════════════════════════════════════════════════════════
def save_scan(user_id, username, scan_type, total, attacks, normal, risk_percent):
    conn = get_conn()
    c    = conn.cursor()
    c.execute("""
        INSERT INTO scans (user_id, username, scan_type, total, attacks, normal, risk_percent)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (user_id, username, scan_type, total, attacks, normal, round(risk_percent, 2)))
    conn.commit()
    conn.close()

def get_user_scans(user_id):
    conn = get_conn()
    c    = conn.cursor()
    c.execute("""
        SELECT scan_type, total, attacks, normal, risk_percent, timestamp
        FROM scans WHERE user_id=? ORDER BY id DESC
    """, (user_id,))
    scans = c.fetchall()
    conn.close()
    return scans

def get_user_stats(user_id):
    conn = get_conn()
    c    = conn.cursor()
    c.execute("""
        SELECT COUNT(*), SUM(total), SUM(attacks), SUM(normal), AVG(risk_percent)
        FROM scans WHERE user_id=?
    """, (user_id,))
    stats = c.fetchone()
    conn.close()
    return {
        "total_scans"   : stats[0] or 0,
        "total_conn"    : stats[1] or 0,
        "total_attacks" : stats[2] or 0,
        "total_normal"  : stats[3] or 0,
        "avg_risk"      : round(stats[4] or 0, 1)
    }

def get_all_scans():
    conn = get_conn()
    c    = conn.cursor()
    c.execute("""
        SELECT username, scan_type, total, attacks, risk_percent, timestamp
        FROM scans ORDER BY id DESC
    """)
    scans = c.fetchall()
    conn.close()
    return scans

def get_global_stats():
    conn = get_conn()
    c    = conn.cursor()
    c.execute("SELECT COUNT(*) FROM users")
    total_users = c.fetchone()[0]
    c.execute("SELECT COUNT(*), SUM(total), SUM(attacks), AVG(risk_percent) FROM scans")
    s = c.fetchone()
    conn.close()
    return {
        "total_users"  : total_users,
        "total_scans"  : s[0] or 0,
        "total_conn"   : s[1] or 0,
        "total_attacks": s[2] or 0,
        "avg_risk"     : round(s[3] or 0, 1)
    }

print("✅ database.py ready!")