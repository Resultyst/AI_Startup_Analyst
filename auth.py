import sqlite3
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Cookie
import hashlib
from jose import JWTError, jwt
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return hash_password(plain_password) == hashed_password

SECRET_KEY = "iD-WadfdROQL5BOsvvX9eBEVR5yRUiY-lRy9Zb48sWQ"  
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 5

def init_db():
    """Initialize the SQLite database for users"""
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            hashed_password TEXT NOT NULL,
            is_admin BOOLEAN DEFAULT 0,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    admin_password = hash_password("admin123")
    cursor.execute('''
        INSERT OR IGNORE INTO users (username, email, hashed_password, is_admin)
        VALUES (?, ?, ?, ?)
    ''', ("admin", "admin@ai-analyst.com", admin_password, 1))
    
    conn.commit()
    conn.close()
    logger.info("Database initialized with admin user")

def get_db():
    conn = sqlite3.connect('users.db')
    conn.row_factory = sqlite3.Row
    return conn

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    logger.debug(f"Created access token for {data.get('sub')}")
    return encoded_jwt

def get_current_user(token: str = Cookie(None)):
    logger.debug(f"get_current_user called with token: {token}")
    if token is None:
        logger.debug("No token found in cookie")
        return None
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            logger.debug("No username in token payload")
            return None
        logger.debug(f"Authenticated user: {username}")
        return username
    except JWTError as e:
        logger.debug(f"JWT error: {e}")
        return None

init_db()