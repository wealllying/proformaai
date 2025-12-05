# database.py - Database and user management
"""
Handles all database operations for user authentication.
Usage: from database import init_users_table, get_user, create_user
"""

import os
import logging
import hashlib

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    DATABASE_AVAILABLE = True
except:
    DATABASE_AVAILABLE = False

logger = logging.getLogger("proforma")

def get_db_connection():
    """Get database connection with error handling"""
    try:
        DATABASE_URL = os.getenv("DATABASE_URL")
        if not DATABASE_URL:
            logger.warning("DATABASE_URL not set")
            return None
        conn = psycopg2.connect(DATABASE_URL, sslmode='require')
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None

def init_users_table():
    """
    Initialize users table on first run.
    Call this once when deploying the app.
    """
    conn = get_db_connection()
    if not conn:
        logger.error("Cannot init table - no database connection")
        return False
    
    try:
        with conn.cursor() as cur:
            # Create users table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username VARCHAR(100) PRIMARY KEY,
                    password_hash VARCHAR(256) NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    plan VARCHAR(50) DEFAULT 'one',
                    last_login TIMESTAMP
                )
            """)
            
            # Create scenarios table for saving deals
            cur.execute("""
                CREATE TABLE IF NOT EXISTS scenarios (
                    id SERIAL PRIMARY KEY,
                    username VARCHAR(100) REFERENCES users(username),
                    scenario_name VARCHAR(200),
                    inputs JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add default admin user
            admin_hash = hashlib.sha256("proforma2025".encode()).hexdigest()
            cur.execute("""
                INSERT INTO users (username, password_hash, plan)
                VALUES (%s, %s, %s)
                ON CONFLICT (username) DO NOTHING
            """, ("admin", admin_hash, "unlimited"))
            
            conn.commit()
            logger.info("Database tables initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Table initialization failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def get_user(username):
    """
    Get user record from database.
    Returns dict with user info or None if not found.
    """
    conn = get_db_connection()
    if not conn:
        # Fallback for local development
        default_users = {
            "admin": {
                "username": "admin",
                "password_hash": hashlib.sha256("proforma2025".encode()).hexdigest(),
                "plan": "unlimited"
            }
        }
        return default_users.get(username)
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT username, password_hash, plan, created_at, last_login
                FROM users 
                WHERE username = %s
            """, (username,))
            user = cur.fetchone()
            
            # Update last login
            if user:
                cur.execute("""
                    UPDATE users 
                    SET last_login = CURRENT_TIMESTAMP 
                    WHERE username = %s
                """, (username,))
                conn.commit()
            
            return dict(user) if user else None
    except Exception as e:
        logger.error(f"Get user failed: {e}")
        return None
    finally:
        conn.close()

def create_user(username, password, plan="one"):
    """
    Create new user account.
    Returns True if successful, False if user exists or error.
    """
    conn = get_db_connection()
    if not conn:
        logger.warning("Cannot create user - no database connection")
        return False
    
    try:
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (username, password_hash, plan)
                VALUES (%s, %s, %s)
            """, (username, password_hash, plan))
            conn.commit()
        logger.info(f"User {username} created successfully")
        return True
    except psycopg2.errors.UniqueViolation:
        logger.warning(f"User {username} already exists")
        conn.rollback()
        return False
    except Exception as e:
        logger.error(f"Create user failed: {e}")
        conn.rollback()
        return False
    finally:
        conn.close()

def update_user_plan(username, new_plan):
    """Update user's subscription plan"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE users 
                SET plan = %s 
                WHERE username = %s
            """, (new_plan, username))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Update plan failed: {e}")
        return False
    finally:
        conn.close()

def save_scenario(username, scenario_name, inputs):
    """Save a deal scenario for later reference"""
    conn = get_db_connection()
    if not conn:
        return False
    
    try:
        import json
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO scenarios (username, scenario_name, inputs)
                VALUES (%s, %s, %s)
            """, (username, scenario_name, json.dumps(inputs)))
            conn.commit()
        return True
    except Exception as e:
        logger.error(f"Save scenario failed: {e}")
        return False
    finally:
        conn.close()

def get_user_scenarios(username):
    """Get all saved scenarios for a user"""
    conn = get_db_connection()
    if not conn:
        return []
    
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT id, scenario_name, inputs, created_at
                FROM scenarios
                WHERE username = %s
                ORDER BY created_at DESC
            """, (username,))
            return [dict(row) for row in cur.fetchall()]
    except Exception as e:
        logger.error(f"Get scenarios failed: {e}")
        return []
    finally:
        conn.close()

def hash_password(pw):
    """Hash password using SHA256"""
    return hashlib.sha256(pw.encode()).hexdigest()
