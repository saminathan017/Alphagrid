"""
core/auth_db.py  —  AlphaGrid v7
==================================
Authentication & Authorization.

Owner account:
  username : admin
  password : Admin@Grid1
  role     : ADMIN (full access)
  is_owner : True  (cannot be deactivated, cannot be deleted)

Owner login is special — they can sign in with just their USERNAME
instead of an email address. All other accounts require email.

Login resolution order:
  1. Try username match (case-insensitive)
  2. Try email match (case-insensitive)
  → first match wins

Stack: SQLAlchemy + SQLite · passlib[bcrypt] · python-jose JWT

Install:
  pip install passlib[bcrypt] python-jose[cryptography] sqlalchemy
"""
from __future__ import annotations

import os, secrets, uuid
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple
from loguru import logger

from sqlalchemy import (
    Boolean, Column, DateTime, ForeignKey, Index,
    Integer, String, Text, create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

# ── JWT ───────────────────────────────────────────────────────────────────────
try:
    from jose import JWTError, jwt as _jose_jwt
    JOSE_OK = True
except Exception:
    JOSE_OK = False
    logger.warning("python-jose not installed — run: pip install python-jose[cryptography]")

# ── bcrypt ────────────────────────────────────────────────────────────────────
try:
    from passlib.context import CryptContext
    _pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)
    PASSLIB_OK = True
except Exception:
    PASSLIB_OK = False
    _pwd_ctx   = None
    logger.warning("passlib/bcrypt unavailable — install: passlib[bcrypt] bcrypt<4.0")


# ── Constants ─────────────────────────────────────────────────────────────────
JWT_SECRET    = os.getenv("ALPHAGRID_JWT_SECRET", secrets.token_hex(32))
JWT_ALGO      = "HS256"
ACCESS_MIN    = 60 * 24         # 24 h
REFRESH_MIN   = 60 * 24 * 30   # 30 days
DB_PATH       = Path(__file__).parent.parent / "alphagrid_auth.db"

# ── Owner defaults — override via environment variables in .env ───────────────
OWNER_USERNAME  = os.getenv("ALPHAGRID_OWNER_USERNAME", "admin")
OWNER_PASSWORD  = os.getenv("ALPHAGRID_OWNER_PASSWORD", "Admin@Grid1")
OWNER_EMAIL     = os.getenv("ALPHAGRID_OWNER_EMAIL",    "owner@alphagrid.app")
OWNER_NAME      = "Owner"


# ── Roles ─────────────────────────────────────────────────────────────────────
class UserRole(str, Enum):
    TRADER  = "trader"
    BUILDER = "builder"
    ADMIN   = "admin"

    @property
    def display(self) -> str:
        return {"trader": "Trader", "builder": "Builder", "admin": "Admin"}[self.value]

    @property
    def color(self) -> str:
        return {"trader": "#00b4ff", "builder": "#00ffc8", "admin": "#9d5cff"}[self.value]

# Backward-compat alias
Role = UserRole


# ── ORM ───────────────────────────────────────────────────────────────────────
class AuthBase(DeclarativeBase):
    pass


class User(AuthBase):
    __tablename__ = "users"

    id              = Column(String(36),  primary_key=True, default=lambda: str(uuid.uuid4()))
    # Username — required for owner, optional for others (defaults to email prefix)
    username        = Column(String(80),  unique=True, nullable=False)
    email           = Column(String(254), unique=True, nullable=False)
    display_name    = Column(String(80),  nullable=False)
    hashed_password = Column(String(255), nullable=False)
    role            = Column(String(20),  nullable=False, default=UserRole.TRADER.value)
    is_active       = Column(Boolean,     default=True,  nullable=False)
    is_owner        = Column(Boolean,     default=False, nullable=False)  # owner cannot be deactivated
    avatar_initials = Column(String(3),   nullable=True)
    # Preferences stored as JSON string
    preferences     = Column(Text,        nullable=True)
    created_at      = Column(DateTime,    default=datetime.utcnow)
    last_login_at   = Column(DateTime,    nullable=True)
    login_count     = Column(Integer,     default=0)

    __table_args__ = (
        Index("ix_users_email",    "email"),
        Index("ix_users_username", "username"),
    )

    def to_dict(self) -> dict:
        return {
            "id":             self.id,
            "username":       self.username,
            "email":          self.email,
            "display_name":   self.display_name,
            "role":           self.role,
            "is_active":      self.is_active,
            "is_owner":       self.is_owner,
            "avatar_initials":self.avatar_initials,
            "created_at":     self.created_at.isoformat() if self.created_at else None,
            "last_login_at":  self.last_login_at.isoformat() if self.last_login_at else None,
            "login_count":    self.login_count,
        }

    # Public-safe view (no hashed_password, used in API responses)
    def to_public(self) -> dict:
        return {
            "id":             self.id,
            "username":       self.username,
            "email":          self.email,
            "display_name":   self.display_name,
            "role":           self.role,
            "is_owner":       self.is_owner,
            "avatar_initials":self.avatar_initials,
            "login_count":    self.login_count,
        }


class UserSession(AuthBase):
    __tablename__ = "user_sessions"

    id            = Column(String(36),  primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id       = Column(String(36),  ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    refresh_token = Column(String(512), unique=True, nullable=False, index=True)
    ip_address    = Column(String(64),  nullable=True)
    user_agent    = Column(String(256), nullable=True)
    created_at    = Column(DateTime,    default=datetime.utcnow)
    expires_at    = Column(DateTime,    nullable=False)
    revoked       = Column(Boolean,     default=False)
    revoked_at    = Column(DateTime,    nullable=True)

    __table_args__ = (Index("ix_sessions_uid", "user_id"),)


class AuditLog(AuthBase):
    __tablename__ = "audit_log"

    id         = Column(Integer,     primary_key=True, autoincrement=True)
    user_id    = Column(String(36),  nullable=True)
    username   = Column(String(80),  nullable=True)
    email      = Column(String(254), nullable=True)
    action     = Column(String(60),  nullable=False)
    ip_address = Column(String(64),  nullable=True)
    detail     = Column(Text,        nullable=True)
    success    = Column(Boolean,     default=True)
    timestamp  = Column(DateTime,    default=datetime.utcnow, nullable=False)

    __table_args__ = (Index("ix_audit_ts", "timestamp"),)


# ── DB engine ─────────────────────────────────────────────────────────────────
def _make_engine():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    eng = create_engine(
        f"sqlite:///{DB_PATH}",
        connect_args={"check_same_thread": False},
        echo=False,
    )
    AuthBase.metadata.create_all(eng)
    return eng

_auth_engine = _make_engine()
_AuthSession  = sessionmaker(bind=_auth_engine, expire_on_commit=False)

def get_auth_session() -> Session:
    return _AuthSession()


# ── Password helpers ──────────────────────────────────────────────────────────

def _hash_password(plain: str) -> str:
    if PASSLIB_OK:
        return _pwd_ctx.hash(plain)
    import hashlib
    return "sha256:" + hashlib.sha256(plain.encode()).hexdigest()


def _verify_password(plain: str, hashed: str) -> bool:
    if PASSLIB_OK:
        try:
            return _pwd_ctx.verify(plain, hashed)
        except Exception:
            return False
    import hashlib
    return hashed == "sha256:" + hashlib.sha256(plain.encode()).hexdigest()


# ── JWT helpers ───────────────────────────────────────────────────────────────

def _create_access_token(user: User) -> str:
    payload = {
        "sub":      user.id,
        "username": user.username,
        "email":    user.email,
        "name":     user.display_name,
        "role":     user.role,
        "avatar":   user.avatar_initials or "",
        "is_owner": user.is_owner,
        "exp":      datetime.utcnow() + timedelta(minutes=ACCESS_MIN),
        "type":     "access",
    }
    if JOSE_OK:
        return _jose_jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)
    import base64, json
    return base64.urlsafe_b64encode(
        json.dumps(payload, default=str).encode()
    ).decode()


def decode_access_token(token: str) -> Optional[dict]:
    """Decode and verify a JWT access token. Returns payload dict or None."""
    if JOSE_OK:
        try:
            return _jose_jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        except JWTError:
            return None
    try:
        import base64, json
        raw = token + "=" * (-len(token) % 4)
        return json.loads(base64.urlsafe_b64decode(raw))
    except Exception:
        return None


def _create_refresh_token() -> str:
    return secrets.token_urlsafe(64)


# ── UserManager — singleton class ─────────────────────────────────────────────

class _UserManager:
    """
    Central user management service.
    Single instance exposed as module-level `user_manager`.

    Login accepts:
      - username  (exact match, case-insensitive) → any account
      - email     (exact match, case-insensitive) → any account

    Owner account:
      username : admin
      password : Admin@Grid1
      Can only be looked up by username — no email required on login page.
    """

    # ── Lookup ─────────────────────────────────────────────────────────────

    def get_by_id(self, user_id: str) -> Optional[User]:
        with get_auth_session() as s:
            return s.query(User).filter_by(id=user_id).first()

    def get_by_email(self, email: str) -> Optional[User]:
        with get_auth_session() as s:
            return s.query(User).filter(
                User.email == email.lower().strip()
            ).first()

    def get_by_username(self, username: str) -> Optional[User]:
        with get_auth_session() as s:
            return s.query(User).filter(
                User.username == username.lower().strip()
            ).first()

    def get_owner(self) -> Optional[User]:
        with get_auth_session() as s:
            return s.query(User).filter_by(is_owner=True).first()

    # ── Resolve identifier (username OR email) ─────────────────────────────

    def resolve(self, identifier: str) -> Optional[User]:
        """
        Resolve login identifier to a User.
        Tries username first, then email.
        Case-insensitive for both.
        """
        ident = identifier.strip()
        # Try username
        u = self.get_by_username(ident)
        if u:
            return u
        # Try email (only if it looks like an email)
        if "@" in ident:
            return self.get_by_email(ident)
        return None

    # ── Create ─────────────────────────────────────────────────────────────

    def create_user(
        self,
        email:        str,
        password:     str,
        display_name: str,
        role:         UserRole = UserRole.TRADER,
        username:     str      = "",
        first_name:   str      = "",
        last_name:    str      = "",
        is_owner:     bool     = False,
    ) -> Tuple[Optional[User], str]:
        """
        Create a new user account.
        Returns (User, "") on success or (None, error_message) on failure.
        """
        email = email.lower().strip()
        if not email or "@" not in email:
            return None, "Invalid email address"
        if len(password) < 8:
            return None, "Password must be at least 8 characters"

        # Derive display name
        if not display_name.strip():
            if first_name or last_name:
                display_name = f"{first_name} {last_name}".strip()
            else:
                display_name = email.split("@")[0].title()
        display_name = display_name.strip()

        # Derive username from email prefix if not provided
        if not username.strip():
            base = email.split("@")[0].lower().replace(".", "_")
            username = base
            # Make unique if taken
            with get_auth_session() as s:
                n = 1
                while s.query(User).filter_by(username=username).first():
                    username = f"{base}{n}"
                    n += 1
        else:
            username = username.lower().strip()

        # Uniqueness checks
        if self.get_by_email(email):
            return None, "An account with this email already exists"
        if self.get_by_username(username):
            return None, f"Username '{username}' is already taken"

        initials = "".join(
            w[0].upper() for w in display_name.split()[:2]
        ) or display_name[0].upper()

        user = User(
            username        = username,
            email           = email,
            display_name    = display_name,
            hashed_password = _hash_password(password),
            role            = role.value,
            is_active       = True,
            is_owner        = is_owner,
            avatar_initials = initials,
        )
        with get_auth_session() as s:
            s.add(user)
            s.commit()
            s.refresh(user)
            logger.info(
                f"User created: @{username} <{email}> "
                f"role={role.value} owner={is_owner}"
            )
            return user, ""

    # ── Authenticate ───────────────────────────────────────────────────────

    def authenticate(
        self,
        identifier: str,
        password:   str,
        ip:         str = "",
        ua:         str = "",
    ) -> Tuple[Optional[User], str]:
        """
        Verify credentials.
        identifier can be a username OR email address.
        Returns (User, "") on success or (None, error_message) on failure.
        """
        user = self.resolve(identifier)

        if not user:
            # Give a generic message — don't reveal whether username/email exists
            return None, "Invalid username or password"

        if not user.is_active:
            return None, "Account is deactivated. Contact support."

        if not _verify_password(password, user.hashed_password):
            self._write_audit("login_fail", False,
                              user_id=user.id, username=user.username,
                              email=user.email, ip=ip, detail="wrong password")
            return None, "Invalid username or password"

        # Update login stats
        with get_auth_session() as s:
            u = s.query(User).filter_by(id=user.id).first()
            if u:
                u.last_login_at = datetime.utcnow()
                u.login_count   = (u.login_count or 0) + 1
                s.commit()

        self._write_audit("login", True,
                          user_id=user.id, username=user.username,
                          email=user.email, ip=ip)
        return user, ""

    # ── Sessions ───────────────────────────────────────────────────────────

    def create_session(
        self,
        user_id: str,
        ip:      str = "",
        ua:      str = "",
    ) -> Tuple[str, str]:
        """
        Create access + refresh token pair.
        Returns (access_token, refresh_token).
        """
        user = self.get_by_id(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")
        access  = _create_access_token(user)
        refresh = _create_refresh_token()
        sess = UserSession(
            user_id       = user_id,
            refresh_token = refresh,
            ip_address    = ip[:64] if ip else "",
            user_agent    = ua[:255] if ua else "",
            expires_at    = datetime.utcnow() + timedelta(minutes=REFRESH_MIN),
        )
        with get_auth_session() as s:
            s.add(sess)
            s.commit()
        return access, refresh

    def revoke_session(self, refresh_token: str) -> bool:
        with get_auth_session() as s:
            sess = s.query(UserSession).filter_by(
                refresh_token=refresh_token
            ).first()
            if sess:
                sess.revoked    = True
                sess.revoked_at = datetime.utcnow()
                s.commit()
                return True
        return False

    def revoke_all_sessions(self, user_id: str) -> int:
        with get_auth_session() as s:
            rows = s.query(UserSession).filter_by(
                user_id=user_id, revoked=False
            ).all()
            for r in rows:
                r.revoked    = True
                r.revoked_at = datetime.utcnow()
            s.commit()
            return len(rows)

    def refresh_access_token(
        self, refresh_token: str
    ) -> Tuple[Optional[str], Optional[User]]:
        """
        Exchange a valid refresh token for a new access token.
        Returns (access_token, user) or (None, None).
        """
        with get_auth_session() as s:
            sess = s.query(UserSession).filter_by(
                refresh_token=refresh_token, revoked=False
            ).first()
            if not sess or sess.expires_at < datetime.utcnow():
                return None, None
            user = s.query(User).filter_by(
                id=sess.user_id, is_active=True
            ).first()
            if not user:
                return None, None
            access = _create_access_token(user)
            return access, user

    def get_user_by_token(self, access_token: str) -> Optional[User]:
        payload = decode_access_token(access_token)
        if not payload:
            return None
        return self.get_by_id(payload.get("sub", ""))

    # ── Password change ────────────────────────────────────────────────────

    def change_password(
        self,
        user_id:          str,
        current_password: str,
        new_password:     str,
    ) -> Tuple[bool, str]:
        if len(new_password) < 8:
            return False, "New password must be at least 8 characters"
        user = self.get_by_id(user_id)
        if not user:
            return False, "User not found"
        if not _verify_password(current_password, user.hashed_password):
            return False, "Current password is incorrect"
        with get_auth_session() as s:
            u = s.query(User).filter_by(id=user_id).first()
            if u:
                u.hashed_password = _hash_password(new_password)
                s.commit()
        self.revoke_all_sessions(user_id)
        self._write_audit("password_change", True,
                          user_id=user_id, username=user.username)
        return True, ""

    # ── Admin ──────────────────────────────────────────────────────────────

    def list_users(self) -> list[dict]:
        with get_auth_session() as s:
            return [
                u.to_public()
                for u in s.query(User).order_by(User.created_at).all()
            ]

    def deactivate(self, user_id: str) -> Tuple[bool, str]:
        user = self.get_by_id(user_id)
        if not user:
            return False, "User not found"
        if user.is_owner:
            return False, "The owner account cannot be deactivated"
        with get_auth_session() as s:
            u = s.query(User).filter_by(id=user_id).first()
            if u:
                u.is_active = False
                s.commit()
        self.revoke_all_sessions(user_id)
        return True, ""

    def update_preferences(self, user_id: str, prefs: dict) -> bool:
        import json
        with get_auth_session() as s:
            u = s.query(User).filter_by(id=user_id).first()
            if not u:
                return False
            try:
                existing = json.loads(u.preferences or "{}")
                existing.update(prefs)
                u.preferences = json.dumps(existing)
                s.commit()
                return True
            except Exception:
                return False

    def get_audit_log(self, limit: int = 100) -> list[dict]:
        with get_auth_session() as s:
            rows = s.query(AuditLog).order_by(
                AuditLog.timestamp.desc()
            ).limit(limit).all()
            return [
                {
                    "id":        r.id,
                    "user_id":   r.user_id,
                    "username":  r.username,
                    "email":     r.email,
                    "action":    r.action,
                    "ip":        r.ip_address,
                    "success":   r.success,
                    "detail":    r.detail,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                }
                for r in rows
            ]

    # ── Internal audit writer ──────────────────────────────────────────────

    def _write_audit(
        self,
        action:   str,
        success:  bool     = True,
        user_id:  str      = None,
        username: str      = None,
        email:    str      = None,
        ip:       str      = "",
        detail:   str      = "",
    ) -> None:
        try:
            with get_auth_session() as s:
                s.add(AuditLog(
                    user_id    = user_id,
                    username   = username,
                    email      = email,
                    action     = action,
                    ip_address = ip[:64] if ip else "",
                    detail     = (detail or "")[:999],
                    success    = success,
                ))
                s.commit()
        except Exception as e:
            logger.warning(f"Audit write failed: {e}")


# ── Module-level singleton ────────────────────────────────────────────────────
user_manager = _UserManager()


# ── Seed accounts ─────────────────────────────────────────────────────────────

def seed_default_accounts() -> None:
    """
    Create default accounts on startup if they don't exist.

    Owner account (username login, no email needed on login page):
      username : ALPHAGRID_OWNER_USERNAME  (default: admin)
      password : ALPHAGRID_OWNER_PASSWORD  (default: Admin@Grid1 — change after first login)
      role     : ADMIN
      is_owner : True  ← cannot be deactivated

    Demo accounts (email login):
      ALPHAGRID_BUILDER_PASSWORD  → builder@alphagrid.app  (BUILDER)
      ALPHAGRID_TRADER_PASSWORD   → trader@alphagrid.app   (TRADER)
    """
    created = 0

    # ── Owner ─────────────────────────────────────────────────────────────
    owner = user_manager.get_by_username(OWNER_USERNAME)
    if not owner:
        u, err = user_manager.create_user(
            email        = OWNER_EMAIL,
            password     = OWNER_PASSWORD,
            display_name = OWNER_NAME,
            username     = OWNER_USERNAME,
            role         = UserRole.ADMIN,
            is_owner     = True,
        )
        if u:
            created += 1
            logger.info(
                f"Owner account seeded | username: {OWNER_USERNAME} "
                f"| CHANGE THIS PASSWORD AFTER FIRST LOGIN"
            )
        else:
            logger.error(f"Failed to seed owner: {err}")
    else:
        # Ensure is_owner flag is set (migration safety)
        if not owner.is_owner:
            with get_auth_session() as s:
                u = s.query(User).filter_by(id=owner.id).first()
                if u:
                    u.is_owner = True
                    s.commit()

    # ── Demo accounts ──────────────────────────────────────────────────────
    demos = [
        ("builder@alphagrid.app", os.getenv("ALPHAGRID_BUILDER_PASSWORD", "Builder1!"), "Builder", UserRole.BUILDER, "builder"),
        ("trader@alphagrid.app",  os.getenv("ALPHAGRID_TRADER_PASSWORD",  "Trader1!"),  "Trader",  UserRole.TRADER,  "trader"),
        ("demo@alphagrid.app",    "demo1234", "Demo User", UserRole.TRADER, "demo"),
    ]
    for em, pw, name, role, uname in demos:
        if not user_manager.get_by_email(em):
            u, err = user_manager.create_user(
                email=em, password=pw, display_name=name,
                role=role, username=uname,
            )
            if u:
                created += 1
                logger.info(f"Demo account seeded: @{uname} <{em}>")
            else:
                # Username conflict — try with suffix
                u, err = user_manager.create_user(
                    email=em, password=pw, display_name=name,
                    role=role, username=uname + "_demo",
                )
                if u: created += 1

    if created:
        logger.info(f"Auth DB: {created} account(s) seeded | DB: {DB_PATH}")
    else:
        logger.info(f"Auth DB: all accounts present | DB: {DB_PATH}")


# ── Backward-compat for older import patterns ─────────────────────────────────
def hash_password(plain: str) -> str:   return _hash_password(plain)
def verify_password(p: str, h: str):    return _verify_password(p, h)
def create_access_token(user: User):    return _create_access_token(user)
def create_refresh_token():             return _create_refresh_token()
decode_token = decode_access_token      # backward-compat alias

def audit(action: str, success: bool = True, **kwargs) -> None:
    user_manager._write_audit(action, success, **kwargs)

def get_audit_log(limit: int = 100) -> list:
    return user_manager.get_audit_log(limit)

class UserManager:
    """Backward-compat class facade over user_manager singleton."""
    @staticmethod
    def get_by_email(e):      return user_manager.get_by_email(e)
    @staticmethod
    def get_by_id(i):         return user_manager.get_by_id(i)
    @staticmethod
    def authenticate(e, p):   return user_manager.authenticate(e, p)
    @staticmethod
    def create(*a, **kw):     return user_manager.create_user(*a, **kw)
    @staticmethod
    def revoke_session(t):    return user_manager.revoke_session(t)
    @staticmethod
    def revoke_all(uid):      return user_manager.revoke_all_sessions(uid)
    @staticmethod
    def change_password(uid, pw):
        import warnings; warnings.warn("Use user_manager.change_password()")
        with get_auth_session() as s:
            u = s.query(User).filter_by(id=uid).first()
            if u: u.hashed_password = _hash_password(pw); s.commit(); return True,""
        return False,"User not found"
    @staticmethod
    def all_users():          return user_manager.list_users()
    @staticmethod
    def deactivate(uid):      ok,_ = user_manager.deactivate(uid); return ok
