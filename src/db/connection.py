"""
Direct Postgres connection to Supabase, bypassing supabase-py's REST client —
the REST client doesn't support `ORDER BY embedding <=> %s` vector queries
without writing a Postgres RPC function first. Going direct is simpler and
keeps the SQL visible/debuggable in this file instead of hidden in an RPC.

IMPORTANT: use Supabase's connection *pooler* URI (port 6543, "Transaction"
mode), not the direct connection (port 5432), for SUPABASE_DB_URL. Render's
free tier can spin up multiple instances / reconnect frequently, and
Supabase's free tier direct-connection limit is small — the pooler avoids
exhausting it. Find this under Project Settings → Database → Connection
pooling in the Supabase dashboard.
"""
from contextlib import contextmanager
import psycopg2
from pgvector.psycopg2 import register_vector

from src.config import settings


@contextmanager
def get_connection():
    if not settings.supabase_db_url:
        raise RuntimeError("SUPABASE_DB_URL not set — see .env.example")
    conn = psycopg2.connect(settings.supabase_db_url)
    register_vector(conn)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
