from __future__ import annotations

import psycopg2
from pgvector.psycopg2 import register_vector
from typing import TYPE_CHECKING
import numpy as np
import hashlib
import os

if TYPE_CHECKING:
    from ingest import Chunk
    

def connect_db() -> psycopg2.extensions.connection:
    connection = psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", "5432")),
        dbname=os.environ.get("PGDATABASE", "postgres"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", "postgres"),
    )
    try:
        register_vector(connection)
    except Exception:
        connection.close()
        raise
    
    return connection

def create_table() -> None:
    conn = None
    try:
        conn = connect_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS kb_chunks (
                        id            SERIAL PRIMARY KEY,
                        source        TEXT NOT NULL,          
                        heading_path  TEXT NOT NULL,         
                        content       TEXT NOT NULL,          
                        chunk_hash    TEXT NOT NULL UNIQUE, 
                        embedding     VECTOR(512)
                    );
                    """
                )
    except psycopg2.Error as e:
        print(f"报错： psycopg2.Error {e}")
        raise
    finally:
        if conn is not None:
            conn.close()


def chunk_hash(chunk: Chunk) -> str:
    combined = f"{chunk.source}\n{chunk.heading_path}\n{chunk.text}"
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()

def insert_chunks(chunks_with_vec: list[tuple[Chunk, np.ndarray]]) -> None:
    conn = None
    try:
        conn = connect_db()
        with conn:
            with conn.cursor() as cur:
                for chunk, vector in chunks_with_vec:
                    cur.execute(
                        """
                        INSERT INTO kb_chunks (source, heading_path, content, chunk_hash, embedding)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (chunk_hash) DO NOTHING
                        """,
                        (chunk.source, chunk.heading_path, chunk.text, chunk_hash(chunk), vector)
                    )
    except psycopg2.Error as e:
        print(f"插入失败：{e}")
        raise
    finally:
        if conn is not None:
            conn.close()

def count_chunk() -> int:
    conn = None
    try:
        conn = connect_db()
        with conn:
            with conn.cursor() as cur:
                cur.execute("""SELECT COUNT(*) FROM kb_chunks;""")
                result = cur.fetchone()
                return result[0] if result else 0
    except psycopg2.Error as e:
        print(f"查询失败 :{e}")
        raise
    finally:
        if conn is not None:
            conn.close()
