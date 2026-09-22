import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .config import DB_PATH, ensure_dirs

# SQLite + FTS5: chunks are indexed with full-text search (BM25) and their
# embeddings are stored as float32 BLOBs for semantic search.
# Note: in this SQLite build (3.50.x) the FTS5 'exact-content' delete command
# returns 'SQL logic error', so deletions are done by explicit rowid.
SCHEMA = """
PRAGMA foreign_keys = ON;

DROP TRIGGER IF EXISTS chunks_fts_ad;
DROP TRIGGER IF EXISTS chunks_fts_au;
DROP TRIGGER IF EXISTS documents_chunks_fts_ad;

CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    source TEXT NOT NULL UNIQUE,
    doc_type TEXT NOT NULL DEFAULT 'general',
    meta TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding BLOB
);

CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);

-- Audio library: index of songs with their exact location.
CREATE TABLE IF NOT EXISTS music_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL UNIQUE,
    artist TEXT NOT NULL,
    album TEXT,
    track TEXT NOT NULL,
    ext TEXT NOT NULL DEFAULT 'mp3',
    artist_norm TEXT NOT NULL,
    track_norm TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_music_files_track ON music_files(track_norm);
CREATE INDEX IF NOT EXISTS idx_music_files_artist ON music_files(artist_norm);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    tokenize = 'unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_ai AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, content) VALUES (new.id, new.content);
END;

-- Clean the FTS index before the cascade deletes the document's chunks.
CREATE TRIGGER IF NOT EXISTS documents_chunks_fts_ad BEFORE DELETE ON documents BEGIN
    DELETE FROM chunks_fts WHERE rowid IN (SELECT id FROM chunks WHERE doc_id = old.id);
END;
"""


@contextmanager
def get_conn(db_path: str | Path | None = None):
    ensure_dirs()
    path = Path(db_path) if db_path else DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def reset_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            "DROP TABLE IF EXISTS chunks_fts_au;"
            "DROP TABLE IF EXISTS chunks_fts_ad;"
            "DROP TABLE IF EXISTS chunks_fts_ai;"
            "DROP TABLE IF EXISTS chunks_fts;"
            "DROP TABLE IF EXISTS chunks;"
            "DROP TABLE IF EXISTS documents;"
            "DROP TABLE IF EXISTS music_files;"
        )
        conn.executescript(SCHEMA)


def purge_sources(sources: list[str]) -> int:
    """Delete the documents whose `source` is in the list (FTS cleaned via trigger)."""
    if not sources:
        return 0
    with get_conn() as conn:
        for source in sources:
            conn.execute("DELETE FROM documents WHERE source = ?", (source,))
    return len(sources)


def db_stats() -> dict:
    init_db()
    with get_conn() as conn:
        docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        chunks = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
        audio = conn.execute("SELECT COUNT(*) FROM music_files").fetchone()[0]
        by_type = {
            r["doc_type"]: r["n"]
            for r in conn.execute(
                "SELECT doc_type, COUNT(*) AS n FROM documents GROUP BY doc_type ORDER BY n DESC"
            )
        }
    return {"documents": docs, "chunks": chunks, "audio_files": audio, "by_type": by_type}


def list_documents() -> list[dict]:
    init_db()
    with get_conn() as conn:
        rows = conn.execute(
            """
            SELECT d.id, d.title, d.source, d.doc_type, d.meta, d.created_at,
                   COUNT(c.id) AS n_chunks
            FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id
            GROUP BY d.id ORDER BY d.created_at DESC
            """
        ).fetchall()
    return [dict(r) | {"meta": json.loads(r["meta"] or "{}")} for r in rows]