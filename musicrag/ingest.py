"""Unified ingestion pipeline for every MusicRAG source.

All inputs share the same chunking, embeddings and SQLite storage, and each
document is idempotent per its `source` (re-ingesting only replaces the
documents of that source):

- `knowledge/`: Markdown/plain text with YAML frontmatter (`parse_frontmatter` +
  `chunk_text`) -> any doc_type (artist, album, song, genre, general).
- `new_sqlite/`: text discography, one `[ARTIST].txt` per artist
  (album lines `Title (YY)` + track lines `NN. Track`).
- `ingest/`: drop folder. Put new `[ARTIST].txt` (grupos) files here to update
  the catalog without touching `new_sqlite/`. Same format, same parser.
- `sqlite/music.db`: external discography database (artists/albums/tracks
  tables), ingested into `artist` + `album` documents.
"""
import json
import re
import sqlite3
from pathlib import Path

from . import embeddings
from .config import (
    CHUNK_OVERLAP_TOKENS,
    CHUNK_TARGET_TOKENS,
    CHUNK_TYPES,
    EMBED_MODEL,
    INGEST_DIR,
    KNOWLEDGE_DIR,
    MUSIC_DB_PATH,
    NEW_SQLITE_DIR,
)
from .db import get_conn, init_db
from .embed_backend import doc_prefix, embed_texts, recompute_idf_from_chunks

FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)

DEFAULT_SOURCE_MUSICDB = "sqlite://music.db"
DEFAULT_SOURCE_NEW = "sqlite://new_sqlite"
DEFAULT_SOURCE_DROP = "sqlite://ingest"

TRACK_RE = re.compile(r"^\s*(\d{1,2})\.\s+(.*?)\s*$")
ALBUM_RE = re.compile(r"^(.*)\((\d{1,2})\)\s*$")


# ------------------------------------------------------------------ knowledge/
def parse_frontmatter(text: str) -> tuple[str, str, dict, str]:
    """Extract simplified YAML metadata from the document header."""
    meta: dict = {}
    body = text
    m = FRONTMATTER_RE.match(text)
    if m:
        body = text[m.end():]
        for line in m.group(1).splitlines():
            if ":" in line:
                key, _, value = line.partition(":")
                meta[key.strip().lower()] = value.strip()
    title = str(meta.get("title", "")).strip()
    doc_type = str(meta.get("type", "general")).strip().lower()
    if doc_type not in CHUNK_TYPES:
        doc_type = "general"
    return title, doc_type, meta, body


def chunk_text(body: str, target: int = CHUNK_TARGET_TOKENS,
               overlap: int = CHUNK_OVERLAP_TOKENS) -> list[str]:
    """Split the body into chunks (~words) with overlap, cutting at sentences when possible."""
    words = re.findall(r"\S+", re.sub(r"[ \t]+", " ", body))
    if not words:
        return []
    chunks: list[str] = []
    start, n = 0, len(words)
    while start < n:
        end = min(start + target, n)
        chunk = " ".join(words[start:end])
        if end < n:
            lo = max(start, end - max(target // 5, 1))
            seg = " ".join(words[lo:end])
            matches = list(re.finditer(r"[.!?](?=\s|$)", seg))
            if matches:
                end = lo + matches[-1].start() + 1
                chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= n:
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c.strip()]


def knowledge_files(directory=None) -> list[Path]:
    directory = Path(directory) if directory else KNOWLEDGE_DIR
    directory.mkdir(parents=True, exist_ok=True)
    return sorted(p for ext in ("*.txt", "*.md") for p in directory.rglob(ext))


def ingest_file(path, model: str = EMBED_MODEL, progress=None) -> tuple[int, int | None]:
    init_db()
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    title, doc_type, meta, body = parse_frontmatter(text)
    if not title:
        title = path.stem
    meta.setdefault("filename", path.name)
    meta["path"] = str(path)

    chunks = chunk_text(body)
    if not chunks:
        return 0, None

    prefixed = [f"{doc_prefix()}{c}" for c in chunks]
    vecs = embed_texts(model, prefixed, progress=progress)

    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM documents WHERE source = ?", (str(path),)
        ).fetchone()
        if existing:
            doc_id = existing["id"]
            conn.execute(
                "DELETE FROM chunks_fts WHERE rowid IN "
                "(SELECT id FROM chunks WHERE doc_id = ?)",
                (doc_id,),
            )
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute(
                "UPDATE documents SET title=?, doc_type=?, meta=? WHERE id=?",
                (title, doc_type, json.dumps(meta, ensure_ascii=False), doc_id),
            )
        else:
            cur = conn.execute(
                "INSERT INTO documents (title, source, doc_type, meta) VALUES (?,?,?,?)",
                (title, str(path), doc_type, json.dumps(meta, ensure_ascii=False)),
            )
            doc_id = cur.lastrowid
        for i, (chunk, vec) in enumerate(zip(chunks, vecs)):
            conn.execute(
                "INSERT INTO chunks (doc_id, content, chunk_index, embedding) "
                "VALUES (?,?,?,?)",
                (doc_id, chunk, i, embeddings.to_blob(vec)),
            )
    return len(chunks), doc_id


def ingest_dir(directory=None, model: str = EMBED_MODEL,
               files=None, progress=None) -> dict:
    target = [Path(f) for f in files] if files else knowledge_files(directory)
    results: dict = {"ok": [], "errors": []}
    n = len(target)

    def sub_progress(p: float, i: int) -> None:
        if progress and n:
            progress((i + p) / n)

    for i, path in enumerate(target):
        try:
            n_chunks, doc_id = ingest_file(
                path, model=model,
                progress=(lambda p, i=i: sub_progress(p, i)),
            )
            results["ok"].append({"path": str(path), "chunks": n_chunks, "doc_id": doc_id})
        except Exception as exc:  # noqa: BLE001
            results["errors"].append({"path": str(path), "error": str(exc)})
        if progress:
            progress((i + 1) / n)
    return results


# -------------------------------------------------------- music.db catalog (.db)
def load_catalog(musicdb: Path | None = None):
    """Read artists/albums/tracks from the external sqlite/music.db."""
    musicdb = Path(musicdb or MUSIC_DB_PATH)
    if not musicdb.exists():
        raise FileNotFoundError(f"The catalog does not exist: {musicdb}")
    conn = sqlite3.connect(musicdb)
    conn.row_factory = sqlite3.Row
    artists = conn.execute(
        "SELECT id, name FROM artists ORDER BY name"
    ).fetchall()
    albums = conn.execute(
        """
        SELECT a.id, a.title, a.year, a.artist_id, x.name AS artist
        FROM albums a JOIN artists x ON x.id = a.artist_id
        ORDER BY a.artist_id, a.year, a.title
        """
    ).fetchall()
    tracks = conn.execute(
        """
        SELECT t.title, t.track_pos, t.cover_artist, t.album_id
        FROM tracks t ORDER BY t.album_id, t.track_pos
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in artists], [dict(r) for r in albums], [dict(r) for r in tracks]


def build_docs_musicdb(musicdb: Path | None = None) -> list[dict]:
    """Build artist + album documents from music.db."""
    artists, albums, tracks = load_catalog(musicdb)

    tracks_by_album: dict[int, list[dict]] = {}
    for t in tracks:
        tracks_by_album.setdefault(t["album_id"], []).append(t)

    albums_by_artist: dict[int, list[dict]] = {}
    for a in albums:
        albums_by_artist.setdefault(a["artist_id"], []).append(a)

    docs: list[dict] = []
    for art in artists:
        albs = albums_by_artist.get(art["id"], [])
        n_tracks = sum(len(tracks_by_album.get(a["id"], [])) for a in albs)
        years = [a["year"] for a in albs if a["year"]]
        lines = [
            f"# {art['name']}",
            "",
            f"Discography: {len(albs)} albums, {n_tracks} songs"
            + (f", {min(years)}-{max(years)}." if years else "."),
            "",
            "Albums:",
        ]
        for a in albs:
            tag = f" ({a['year']})" if a["year"] else ""
            lines.append(f"- {a['title']}{tag}")
        docs.append({
            "title": art["name"],
            "doc_type": "artist",
            "meta": {
                "source": f"{DEFAULT_SOURCE_MUSICDB}#artist:{art['id']}",
                "musicdb": str(musicdb or MUSIC_DB_PATH),
                "albums": len(albs),
                "tracks": n_tracks,
                "year_min": min(years) if years else None,
                "year_max": max(years) if years else None,
            },
            "content": "\n".join(lines),
        })

        for a in albs:
            tl = tracks_by_album.get(a["id"], [])
            title = f"{art['name']} - {a['title']}"
            lines = [f"# {title}", ""]
            header = art["name"]
            if a["year"]:
                header += f" · {a['year']}"
            lines.append(header)
            lines += ["", "Tracklist:"]
            for t in tl:
                track_name = t["title"]
                if t.get("cover_artist"):
                    track_name = f"{track_name} (cover de {t['cover_artist']})"
                lines.append(f"{t['track_pos']:02d}. {track_name}")
            docs.append({
                "title": title,
                "doc_type": "album",
                "meta": {
                    "source": f"{DEFAULT_SOURCE_MUSICDB}#album:{a['id']}",
                    "musicdb": str(musicdb or MUSIC_DB_PATH),
                    "artist": art["name"],
                    "year": a["year"],
                    "tracks": len(tl),
                },
                "content": "\n".join(lines),
            })
    return docs


def ingest_musicdb(musicdb: Path | None = None, model: str = EMBED_MODEL,
                   progress=None) -> dict:
    """Ingest music.db into MusicRAG. Returns {(artists, albums, chunks)}."""
    init_db()
    docs = build_docs_musicdb(musicdb)

    flat: list[tuple[dict, str]] = []
    for doc in docs:
        for chunk in chunk_text(doc["content"]):
            flat.append((doc, chunk))

    vecs = embed_texts(
        model,
        [f"{doc_prefix()}{chunk}" for _doc, chunk in flat],
        progress=progress,
    )

    artist_count = sum(1 for d in docs if d["doc_type"] == "artist")
    album_count = sum(1 for d in docs if d["doc_type"] == "album")

    with get_conn() as conn:
        idx = 0
        for doc in docs:
            doc_chunks = []
            while idx < len(flat) and flat[idx][0] is doc:
                doc_chunks.append((flat[idx][1], vecs[idx]))
                idx += 1
            _store_doc(conn, doc, doc_chunks)

    return {"artists": artist_count, "albums": album_count,
            "chunks": len(flat), "documents": len(docs)}


# ----------------------------------------------------- text catalog (new_sqlite/)
def _full_year(yy: int) -> int:
    """2-digit year: YY<=30 -> 20YY, else 19YY."""
    return 2000 + yy if yy <= 30 else 1900 + yy


def parse_artist_file(path: Path) -> list[dict]:
    """Return [{title, year, tracks:[{pos, name}]}] for an artist file."""
    name = path.stem.strip("[]")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    albums: list[dict] = []
    current: dict | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith(name + " - "):
            continue  # trailer: self-references to its own albums
        m = TRACK_RE.match(line)
        if m:
            if current is None:
                current = None  # orphan tracks before any album
                continue
            current["tracks"].append({"pos": m.group(1), "name": m.group(2)})
            continue
        m = ALBUM_RE.match(line)
        if m:
            if current is not None:
                albums.append(current)
            current = {
                "title": m.group(1).strip(),
                "year": _full_year(int(m.group(2))),
                "tracks": [],
            }
    if current is not None:
        albums.append(current)
    return albums


def _build_docs_text(directory: Path, source_prefix: str, pattern: str) -> list[dict]:
    """artist + album documents for every `[ARTIST].txt` in the directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"The catalog does not exist: {directory}")
    docs: list[dict] = []
    for path in sorted(directory.glob(pattern)):
        artist = path.stem.strip("[]")
        albums = parse_artist_file(path)
        # Skip files that are not real catalog entries (e.g. extra notes/readme
        # left in the drop folder): no album lines -> nothing to ingest.
        if not albums:
            continue
        n_tracks = sum(len(a["tracks"]) for a in albums)
        years = [a["year"] for a in albums if a["year"]]
        lines = [
            f"# {artist}",
            "",
            f"Discography: {len(albums)} albums, {n_tracks} songs"
            + (f", {min(years)}-{max(years)}." if years else "."),
            "",
            "Albums:",
        ]
        for a in albums:
            tag = f" ({a['year']})" if a["year"] else ""
            lines.append(f"- {a['title']}{tag}")
        docs.append({
            "title": artist,
            "doc_type": "artist",
            "meta": {
                "source": f"{source_prefix}#artist:{artist}",
                "new_sqlite": str(path),
                "albums": len(albums),
                "tracks": n_tracks,
                "year_min": min(years) if years else None,
                "year_max": max(years) if years else None,
            },
            "content": "\n".join(lines),
        })
        for a in albums:
            title = f"{artist} - {a['title']}"
            content = [f"# {title}", ""]
            header = artist
            if a["year"]:
                header += f" · {a['year']}"
            content.append(header)
            content += ["", "Tracklist:"]
            for t in a["tracks"]:
                content.append(f"{t['pos']}. {t['name']}")
            docs.append({
                "title": title,
                "doc_type": "album",
                "meta": {
                    "source": f"{source_prefix}#album:{artist}|{a['year']}|{a['title']}",
                    "new_sqlite": str(path),
                    "artist": artist,
                    "year": a["year"],
                    "tracks": len(a["tracks"]),
                },
                "content": "\n".join(content),
            })
    return docs


def _ingest_docs(docs: list[dict], model: str = EMBED_MODEL,
                 progress=None, train: bool = True) -> dict:
    """Embed and store the built documents. Returns {(artists, albums, chunks)}."""
    init_db()
    flat: list[tuple[dict, str]] = []
    for doc in docs:
        for chunk in chunk_text(doc["content"]):
            flat.append((doc, chunk))

    vecs = embed_texts(
        model,
        [f"{doc_prefix()}{chunk}" for _doc, chunk in flat],
        progress=progress, train=train,
    )

    with get_conn() as conn:
        idx = 0
        for doc in docs:
            doc_chunks = []
            while idx < len(flat) and flat[idx][0] is doc:
                doc_chunks.append((flat[idx][1], vecs[idx]))
                idx += 1
            _store_doc(conn, doc, doc_chunks)

    return {"artists": sum(1 for d in docs if d["doc_type"] == "artist"),
            "albums": sum(1 for d in docs if d["doc_type"] == "album"),
            "chunks": len(flat), "documents": len(docs)}


def build_docs_new(directory: Path | None = None) -> list[dict]:
    """Build artist + album documents from new_sqlite/."""
    return _build_docs_text(Path(directory) if directory else NEW_SQLITE_DIR,
                            DEFAULT_SOURCE_NEW, "[[]*].txt")


def build_docs_drop(directory: Path | None = None) -> list[dict]:
    """Build artist + album documents from the ingest/ drop folder (any .txt)."""
    return _build_docs_text(Path(directory) if directory else INGEST_DIR,
                            DEFAULT_SOURCE_DROP, "*.txt")


def ingest_new(directory: Path | None = None, model: str = EMBED_MODEL,
               progress=None) -> dict:
    """Ingest new_sqlite/ (or `directory`) into MusicRAG. Returns a summary."""
    return _ingest_docs(build_docs_new(directory), model=model, progress=progress,
                        train=False)


def ingest_drop(directory: Path | None = None, model: str = EMBED_MODEL,
                progress=None) -> dict:
    """Ingest the ingest/ drop folder (new grupos .txt). Returns a summary."""
    return _ingest_docs(build_docs_drop(directory), model=model, progress=progress,
                        train=False)


# ------------------------------------------------------------------- storage
def _store_doc(conn, doc: dict, chunk_vecs: list[tuple[str, list]]) -> int:
    """Insert-or-replace one document and its chunks (idempotent by source)."""
    meta = doc["meta"]
    source = meta["source"]
    existing = conn.execute(
        "SELECT id FROM documents WHERE source = ?", (source,)
    ).fetchone()
    if existing:
        doc_id = existing["id"]
        conn.execute(
            "DELETE FROM chunks_fts WHERE rowid IN "
            "(SELECT id FROM chunks WHERE doc_id = ?)",
            (doc_id,),
        )
        conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        conn.execute(
            "UPDATE documents SET title=?, doc_type=?, meta=? WHERE id=?",
            (doc["title"], doc["doc_type"], json.dumps(meta, ensure_ascii=False), doc_id),
        )
    else:
        cur = conn.execute(
            "INSERT INTO documents (title, source, doc_type, meta) VALUES (?,?,?,?)",
            (doc["title"], source, doc["doc_type"], json.dumps(meta, ensure_ascii=False)),
        )
        doc_id = cur.lastrowid
    for i, (chunk, vec) in enumerate(chunk_vecs):
        conn.execute(
            "INSERT INTO chunks (doc_id, content, chunk_index, embedding) "
            "VALUES (?,?,?,?)",
            (doc_id, chunk, i, embeddings.to_blob(vec)),
        )
    return doc_id


def reindex(progress=None) -> None:
    """Re-trains the IDF and re-embeds all chunks (local backends)."""
    from .embed_backend import embed_local

    with get_conn() as conn:
        recompute_idf_from_chunks(conn)
        rows = conn.execute("SELECT id, content FROM chunks").fetchall()
        n = len(rows)
        if not n:
            return
        texts = [f"{doc_prefix()}{r['content']}" for r in rows]
        vecs = embed_local(texts, progress=progress, train=True)
        for r, vec in zip(rows, vecs):
            conn.execute(
                "UPDATE chunks SET embedding=? WHERE id=?",
                (embeddings.to_blob(vec), r["id"]),
            )