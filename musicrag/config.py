import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = Path(os.getenv("MUSICRAG_KNOWLEDGE_DIR", BASE_DIR / "knowledge"))
DATA_DIR = Path(os.getenv("MUSICRAG_DATA_DIR", BASE_DIR / "data"))
DB_PATH = Path(os.getenv("MUSICRAG_DB", DATA_DIR / "musicrag.db"))

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# External discography catalog (sqlite/music.db) for optional ingestion.
MUSIC_DB_PATH = Path(
    os.getenv("MUSICRAG_MUSICDB", Path(__file__).resolve().parents[2] / "sqlite" / "music.db")
)

# Text-based catalog: one `[ARTIST].txt` per artist.
NEW_SQLITE_DIR = Path(
    os.getenv("MUSICRAG_NEW_SQLITE", Path(__file__).resolve().parents[2] / "new_sqlite")
)

# Drop folder: snippets/full `[ARTIST].txt` files dropped here are picked up by
# `ingest-new` to update the catalog without touching new_sqlite/.
INGEST_DIR = Path(
    os.getenv("MUSICRAG_INGEST_DIR", Path(__file__).resolve().parents[1] / "ingest")
)

# Audio library: indexed into music_files to locate and play songs.
AUDIO_DIR = Path(os.getenv("MUSICRAG_AUDIO_DIR", "G:/[AUDIO]"))
AIMP_EXE = os.getenv("MUSICRAG_AIMP", r"C:\Program Files\AIMP\AIMP.exe")

EMBED_MODEL = os.getenv("MUSICRAG_EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("MUSICRAG_LLM_MODEL", "qwen2.5-coder:1.5b")

# auto (Ollama -> local), ollama (strict) or local (no Ollama).
EMBED_BACKEND = os.getenv("MUSICRAG_EMBED_BACKEND", "auto").lower()

# Prefixes recommended by nomic-embed-text to improve retrieval.
EMBED_DOC_PREFIX = os.getenv("MUSICRAG_EMBED_DOC_PREFIX", "search_document: ")
EMBED_QUERY_PREFIX = os.getenv("MUSICRAG_EMBED_QUERY_PREFIX", "search_query: ")

TOP_K = int(os.getenv("MUSICRAG_TOP_K", "5"))
CHUNK_TARGET_TOKENS = int(os.getenv("MUSICRAG_CHUNK_TOKENS", "300"))
CHUNK_OVERLAP_TOKENS = int(os.getenv("MUSICRAG_CHUNK_OVERLAP", "40"))

EMBED_BATCH_SIZE = int(os.getenv("MUSICRAG_EMBED_BATCH", "16"))

CHUNK_TYPES = ["artist", "album", "song", "genre", "general"]

VALID_MODES = ("hybrid", "semantic", "lexical")


def ensure_dirs() -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)