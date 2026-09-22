import json
import re

from .config import LLM_MODEL, TOP_K
from .ollama_client import chat_stream
from .search import _query_years, lexical_search, retrieve

SYSTEM_PROMPT = (
    "You are MusicRAG, a music expert assistant that operates 100% locally. "
    "Respond in the language of the question and use EXCLUSIVELY the provided <context>. "
    "If the context does not contain the answer, honestly state that it does not "
    "appear in the knowledge base, without fabricating data. "
    "Be concise yet helpful. End with a 'Sources:' section listing the "
    "titles of the documents you used. "
    "When listing an artist's albums, present each one as "
    "'ARTIST AlbumTitle (year)', with the artist's name in uppercase and without hyphens."
)


def build_context(results: list[tuple[float, dict]]) -> str:
    parts = []
    for i, (_score, row) in enumerate(results, 1):
        label = row["title"]
        parts.append(f"[{i}] ({row['doc_type']}) {label}\n{row['content']}")
    return "\n\n---\n\n".join(parts)


def build_messages(question: str, context: str) -> list[dict]:
    user = (
        f"<context>\n{context}\n</context>\n\n" 
        f"Question: {question}\n\n" 
        "Answer based solely on context."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def retrieve_and_build(question: str, top_k: int = TOP_K, mode: str = "hybrid",
                       embed_model: str | None = None):
    _qv, results = retrieve(question, top_k=top_k, mode=mode, embed_model=embed_model)
    messages = build_messages(question, build_context(results))
    return results, messages


def _artist_display(name: str) -> str:
    """Normalize the artist: 'ALAN PARSONS PROJECT, The' -> 'THE ALAN PARSONS PROJECT'."""
    n = (name or "").strip()
    if n.lower().endswith(", the"):
        n = ("The " + n[:-5].strip())
    return n.upper()


def catalog_year_listing(question: str) -> str | None:
    """Deterministic 'ARTIST Album (year)' listing for year questions.

    Returns None if it does not apply (no year or no albums of that year), so
    the chat then generates the answer.
    """
    years = _query_years(question)
    if not years:
        return None
    year = int(years[0])
    hits = lexical_search(str(year), top_k=60)
    lines: list[str] = []
    seen: set[tuple] = set()
    for _score, row in hits:
        if row["doc_type"] != "album":
            continue
        meta = json.loads(row.get("meta") or "{}")
        if int(meta.get("year") or 0) != year:
            continue
        artist, album = "", row["title"]
        if " - " in row["title"]:
            artist, album = row["title"].split(" - ", 1)
        key = (artist, album)
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"{_artist_display(artist)} {album} ({year})")
    if not lines:
        return None
    # Hard Markdown line break ("  \\n") so the Streamlit UI (which renders
    # Markdown) does not collapse one album per line into spaces.
    return "\n  \n".join(lines)


def stream_answer(question: str, top_k: int = TOP_K, mode: str = "hybrid",
                  model: str = LLM_MODEL, embed_model: str | None = None):
    """Retrieves context and returns (results, LLM token generator)."""
    results, messages = retrieve_and_build(question, top_k, mode, embed_model)
    listing = catalog_year_listing(question)
    if listing is not None:
        return results, iter([listing])

    def gen_tokens():
        for chunk in chat_stream(model, messages):
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content

    return results, gen_tokens()