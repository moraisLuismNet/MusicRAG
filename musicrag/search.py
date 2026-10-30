import re
import warnings

import numpy as np

from .config import EMBED_MODEL, TOP_K, VALID_MODES
from .db import get_conn
from .embeddings import cosine_similarity, from_blob
from .embed_backend import corpus_term_df, effective_backend, embed_one, query_prefix

ROW_SQL = """
    SELECT c.id, c.content, c.chunk_index, c.embedding,
           d.title, d.doc_type, d.meta, d.source
    FROM chunks c JOIN documents d ON d.id = c.doc_id
"""


def semantic_search(query_embedding, top_k: int = TOP_K) -> list[tuple[float, dict]]:
    """Semantic search: cosine similarity over all embeddings (float32 BLOBs)."""
    q = np.asarray(query_embedding, dtype="float32")
    with get_conn() as conn:
        rows = conn.execute(ROW_SQL).fetchall()
    scored = []
    mismatch_warned = False
    for r in rows:
        vec = from_blob(r["embedding"])
        if len(vec) != len(q):
            if not mismatch_warned:
                mismatch_warned = True
                warnings.warn(
                    f"Stored embeddings have {len(vec)} dims but the query has "
                    f"{len(q)}; re-ingest to fix. Skipping semantic ranking.",
                    stacklevel=2,
                )
            continue
        s = cosine_similarity(query_embedding, vec)
        scored.append((s, dict(r)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# Stop words: they barely help BM25 and only pollute the FTS query.
_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del", "que", "qué",
    "como", "para", "por", "con", "sin", "entre", "sobre", "hay", "es", "son", "se",
    "su", "sus", "al", "y", "o", "u", "a", "me", "mi", "tu", "the", "and", "for",
    "this", "that", "with", "from", "have", "has", "no", "si", "cual", "cuál",
    "cuáles", "cuantas", "cuántas", "album", "álbum", "albumes", "álbumes",
    "canciones", "cancion", "canción", "incluye", "contiene",
    "in", "of", "to", "at", "on", "me", "it", "its", "do", "does", "did",
    "what", "when", "where", "how", "why", "who", "whom", "is", "are",
    "was", "were", "don", "doesn", "en", "no", "ni", "mas", "más",
}


def _fts_terms(query: str) -> tuple[list[str], list[str]]:
    """Return (terms) and (significant terms) for FTS5."""
    words = re.findall(r"[^\W_]+", re.sub(r"[^0-9A-Za-zÁÉÍÓÚÜÑáéíóúüñ\s]", " ", query).lower())
    terms = []
    for w in words:
        if w in _STOPWORDS:
            continue
        if w.isdigit():
            # Years (3-4 digits) do get indexed in FTS5: they allow "1971 albums".
            if len(w) in (3, 4):
                terms.append(w)
        elif len(w) > 1:
            terms.append(w)
    terms = terms[:12]
    # Sort from least to most generic and drop corpus-frequent terms,
    # which flood the BM25 pool with loose matches.
    with_ratio = sorted(
        terms,
        key=lambda t: (corpus_term_df(t) is not None, corpus_term_df(t) or 0.0),
    )
    filtered = [t for t in with_ratio
                if (rel_df := corpus_term_df(t)) is None or rel_df <= 0.15]
    if not filtered:
        filtered = with_ratio[:2]
    return with_ratio, filtered


def _fts_query(query: str) -> str:
    """FTS5 query with AND (precision). Returns "" when there are no terms."""
    _, terms = _fts_terms(query)
    if not terms:
        return ""
    if len(terms) == 1:
        return f'"{terms[0]}"'
    return " AND ".join(f'"{t}"' for t in terms)


def _fts_query_or(query: str) -> str:
    """FTS5 query with OR (recall). Returns "" when there are no terms."""
    _, terms = _fts_terms(query)
    if not terms:
        return ""
    return " OR ".join(f'"{t}"' for t in terms)


def lexical_search(query: str, top_k: int = TOP_K) -> list[tuple[float, dict]]:
    """Lexical search with FTS5/BM25 (accent-insensitive), AND with OR fallback."""
    fts = _fts_query(query)
    if not fts:
        return []
    fts_or = (_fts_query_or(query) if " AND " in fts else None)
    rows = []
    for attempt in (fts, fts_or):
        if not attempt:
            continue
        with get_conn() as conn:
            rows = conn.execute(
                f"""
                SELECT c.id, c.content, c.chunk_index,
                       d.title, d.doc_type, d.meta, d.source,
                       bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.rowid
                JOIN documents d ON d.id = c.doc_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (attempt, top_k),
            ).fetchall()
        if rows:
            break
    out = []
    for r in rows:
        rank = float(r["rank"])
        rec = dict(r)
        rec["rank"] = rank
        out.append((1.0 / (1.0 + rank), rec))
    return out


_YEAR_DIGITS_RE = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")

_UNIDADES = {"uno": 1, "un": 1, "una": 1, "dos": 2, "tres": 3, "cuatro": 4,
             "cinco": 5, "seis": 6, "siete": 7, "ocho": 8, "nueve": 9}
_DECENAS = {"diez": 10, "veinte": 20, "treinta": 30, "cuarenta": 40,
            "cincuenta": 50, "sesenta": 60, "setenta": 70, "ochenta": 80,
            "noventa": 90}
_COMPUESTOS = {"once": 11, "doce": 12, "trece": 13, "catorce": 14, "quince": 15,
               "dieciseis": 16, "dieciséis": 16, "diecisiete": 17, "dieciocho": 18,
               "diecinueve": 19, "veintiuno": 21, "veintiún": 21, "veintiuna": 21,
               "veintidos": 22, "veintidós": 22, "veintitres": 23, "veintitrés": 23,
               "veinticuatro": 24, "veinticinco": 25, "veintiseis": 26,
               "veintiséis": 26, "veintisiete": 27, "veintiocho": 28,
               "veintinueve": 29}
_CENTENAS = {"cien": 100, "ciento": 100, "doscientos": 200, "doscientas": 200,
             "trescientos": 300, "trescientas": 300, "cuatrocientos": 400,
             "cuatrocientas": 400, "quinientos": 500, "quinientas": 500,
             "seiscientos": 600, "seiscientas": 600, "setecientos": 700,
             "setecientas": 700, "ochocientos": 800, "ochocientas": 800,
             "novecientos": 900, "novecientas": 900}


def _parse_spanish_year(phrase: str) -> int | None:
    """Convert a written-out year ('mil novecientos ochenta y cinco') to int."""
    tokens = re.sub(r"[^\wáéíóúü]+", " ", phrase.lower()).split()
    value = 0
    i = 0
    if len(tokens) >= 3 and tokens[0] == "dos" and tokens[1] == "mil":
        value += 2000
        i = 2
    elif tokens and tokens[0] == "mil":
        value += 1000
        i = 1
    elif tokens and tokens[0] in _CENTENAS:
        # a leading hundreds word means "... thousand nine hundred...": 1000 + n
        value += 1000 + _CENTENAS[tokens[0]]
        i = 1
    else:
        return None
    if i < len(tokens) and tokens[i] in _CENTENAS:
        value += _CENTENAS[tokens[i]]
        i += 1
    if i < len(tokens):
        t = tokens[i]
        if t in _COMPUESTOS:
            value += _COMPUESTOS[t]
            i += 1
        elif t in _DECENAS:
            value += _DECENAS[t]
            i += 1
            if i < len(tokens) and tokens[i] == "y":
                i += 1
            if i < len(tokens) and tokens[i] in _UNIDADES:
                value += _UNIDADES[tokens[i]]
                i += 1
        elif t in _UNIDADES:
            value += _UNIDADES[t]
            i += 1
    if i != len(tokens):
        return None
    return value if 1900 <= value <= 2099 else None


def _spanish_years(query: str) -> list[int]:
    return [y for y in (_parse_spanish_year(m) for m
                        in re.findall(r"\b(?:dos\s+mil\s+|mil\s+|(?:cien|"
                                      r"doscientos|trescientos|cuatrocientos|"
                                      r"quinientos|seiscientos|setecientos|"
                                      r"ochocientos|novecientos)(?:as)?\s*)"
                                      r"(?:[a-záéíóúü]+\s*)+", query))
            if y is not None]


def _query_years(query: str) -> list[str]:
    years = _YEAR_DIGITS_RE.findall(query)
    years += [str(y) for y in _spanish_years(query)]
    return sorted(set(years), key=int)


def hybrid_search(query_embedding, query: str, top_k: int = TOP_K,
                  k: int = 60, semantic_weight: float = 1.0,
                  lexical_weight: float = 1.0) -> list[tuple[float, dict]]:
    """Reciprocal rank fusion (RRF) of the two engines."""
    sem = semantic_search(query_embedding, top_k=top_k * 3)
    lex = lexical_search(query, top_k=top_k * 3)
    merged: dict[int, dict] = {}
    for rank, (_score, row) in enumerate(sem):
        entry = merged.setdefault(row["id"], {"row": row, "score": 0.0})
        entry["score"] += semantic_weight / (k + rank + 1)
    for rank, (_score, row) in enumerate(lex):
        entry = merged.setdefault(row["id"], {"row": row, "score": 0.0})
        entry["score"] += lexical_weight / (k + rank + 1)
    years = _query_years(query)
    if years:
        # Year questions (e.g. "1971 albums"): prioritize albums that mention
        # that year ("ARTIST · 1971" or " (1971)") over other docs.
        for entry in merged.values():
            content = entry["row"].get("content", "")
            album_years = [y for y in years
                           if re.search(rf"·\s*{y}\b", content) or f"({y})" in content]
            if album_years:
                entry["score"] += 0.05
            elif any(y in content for y in years):
                entry["score"] += 0.02
    ranked = sorted(merged.values(), key=lambda e: e["score"], reverse=True)
    return [(e["score"], e["row"]) for e in ranked[:top_k]]


def retrieve(question: str, top_k: int = TOP_K, mode: str = "hybrid",
             embed_model: str | None = None) -> tuple[list, list[tuple[float, dict]]]:
    """Return (query_embedding, results). mode: hybrid | semantic | lexical."""
    model = embed_model or EMBED_MODEL
    if mode not in VALID_MODES:
        mode = "hybrid"
    query_vec = embed_one(model, query_prefix() + question)
    if mode == "semantic":
        return query_vec, semantic_search(query_vec, top_k)
    if mode == "lexical":
        return query_vec, lexical_search(question, top_k)
    # With the local backend (Ollama-free fallback) the text part discriminates
    # far more than the approximate semantics: give it more weight in the RRF fusion.
    if effective_backend(model) == "local":
        return query_vec, hybrid_search(
            query_vec, question, top_k,
            semantic_weight=0.5, lexical_weight=1.5,
        )
    return query_vec, hybrid_search(query_vec, question, top_k)