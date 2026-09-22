"""Embedding dispatch.

- `ollama`: uses the Ollama embedding model (e.g. nomic-embed-text). Real
  semantic search quality, 100% local.
- `local`: network-free fallback: n-grams (2-3 chars and word bigrams) with
  **TF-IDF weighting computed over the ingested corpus**. Used automatically
  when the Ollama model is unavailable and `EMBED_BACKEND=auto`.
- `auto` (default): Ollama if the model exists, otherwise `local` with a warning.

When the backend is `local`, the 'search_document:'/'search_query:' prefixes
(nomic-embed-text optimization) are NOT added: they would be a constant feature
that TF-IDF would turn into noise.
"""
import hashlib
import json
import math
import re
import warnings
from collections import Counter

# To avoid repeating the local-backend warning on every query/startup.
_WARNED_LOCAL = False

import numpy as np

from .config import DATA_DIR, EMBED_BACKEND, EMBED_DOC_PREFIX, EMBED_QUERY_PREFIX
from .ollama_client import list_all_models

_LOCAL_DIM = 512
_IDF_FILE = DATA_DIR / "embed_idf.json"
_idf_state: dict | None = None


def doc_prefix() -> str:
    return EMBED_DOC_PREFIX if EMBED_BACKEND != "local" else ""


def query_prefix() -> str:
    return EMBED_QUERY_PREFIX if EMBED_BACKEND != "local" else ""


# ---------------------------------------------------------------- local backend
_TOKEN_RE = re.compile(r"[^\W_]+")


def _strip_prefix(text: str) -> str:
    for prefix in (EMBED_DOC_PREFIX, EMBED_QUERY_PREFIX):
        if text.startswith(prefix):
            return text[len(prefix):]
    return text


def _features(text: str) -> Counter:
    tokens = [w for w in _TOKEN_RE.findall(text.lower())
              if (not w.isdigit() and len(w) >= 3)]
    feats: Counter = Counter()
    for word in tokens:
        feats[("w", word)] += 1
    for a, b in zip(tokens, tokens[1:]):
        feats[("w2", f"{a} {b}")] += 1
    return feats


def _weight(feature: tuple, idf: dict | None) -> float:
    if idf is None:
        return 1.0
    n = idf.get("n", 0)
    df = idf.get("df", {}).get(":".join(feature), 0)
    return math.log((n + 1) / (df + 1)) + 1.0


def _embed_vec(text: str, idf: dict | None) -> list[float]:
    vec = np.zeros(_LOCAL_DIM, dtype="float32")
    for feature, count in _features(text).items():
        digest = hashlib.md5(repr(feature).encode("utf-8")).digest()
        idx = int.from_bytes(digest[:4], "little") % _LOCAL_DIM
        sign = 1.0 if digest[4] & 1 else -1.0
        vec[idx] += (count * sign * _weight(feature, idf))
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec.tolist()


def _load_idf() -> dict | None:
    global _idf_state
    if _idf_state is None and _IDF_FILE.exists():
        try:
            _idf_state = json.loads(_IDF_FILE.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            _idf_state = None
    return _idf_state


def _train_idf(texts: list[str]) -> dict:
    df: Counter = Counter()
    for text in texts:
        for feature in _features(text):
            df[":".join(feature)] += 1
    state = {"n": len(texts), "df": dict(df)}
    _IDF_FILE.parent.mkdir(parents=True, exist_ok=True)
    _IDF_FILE.write_text(json.dumps(state), encoding="utf-8")
    return state


def recompute_idf_from_chunks(conn) -> dict:
    """Re-trains `embed_idf.json` on the current DB corpus (e.g. after purging)."""
    contents = [r["content"] for r in conn.execute("SELECT content FROM chunks")]
    global _idf_state
    _idf_state = _train_idf(contents)
    return _idf_state


def embed_local(texts: list[str], progress=None, train: bool = True) -> list[list[float]]:
    texts = [_strip_prefix(t) for t in texts]
    idf = _train_idf(texts) if (train and len(texts) >= 8) else _load_idf()
    n = len(texts)
    out = [_embed_vec(t, idf) for t in texts]
    if progress:
        progress(1.0)
    return out


# ---------------------------------------------------------------- general dispatch
def embed_texts(model: str, texts: list[str], progress=None,
                train: bool = True) -> list[list[float]]:
    """Embed in batches, choosing the backend from config and availability."""
    global _WARNED_LOCAL
    if not texts:
        return []
    if EMBED_BACKEND == "local":
        if not _WARNED_LOCAL:
            _WARNED_LOCAL = True
            warnings.warn("Using LOCAL backend of embeddings (without Ollama).", stacklevel=2)
        return embed_local(texts, progress=progress, train=train)

    available = list_all_models()
    if model in available:
        try:
            from .ollama_client import embed_texts as ollama_embed
            return ollama_embed(model, texts, progress=progress)
        except Exception as exc:  # noqa: BLE001
            if EMBED_BACKEND == "ollama":
                raise
            warnings.warn(
                f"Ollama failed to embed ({exc}); using LOCAL backend.", stacklevel=2)
            return embed_local(texts, progress=progress, train=train)

    if EMBED_BACKEND == "ollama":
        raise RuntimeError(
            f"The embedding model '{model}' is not in Ollama. "
            f"Run: ollama pull {model}"
        )
    if not _WARNED_LOCAL:
        _WARNED_LOCAL = True
        warnings.warn(
            f"The embedding model '{model}' is not available in Ollama "
            f"(run: ollama pull {model}); using LOCAL backend in the meantime.",
            stacklevel=2,
        )
    return embed_local(texts, progress=progress, train=train)


def corpus_term_df(term: str, ratio: float = 0.30) -> float | None:
    """DF of a term in the corpus as a relative frequency (None if no corpus)."""
    state = _load_idf()
    if not state:
        return None
    n = state.get("n", 0)
    df = state.get("df", {}).get(f"w:{term}", 0)
    return (df / n) if n else None


def effective_backend(model: str) -> str:
    """Backend actually in use for embeddings: 'ollama' or 'local'."""
    if EMBED_BACKEND == "local":
        return "local"
    if EMBED_BACKEND == "ollama":
        return "ollama"
    from .ollama_client import list_all_models
    return "ollama" if model in list_all_models() else "local"


def embed_one(model: str, text: str) -> list[float]:
    return embed_texts(model, [text], train=False)[0]