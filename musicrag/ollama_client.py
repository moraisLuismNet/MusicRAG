import threading

import ollama

from .config import EMBED_BATCH_SIZE, OLLAMA_HOST

_client = None
_lock = threading.Lock()

# Heuristic to separate embedding models from chat models.
_EMBED_HINTS = ("embed", "minilm", "bge", "gte", "snowflake")


def get_client() -> ollama.Client:
    global _client
    if _client is None:
        with _lock:
            if _client is None:
                _client = ollama.Client(host=OLLAMA_HOST)
    return _client


def is_embedding_model(name: str) -> bool:
    lower = name.lower()
    return any(h in lower for h in _EMBED_HINTS)


def list_all_models() -> list[str]:
    try:
        res = get_client().list()
        models = getattr(res, "models", None) or res.get("models", [])
        names = []
        for m in models:
            name = getattr(m, "model", None) or getattr(m, "name", None) or (m.get("model") or m.get("name") if isinstance(m, dict) else "")
            if name:
                names.append(name)
        return names
    except Exception:
        return []


def list_chat_models() -> list[str]:
    return [m for m in list_all_models() if not is_embedding_model(m)]


def list_embed_models() -> list[str]:
    return [m for m in list_all_models() if is_embedding_model(m)]


def embed_texts(model: str, texts: list[str], batch_size: int = EMBED_BATCH_SIZE,
                progress=None) -> list[list[float]]:
    """Embed in batches. `progress` is an optional 0..1 callback."""
    if not texts:
        return []
    client = get_client()
    out: list[list[float]] = []
    n = len(texts)
    for start in range(0, n, batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embed(model=model, input=batch)
        out.extend(resp["embeddings"])
        if progress:
            progress(min(start + batch_size, n) / n)
    return out


def embed_one(model: str, text: str) -> list[float]:
    return embed_texts(model, [text])[0]


def chat_stream(model: str, messages: list[dict]):
    return get_client().chat(model=model, messages=messages, stream=True)