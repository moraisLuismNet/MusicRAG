import numpy as np


def to_blob(vec) -> bytes:
    return np.asarray(vec, dtype="float32").tobytes()


def from_blob(blob) -> np.ndarray:
    return np.frombuffer(blob, dtype="float32")


def cosine_similarity(a, b) -> float:
    a = np.asarray(a, dtype="float32")
    b = np.asarray(b, dtype="float32")
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))