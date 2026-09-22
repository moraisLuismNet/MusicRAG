"""Locating and playing songs from the audio library.

The library is organized as `AUDIO_DIR/[Initial]/[Artist]/Album (YY)/Song.mp3`
(with optional subfolders inside the album, e.g. double discs). `index_audio()`
dumps every file into the `music_files` table (`path` field = exact location) to
answer "play X" requests by opening the file in AIMP.
"""
import difflib
import os
import re
import subprocess
import unicodedata
from pathlib import Path

from .config import AIMP_EXE, AUDIO_DIR
from .db import get_conn, init_db

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".wma", ".opus", ".aac", ".ape"}

_PARENS_RE = re.compile(r"\([^)]*\)")
_NUM_PREFIX_RE = re.compile(r"^\s*\d{1,2}[.)-]\s*")
_YEAR_RE = re.compile(r"\s*\(\d{2}\)\s*$")

# Verbs tolerant to voice transcriptions ("reproduce"/"producir"/"reprodu"...).
_PLAY_RE = re.compile(
    r"(?:puedes\s+)?(?:(?:re)?produc\w*|pon(?:me|e|er)?\b|suena(?:me)?\b|"
    r"toca(?:r|me)?\b|escucha(?:r|me)?\b|play\b)",
    re.IGNORECASE,
)

_ARTICLES = {"a", "de", "el", "la", "los", "las", "un", "una", "unos", "unas", "al", "del"}
_STOP_ARTIST = _ARTICLES | {"the", "of", "and", "in", "on", "feat", "ft", "featuring"}


def normalize(name: str) -> str:
    """Lowercase, accent-stripped, no parens or symbols: basis for matching."""
    t = name or ""
    t = _PARENS_RE.sub(" ", t)
    t = re.sub(r"[/_\-]+", " ", t)
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c))
    t = re.sub(r"[^a-z0-9 ]", " ", t.lower())
    return " ".join(t.split())


def clean_track(stem: str) -> str:
    return _NUM_PREFIX_RE.sub("", stem).strip()


def index_audio(directory: Path | None = None, progress=None) -> dict:
    """Walk the library and sync `music_files` with the files on disk."""
    init_db()
    directory = Path(directory) if directory else AUDIO_DIR
    if not directory.is_dir():
        raise FileNotFoundError(f"No existe la biblioteca de audio: {directory}")

    disk: dict[Path, tuple] = {}
    for initial in directory.iterdir():
        if not initial.is_dir():
            continue
        for artist_dir in initial.iterdir():
            if not artist_dir.is_dir():
                continue
            artist = artist_dir.name.strip("[] ")
            for album_dir in artist_dir.iterdir():
                if not album_dir.is_dir():
                    continue
                album = _YEAR_RE.sub("", album_dir.name).strip()
                for f in album_dir.rglob("*"):
                    if not f.is_file() or f.suffix.lower() not in AUDIO_EXTS:
                        continue
                    track = clean_track(f.stem)
                    disk[str(f)] = (
                        artist, album, track, f.suffix.lower().lstrip("."),
                        normalize(artist), normalize(track),
                    )

    with get_conn() as conn:
        existing = [r["path"] for r in conn.execute("SELECT path FROM music_files")]
        to_delete = set(existing) - set(disk)
        for path in sorted(to_delete):
            conn.execute("DELETE FROM music_files WHERE path = ?", (path,))
        for i, (path, values) in enumerate(sorted(disk.items())):
            conn.execute(
                "INSERT OR REPLACE INTO music_files "
                "(path, artist, album, track, ext, artist_norm, track_norm) "
                "VALUES (?,?,?,?,?,?,?)",
                (path, *values),
            )
            if progress and (i % 500 == 0):
                progress(i / max(len(disk), 1))
    return {"files": len(disk), "removed": len(to_delete)}


def parse_play_intent(text: str) -> dict | None:
    """Detect 'play X' / 'play Y of Z'. Returns {title|None, artist|None}."""
    m = _PLAY_RE.search(text)
    if not m:
        return None
    rest = text[m.end():].strip(" .,!?¡¿")
    if not rest:
        return {"title": None, "artist": None}
    rest = re.sub(r"^\s*me\s+", "", rest, count=1)  # "pon me X" -> "X"
    # "A de B de C" -> artist = last segment ("C"), title = "A de B".
    parts = re.split(r"\s+(?:de|del)\s+", rest)
    if len(parts) >= 2:
        title = " ".join(parts[:-1]).strip(" .,!?-") or None
        return {"title": title, "artist": parts[-1].strip()}
    return {"title": rest, "artist": None}


def _strip_articles(norm: str) -> str:
    toks = norm.split()
    while toks and toks[0] in _ARTICLES:
        toks = toks[1:]
    return " ".join(toks)


def _token_sim(query_toks: list[str], title_words: list[str]) -> float:
    if not query_toks:
        return 0.0
    total = 0.0
    for qt in query_toks:
        total += max(
            (difflib.SequenceMatcher(None, qt, w).ratio() for w in title_words),
            default=0.0,
        )
    return total / len(query_toks)


def _resolve_artist(conn, norm_artist: str) -> str | None:
    """Return the real artist_norm that matches the requested text.
    First exact LIKE; otherwise the closest difflib match (>=0.85), which joins
    variants like 'metalica' -> 'metallica' or ASR errors."""
    row = conn.execute(
        "SELECT DISTINCT artist_norm FROM music_files "
        "WHERE artist_norm LIKE ? LIMIT 1", (f"%{norm_artist}%",),
    ).fetchone()
    if row:
        return row[0]
    best, best_r = None, 0.0
    for (a,) in conn.execute("SELECT DISTINCT artist_norm FROM music_files"):
        if not a or a in _STOP_ARTIST:
            continue
        r = difflib.SequenceMatcher(None, norm_artist, a).ratio()
        if r > best_r:
            best, best_r = a, r
    return best if best_r >= 0.85 else None


def find_audio(title: str | None, artist: str | None = None) -> dict | None:
    """Find the best match (title ± artist) in music_files."""
    init_db()
    norm_artist = _strip_articles(normalize(artist)) if artist else None
    norm_title = normalize(title) if title else None
    query_toks = norm_title.split() if norm_title else []

    with get_conn() as conn:
        if norm_artist:
            resolved = _resolve_artist(conn, norm_artist)
            rows = []
            if resolved:
                rows = conn.execute(
                    "SELECT * FROM music_files WHERE artist_norm LIKE ?",
                    (f"%{resolved}%",),
                ).fetchall()
            scoped = bool(rows)
            if not rows:
                # The artist did not match: retry without filter and without bonus.
                rows = conn.execute("SELECT * FROM music_files").fetchall()
                norm_artist = None
        else:
            rows = conn.execute("SELECT * FROM music_files").fetchall()
            scoped = False

        if not rows:
            return None

        if not norm_title:
            if scoped:
                row = rows[0]
                return dict(row) | {"score": 1.0}
            return None

        def _best(rows, need_full: float, use_bonus: bool):
            bonus = 0.15 if use_bonus else 0.0
            best = None
            for r in rows:
                words = r["track_norm"].split()
                full = difflib.SequenceMatcher(None, norm_title, r["track_norm"]).ratio()
                tok = _token_sim(query_toks, words)
                pen = 0.03 if "live" in r["track"].lower() else 0.0
                score = 0.6 * full + 0.4 * tok + bonus - pen
                if full >= need_full and (best is None or score > best["score"]):
                    best = dict(r) | {"score": score}
            return best

        # Without an artist to narrow it, require a very strong title match.
        best = _best(rows, 0.4 if scoped else 0.8, scoped)
        fell_back = False
        split = False
        if best is None and scoped:
            # The artist exists but no title matches: search the whole library
            # with a high threshold (e.g. another artist's version).
            best = _best(conn.execute("SELECT * FROM music_files").fetchall(),
                         0.8, use_bonus=False)
            fell_back = bool(best)
        if not scoped:
            # "reproduce metálica" (artist only): if the whole phrase matches an
            # artist, play its first indexed track.
            resolved = _resolve_artist(conn, " ".join(query_toks))
            if resolved:
                r = conn.execute(
                    "SELECT * FROM music_files WHERE artist_norm LIKE ? "
                    "ORDER BY id LIMIT 1", (f"%{resolved}%",),
                ).fetchone()
                if r:
                    return dict(r) | {"score": 1.0}
        if not scoped and len(query_toks) >= 2:
            # Voice/type without "de": "abba dancing queen", "rand hills iron
            # maiden" or "de avalanches because i'm me" (stray "de Artista").
            # The artist may be the PREFIX or the SUFFIX: keep the best one.
            toks_variants = [query_toks]
            if query_toks[0] in _ARTICLES:
                toks_variants.append(query_toks[1:])

            def _artist_ok(toks: list[str]) -> bool:
                # Allow short artists ("ASH", "U2") unless they are stop words;
                # the real filter comes from _resolve_artist() plus the
                # recursion threshold (scoped title >= 0.8).
                return (len(toks) >= 2
                        or (len(toks) == 1 and len(toks[0]) >= 2
                            and toks[0] not in _STOP_ARTIST))

            candidates: list[dict] = []
            for toks in toks_variants:
                if len(toks) < 2:
                    continue
                n = len(toks)
                for i in range(1, n):
                    for head, tail in ((toks[:i], toks[i:]),
                                       (toks[i:], toks[:i])):
                        if not tail or not _artist_ok(head):
                            continue
                        artist = " ".join(head)
                        if _resolve_artist(conn, artist):
                            cand = find_audio(" ".join(tail), artist)
                            if cand:
                                candidates.append(cand)
            if candidates:
                cand_best = max(candidates, key=lambda c: c["score"])
                if best is None or cand_best["score"] >= best["score"]:
                    best = cand_best
                    split = True
    if best:
        # Split candidates already passed their own validation (real artist +
        # scoped threshold); additionally require score >= 0.8 so we do not
        # misfire on short/suggested artists ("ash", "way", ...).
        threshold = 0.85 if fell_back else (0.8 if split or not scoped else 0.7)
        if best["score"] >= threshold:
            return best
    return None


def play_file(path: str) -> bool:
    """Open the file in AIMP (single-instance)."""
    if not Path(path).exists():
        return False
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        subprocess.Popen([AIMP_EXE, path], **kwargs)
        return True
    except OSError:
        return False


def handle_play_request(text: str) -> dict:
    """Full pipeline. Returns {matched, reply, info}."""
    parsed = parse_play_intent(text)
    if parsed is None:
        return {"matched": False, "reply": "", "info": None}

    wanted = (parsed["title"] or "algo").strip()
    if not parsed["title"] and not parsed["artist"]:
        return {
            "matched": True,
            "reply": "¿Qué canción quieres que ponga? Prueba: «reproduce X de ARTISTA».",
            "info": None,
        }
    info = find_audio(parsed["title"], parsed["artist"])
    if info is None:
        return {
            "matched": True,
            "reply": (
                f"No encuentro «{wanted}»"
                + (f" de {parsed['artist']}" if parsed["artist"] else "")
                + ' en la biblioteca. ¿Has ejecutado `musicrag index-audio`?'
            ),
            "info": None,
        }
    if play_file(info["path"]):
        return {
            "matched": True,
            "reply": (
                f"Reproduciendo en AIMP:\n\n**{info['artist']} — {info['track']}**\n"
                f"`{info['path']}`"
            ),
            "info": info,
        }
    return {
        "matched": True,
        "reply": f"Encontrado, pero no pude abrir AIMP ({AIMP_EXE}).\n`{info['path']}`",
        "info": info,
    }