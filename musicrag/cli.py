import argparse
import json
import sys

from . import __version__
from .config import EMBED_MODEL, KNOWLEDGE_DIR, LLM_MODEL, MUSIC_DB_PATH, TOP_K, VALID_MODES
from .db import db_stats, init_db, list_documents, reset_db
from .ingest import ingest_dir, ingest_musicdb
from .ollama_client import list_all_models
from .search import retrieve


def _print_results(results):
    for score, row in results:
        meta = json.loads(row.get("meta") or "{}") if isinstance(row.get("meta"), str) else (row.get("meta") or {})
        year = f" [{meta.get('year')}]" if meta.get("year") else ""
        print(f"\n[{score:.4f}] {row['title']}{year} ({row['doc_type']}) - {row['source']}")
        print(row["content"][:400].replace("\n", " "))


def cmd_ingest(args):
    results = ingest_dir(files=args.files, model=args.embed_model)
    for r in results["ok"]:
        print(f"[ok] {r['path']} -> {r['chunks']} chunks")
    for e in results["errors"]:
        print(f"[error] {e['path']}: {e['error']}", file=sys.stderr)
    print(f"\nTotal: {len(results['ok'])} documentos, "
          f"{sum(r['chunks'] for r in results['ok'])} chunks, "
          f"{len(results['errors'])} errores")


def cmd_search(args):
    init_db()
    _qv, results = retrieve(args.query, top_k=args.top_k, mode=args.mode,
                            embed_model=args.embed_model)
    print(f"modo={args.mode} | results={len(results)}")
    _print_results(results)


def cmd_ask(args):
    from .rag import stream_answer
    init_db()
    results, stream = stream_answer(args.question, top_k=args.top_k, mode=args.mode,
                                    model=args.model, embed_model=args.embed_model)
    print()
    for part in stream:
        print(part, end="", flush=True)
    print("\n\n--- Sources ---")
    for i, (score, row) in enumerate(results, 1):
        print(f"{i}. {row['title']} ({row['doc_type']}) - {score:.3f}")


def cmd_ingest_musicdb(args):
    res = ingest_musicdb(args.musicdb, model=args.embed_model)
    print(
        f"[ok] music.db -> {res['artists']} artists, {res['albums']} albums, "
        f"{res['chunks']} chunks in {res['documents']} documents"
    )


def cmd_docs(_args):
    init_db()
    for d in list_documents():
        print(f"{d['id']:>4} | {d['n_chunks']:>3} chunks | {d['doc_type']:<8} | {d['title']} | {d['source']}")


def cmd_stats(_args):
    init_db()
    s = db_stats()
    print(json.dumps(s, indent=2, ensure_ascii=False))


def cmd_models(_args):
    models = list_all_models()
    if models:
        for m in models:
            print(m)
    else:
        print("Could not connect to Ollama or there are no models.")


def cmd_reset(_args):
    reset_db()
    print("Database restarted.")


def cmd_voice(args):
    from .speech import record_and_recognize
    text = record_and_recognize(max_seconds=args.max_seconds, device=args.device)
    print(text)


def cmd_mic_test(args):
    import musicrag.speech as speech
    from .speech import (_transcribe_whisper, _transcribe_vosk,
                         _record_pcm, engine, last_stats, list_devices)
    print("Microphones detected:")
    for d in list_devices():
        print("  -", d)
    print(f"Voice engine: {engine() or 'none'}")
    print(f"Recording {args.seconds} s... speak whenever you want.")
    pcm = _record_pcm(args.seconds, args.device)
    st = last_stats()
    print(f"Caught: {st['seconds']:.1f} s | Mean RMS {st['rms']:.4f} | "
          f"device: {st['device']}")
    if not pcm:
        print("Nothing was captured (microphone muted or incorrectly selected?).")
        return
    if st["rms"] < speech._MIN_RMS:
        print(
            f"Signal level VERY low ({st['rms']:.4f} < {speech._MIN_RMS}). "
            "Skipping transcription (Whisper hallucinates 'Subscribe!' when there is noise). "
            "Move closer to the microphone or increase the device volume."
        )
        return
    print(f"Raw (without VAD, Vosk): {_transcribe_vosk(pcm)!r}")
    if engine() == "whisper":
        print(f"Whisper+VAD         : {_transcribe_whisper(pcm, 'es')!r}")


def cmd_voice_setup(_args):
    from .speech import ensure_model
    path = ensure_model()
    print(f"[ok] Voice model ready in {path}")


def cmd_speak(args):
    from .tts import available, speak

    voice = available()
    if not voice:
        print("[!] No Windows TTS voice is installed..")
        return 1
    print(f"Voice: {voice}")
    if not speak(args.text, rate=args.rate, wait=True):
        print("[!] The text could not be played as speech..")
        return 1
    return 0


def cmd_purge_knowledge(_args):
    from .db import get_conn, purge_sources
    from .embed_backend import recompute_idf_from_chunks
    from .config import KNOWLEDGE_DIR

    with get_conn() as conn:
        sources = [
            r["source"]
            for r in conn.execute(
                "SELECT source FROM documents "
                "WHERE source LIKE '%\\knowledge\\%' OR source LIKE '%/knowledge/%'"
            )
        ]
        if not sources:
            print("There are no knowledge/ documents in musicrag.db.")
        else:
            purge_sources(sources)
            recompute_idf_from_chunks(conn)
            print(f"Purged {len(sources)} knowledge/ documents and recalculated embed_idf.json.")

    removed = 0
    for f in sorted(KNOWLEDGE_DIR.rglob("*")):
        if f.is_file():
            f.unlink()
            removed += 1
    for d in sorted(KNOWLEDGE_DIR.rglob("*"), reverse=True):
        if d.is_dir():
            try:
                d.rmdir()
            except OSError:
                pass
    try:
        KNOWLEDGE_DIR.rmdir()
    except OSError:
        pass
    print(f"knowledge/: {removed} deleted files.")


def cmd_sync(args):
    from .config import MUSIC_DB_PATH
    from .db import get_conn, purge_sources
    from .embed_backend import recompute_idf_from_chunks
    from .ingest import DEFAULT_SOURCE_MUSICDB as DEFAULT_SOURCE, load_catalog

    musicdb = args.musicdb or MUSIC_DB_PATH
    try:
        artists, albums, _ = load_catalog(musicdb)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1
    valid = {f"{DEFAULT_SOURCE}#artist:{a['id']}" for a in artists} | {
        f"{DEFAULT_SOURCE}#album:{a['id']}" for a in albums
    }
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT source FROM documents WHERE source LIKE ?",
            (f"{DEFAULT_SOURCE}#%",),
        ).fetchall()
        stored = {r["source"] for r in rows}
        orphans = sorted(stored - valid)
        if orphans:
            purge_sources(orphans)
            recompute_idf_from_chunks(conn)
            print(f"{len(orphans)} orphaned documents deleted (no longer exist in {musicdb}).")
        else:
            print(f"No orphans left behind: {len(stored)} synchronized music.db documents.")
        missing = len(valid - stored)
        if missing:
            print(f"Warning: {missing} catalog records not yet ingested; run `musicrag ingest-musicdb`.")
    return 0


def cmd_ingest_new(args):
    from .ingest import ingest_drop, ingest_new, reindex

    def _fmt(res: dict, origin: str) -> None:
        print(
            f"[{origin}] {res['artists']} artistas y {res['albums']} álbumes "
            f"({res['chunks']} chunks in {res['documents']} documents)"
        )

    if args.directory:
        _fmt(ingest_new(args.directory), "new_sqlite")
    else:
        _fmt(ingest_new(), "new_sqlite/")
        _fmt(ingest_drop(), "ingest/")
    reindex()
    print("IDF and embeddings recalculated using the entire corpus.")


def cmd_purge_new(_args):
    from .db import get_conn, purge_sources
    from .embed_backend import recompute_idf_from_chunks
    from .ingest import DEFAULT_SOURCE_DROP, DEFAULT_SOURCE_NEW

    with get_conn() as conn:
        sources = [r["source"] for r in conn.execute(
            "SELECT source FROM documents WHERE source LIKE ? OR source LIKE ?",
            (f"{DEFAULT_SOURCE_NEW}#%", f"{DEFAULT_SOURCE_DROP}#%"),
        )]
        if not sources:
            print("There are no documents from new_sqlite/ or ingest/ in musicrag.db.")
        else:
            purge_sources(sources)
            recompute_idf_from_chunks(conn)
            print(f"Purged {len(sources)} docs (new_sqlite/ + ingest/) and recalculated embed_idf.json.")


def cmd_index_audio(args):
    from .player import index_audio

    try:
        res = index_audio(args.directory, progress=None)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1
    msg = f"Indexed {res['files']} audio files in musicrag.db."
    if res["removed"]:
        msg += f" ({res['removed']} obsolete items removed)"
    print(msg + ".")


def cmd_play(_args):
    from .player import handle_play_request

    res = handle_play_request(_args.text)
    if not res["matched"]:
        print("It does not detect any intention to reproduce in: " + _args.text)
        return 1
    print(res["reply"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="musicrag",
        description=f"MusicRAG v{__version__} - RAG musical 100% local (SQLite + Ollama)")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("ingest-musicdb", help="ingest the sqlite/music.db catalog")
    s.add_argument("--musicdb", default=MUSIC_DB_PATH, help="path to music.db")
    s.add_argument("--embed-model", default=EMBED_MODEL)
    s.set_defaults(func=cmd_ingest_musicdb)

    s = sub.add_parser("ingest-new", help="ingest the text catalog (new_sqlite/ + drop ingest/ folder)")
    s.add_argument("--directory", default=None, help="path to new_sqlite (default config)")
    s.add_argument("--embed-model", default=EMBED_MODEL)
    s.set_defaults(func=cmd_ingest_new)

    s = sub.add_parser("purge-new", help="remove documents from new_sqlite/ and ingest/ from musicrag.db")
    s.set_defaults(func=cmd_purge_new)

    s = sub.add_parser("ingest", help="ingest knowledge (or specific files) into SQLite")
    s.add_argument("files", nargs="*", help="Optional files; otherwise, all knowledge/")
    s.add_argument("--embed-model", default=EMBED_MODEL, help="embedding model")
    s.set_defaults(func=cmd_ingest)

    s = sub.add_parser("search", help="semantic/textual/hybrid search")
    s.add_argument("query")
    s.add_argument("--mode", choices=VALID_MODES, default="hybrid")
    s.add_argument("--top-k", type=int, default=TOP_K)
    s.add_argument("--embed-model", default=EMBED_MODEL)
    s.set_defaults(func=cmd_search)

    s = sub.add_parser("ask", help="LLM response with retrieved context (RAG)")
    s.add_argument("question")
    s.add_argument("--mode", choices=VALID_MODES, default="hybrid")
    s.add_argument("--top-k", type=int, default=TOP_K)
    s.add_argument("--model", default=LLM_MODEL, help="Ollama chat model")
    s.add_argument("--embed-model", default=EMBED_MODEL)
    s.set_defaults(func=cmd_ask)

    s = sub.add_parser("docs", help="list ingested documents")
    s.set_defaults(func=cmd_docs)

    s = sub.add_parser("stats", help="Knowledge base statistics")
    s.set_defaults(func=cmd_stats)

    s = sub.add_parser("models", help="list Ollama models")
    s.set_defaults(func=cmd_models)

    s = sub.add_parser("reset", help="delete the knowledge base")
    s.set_defaults(func=cmd_reset)

    s = sub.add_parser("purge-knowledge", help="remove from musicrag.db and delete the contents of knowledge/")
    s.set_defaults(func=cmd_purge_knowledge)

    s = sub.add_parser("sync", help="delete orphaned docs (no longer exist in sqlite/music.db)")
    s.add_argument("--musicdb", default=None, help="path to music.db (e.g. sqlite/music.db)")
    s.set_defaults(func=cmd_sync)

    s = sub.add_parser("play", help="play a song in AIMP")
    s.add_argument("text")
    s.set_defaults(func=cmd_play)

    s = sub.add_parser("index-audio", help="Index music from the [AUDIO] library.")
    s.add_argument("--directory", default=None, help="path to the audio library")
    s.set_defaults(func=cmd_index_audio)

    s = sub.add_parser("voice", help="transcribe a question via microphone")
    s.add_argument("--max-seconds", type=int, default=10, help="maximum recording seconds")
    s.add_argument("--device", default=None, help="microphone index or name")
    s.set_defaults(func=cmd_voice)

    s = sub.add_parser("mic-test", help="microphone diagnostics (signal level, VAD, raw)")
    s.add_argument("--seconds", type=int, default=6, help="seconds to record")
    s.add_argument("--device", default=None, help="microphone index or name")
    s.set_defaults(func=cmd_mic_test)

    s = sub.add_parser("voice-setup", help="download/verify the voice model (Whisper/Vosk)")
    s.set_defaults(func=cmd_voice_setup)

    s = sub.add_parser("speak", help="Read text aloud (Windows TTS, offline))")
    s.add_argument("text", help="text to be spoken")
    s.add_argument("--rate", type=int, default=0,
                   help="speech speed (-10 slow ... +10 fast)")
    s.set_defaults(func=cmd_speak)

    return p


def main(argv=None) -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            pass
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()