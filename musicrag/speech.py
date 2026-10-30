"""Voice input (100% local).

Preferred engine: faster-whisper (model `Systran/faster-whisper-small`,
≈460 MB, downloaded once from Hugging Face into the user cache).
Fallback: Vosk with `vosk-model-small-es-0.42` (≈40 MB) in `data/`.

Whisper understands Spanish mixed with English titles much better
("play run to the hills by iron maiden"), which is exactly what Vosk's
speech recognizer degraded.
"""
import json
import os
import queue
import time
import unicodedata
import urllib.request
import zipfile
from pathlib import Path

from .config import DATA_DIR

WHISPER_MODEL = os.environ.get("MUSICRAG_WHISPER_MODEL", "small")
WHISPER_REPO = f"Systran/faster-whisper-{WHISPER_MODEL}"

VOSK_MODEL_NAME = "vosk-model-small-es-0.42"
VOSK_MODEL_URL = f"https://alphacephei.com/vosk/models/{VOSK_MODEL_NAME}.zip"
VOSK_MODEL_DIR = DATA_DIR / VOSK_MODEL_NAME

SAMPLE_RATE = 16000
MAX_SECONDS = 10
_SILENCE_RMS = 0.006         # silence threshold (normalized audio -1..1)
_SILENCE_SECONDS = 1.5       # stops recording once you stopped talking
_SILENCE_BLOCK = 4000        # samples per block (0.25 s at 16 kHz)
_NO_SPEECH_CANCEL = 8.0      # if no voice detected within that time, gives up
_MIN_RMS = 0.002             # below: residual signal, do not transcribe

_last_stats: dict | None = None

_LOOPBACK_HINTS = ("mezcla", "stereo mix", "altavoz", "speaker", "output",
                   "loopback", "monitor", "what u hear")


def last_stats() -> dict | None:
    """Summary of the last recording (for UI/CLI diagnostics)."""
    return _last_stats


def _unnamestd(name: str) -> str:
    name = unicodedata.normalize("NFKD", name or "")
    name = "".join(c for c in name if not unicodedata.combining(c)).lower()
    return name.replace("soft", " ")


def _picker(sd) -> int | None:
    devs = sd.query_devices()
    inputs = [i for i in range(len(devs)) if devs[i]["max_input_channels"] > 0]
    if not inputs:
        return None
    for i in inputs:
        name = _unnamestd(devs[i]["name"])
        if ("microfono" in name or "microphone" in name
                or "headset" in name or " mic " in name
                or name.startswith("mic ")):
            return i
    for i in inputs:
        name = _unnamestd(devs[i]["name"])
        if not any(h in name for h in _LOOPBACK_HINTS):
            return i
    return inputs[0]


def _preferred_input_device(sd, device) -> int | str | None:
    """Microphone for recording: the chosen one, or the Windows default if it is
    not loopback, or the first real hardware device."""
    if device is not None:
        return device
    defd = sd.default.device
    if isinstance(defd, (tuple, list)):
        default_in = defd[0]
    elif isinstance(defd, int):
        default_in = defd
    else:
        default_in = None
    if isinstance(default_in, int):
        try:
            name = _unnamestd(sd.query_devices(default_in)["name"])
        except Exception:  # noqa: BLE001
            name = ""
        if not any(h in name for h in _LOOPBACK_HINTS):
            return default_in
    return _picker(sd)

# Spanish nudge so Whisper keeps English titles ("run to the hills") as
# pronounced, instead of rewriting them.
_INITIAL_PROMPT = (
    "We are using voice commands to request a song to play on a music player."
)


def engine() -> str:
    """Recognition engine available: 'whisper', 'vosk' or ''."""
    try:
        import faster_whisper  # noqa: F401
        return "whisper"
    except ImportError:
        pass
    try:
        import vosk  # noqa: F401
        return "vosk"
    except ImportError:
        pass
    return ""


def _whisper_cached() -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(WHISPER_REPO, local_files_only=True)
        return True
    except Exception:
        return False


def ensure_model(progress=None, force: bool = False) -> Path:
    """Prepare the preferred engine's model. Returns its directory/cache."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if engine() == "whisper":
        from huggingface_hub import snapshot_download
        print(f"Download voice model ({WHISPER_REPO}) if missing...")
        path = snapshot_download(WHISPER_REPO)
        return Path(path)
    if VOSK_MODEL_DIR.exists() and not force:
        return VOSK_MODEL_DIR
    import tempfile

    tmp = Path(tempfile.mkstemp(suffix=".zip", dir=str(DATA_DIR))[1])
    try:
        def _report(blocks, block_size, total):
            if progress and total:
                progress(min(blocks * block_size / total, 1.0))
        print(f"Download voice model ({VOSK_MODEL_NAME})...")
        urllib.request.urlretrieve(VOSK_MODEL_URL, tmp, reporthook=_report)
        with zipfile.ZipFile(tmp) as z:
            z.extractall(DATA_DIR)
    finally:
        tmp.unlink(missing_ok=True)
    if not VOSK_MODEL_DIR.exists():
        raise RuntimeError(f"The voice model could not be prepared in {VOSK_MODEL_DIR}")
    return VOSK_MODEL_DIR


def available() -> tuple[bool, str]:
    """(supports_voice, reason). Checks dependencies and model."""
    try:
        import sounddevice  # noqa: F401
    except ImportError:
        return False, (
            "The microphone dependency is missing. From MusicRAG:\n"
            "pip install sounddevice"
        )
    eng = engine()
    if eng == "whisper":
        if _whisper_cached():
            return True, ""
        return False, (
            "The voice model is missing. Run `python -m musicrag voice-setup` "
            "(it downloads only once, ≈460 MB)."
        )
    if eng == "vosk":
        if VOSK_MODEL_DIR.exists():
            return True, ""
        return False, (
            "The voice model is missing. Run `python -m musicrag voice-setup`."
        )
    return False, (
        "Missing voice dependencies. From MusicRAG:\n"
        "pip install faster-whisper sounddevice"
    )


def _record_pcm(max_seconds: int, device: int | str | None = None) -> bytes:
    """Record PCM int16 @16 kHz mono; stops after trailing silence following voice."""
    global _last_stats
    import numpy as np
    import sounddevice as sd

    audio = queue.Queue()
    chunks: list[bytes] = []
    speech_heard = False
    silent_blocks = 0
    quiet_blocks = 0
    speaking_blocks = _SILENCE_SECONDS * SAMPLE_RATE / _SILENCE_BLOCK

    def _callback(indata, frames, time_info, status):
        audio.put(bytes(indata))

    deadline = time.monotonic() + max_seconds
    rms_acc = 0.0
    device = _preferred_input_device(sd, device)
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE, blocksize=_SILENCE_BLOCK, dtype="int16",
        channels=1, callback=_callback, device=device,
    ):
        while time.monotonic() < deadline:
            try:
                data = audio.get(timeout=0.2)
            except queue.Empty:
                continue
            chunks.append(data)
            rms = float(np.abs(
                np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            ).mean())
            rms_acc += rms
            if rms >= _SILENCE_RMS:
                speech_heard = True
                silent_blocks = 0
                quiet_blocks = 0
            elif speech_heard:
                silent_blocks += 1
                if silent_blocks >= speaking_blocks:
                    break
            else:
                quiet_blocks += 1
                if quiet_blocks >= _NO_SPEECH_CANCEL * (SAMPLE_RATE / _SILENCE_BLOCK):
                    break
    pcm = b"".join(chunks)
    n_blocks = max(len(chunks), 1)
    dev_name = (sd.query_devices(device)["name"] if device is not None else "ninguno")
    _last_stats = {
        "device": f"{device}: {dev_name}",
        "seconds": len(pcm) / SAMPLE_RATE / 2 if pcm else 0.0,
        "rms": rms_acc / n_blocks,
    }
    return pcm


_WHISPER_CACHE: dict[str, object] = {}


def _transcribe_whisper(pcm: bytes, lang: str | None) -> str:
    import numpy as np
    from faster_whisper import WhisperModel

    model = _WHISPER_CACHE.get(WHISPER_MODEL)
    if model is None:
        model = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        _WHISPER_CACHE[WHISPER_MODEL] = model

    arr = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
    if arr.size == 0:
        return ""
    # vad_filter + without_timestamps avoid Whisper hallucinations on
    # background noise/music (e.g. "subscribe!"). Slightly low VAD threshold so
    # we do not drop speech that competes with background music.
    segments, _info = model.transcribe(
        arr, language=lang or None, beam_size=5,
        condition_on_previous_text=False,
        initial_prompt=_INITIAL_PROMPT,
        vad_filter=True,
        vad_parameters={"threshold": 0.30},
        without_timestamps=True,
        no_speech_threshold=0.6,
        compression_ratio_threshold=2.4,
    )
    return " ".join(seg.text.strip() for seg in segments).strip()


def _transcribe_vosk(pcm: bytes) -> str:
    import vosk

    model = vosk.Model(str(VOSK_MODEL_DIR))
    recognizer = vosk.KaldiRecognizer(model, SAMPLE_RATE)
    parts: list[str] = []
    for i in range(0, len(pcm), 8000):
        if recognizer.AcceptWaveform(pcm[i:i + 8000]):
            text = json.loads(recognizer.Result()).get("text", "")
            if text:
                parts.append(text)
    final = json.loads(recognizer.FinalResult()).get("text", "")
    if final:
        parts.append(final)
    return " ".join(parts).strip()


def record_and_recognize(lang: str = "es", max_seconds: int = MAX_SECONDS,
                         device: int | str | None = None) -> str:
    """Record from the microphone (stops at silence) and transcribe offline."""
    eng = engine()
    pcm = _record_pcm(max_seconds, device)
    if not pcm:
        return ""
    if _last_stats and _last_stats["rms"] < _MIN_RMS:
        # Residual signal (silence/background noise): do not transcribe to
        # avoid Whisper hallucinations ("subscribe!").
        return ""
    if eng == "whisper":
        return _transcribe_whisper(pcm, lang)
    return _transcribe_vosk(pcm)


def list_devices() -> list[str]:
    import sounddevice as sd
    return [f"{d['name']} (in: {d['max_input_channels']})"
            for d in sd.query_devices() if d["max_input_channels"] > 0]
