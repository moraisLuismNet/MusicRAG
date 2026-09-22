"""Speech synthesis (TTS) 100% local via Windows SAPI (PowerShell).

Uses the Spanish "Microsoft Helena Desktop" voice if installed; otherwise any
system voice. No network or PyPI dependencies.

- `speak(text, wait=False)`: speaks the text without blocking (fire-and-forget).
- `clean_for_speech(text)`: strips markdown so it reads cleanly
  (backticked paths, bold, etc.).
"""
import os
import re
import subprocess
from pathlib import Path

_CREATE_NO_WINDOW = 0x08000000 if os.name == "nt" else 0
_PSCMD = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass"]

_state: dict = {"voice": None, "checked": False}


def _nowin() -> dict:
    return {"creationflags": _CREATE_NO_WINDOW} if os.name == "nt" else {}


def find_voice() -> str | None:
    """Name of an installed system voice (prefers Spanish)."""
    if _state["checked"]:
        return _state["voice"]
    _state["checked"] = True
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        "$es=$s.GetInstalledVoices()|Where-Object{$_.VoiceInfo.Culture.Name"
        " -like 'es-*'}|Select-Object -First 1;"
        "if($es){Write-Output $es.VoiceInfo.Name}"
    )
    try:
        out = subprocess.run(_PSCMD + ["-Command", script],
                             capture_output=True, text=True, timeout=30,
                             **_nowin())
    except Exception:
        return None
    name = out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
    _state["voice"] = name or None
    return _state["voice"]


def available() -> str | None:
    """Available TTS voice (name) or None if missing/failing."""
    return find_voice()


def speak(text: str, rate: int = 0, wait: bool = False) -> bool:
    """Speak `text` through the speakers. Does not block unless `wait=True`."""
    if not text or not find_voice():
        return False
    safe = text.replace("'", "''")
    script = (
        "Add-Type -AssemblyName System.Speech;"
        "$s=New-Object System.Speech.Synthesis.SpeechSynthesizer;"
        f"$s.SelectVoice('{find_voice()}');"
        f"$s.Rate={int(rate)};"
        f"$s.Speak('{safe}')"
    )
    try:
        p = subprocess.Popen(_PSCMD + ["-Command", script], **_nowin())
        if wait:
            p.wait(timeout=180)
        return True
    except Exception:
        return False


def clean_for_speech(text: str) -> str:
    """Text ready for reading aloud (stripped of markdown and paths)."""
    t = re.sub(r"`[^`]*`", "", text or "")      # backticked code/paths
    t = t.replace("**", " ").replace("__", " ")
    t = re.sub(r"[#*>|~_]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t or "Made."