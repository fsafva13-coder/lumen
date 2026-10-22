"""Spoken prompts using the operating system's built-in voice (no downloads).

Windows: one hidden PowerShell process drives System.Speech and stays open,
so each phrase starts immediately and a new phrase cuts off the previous one.
macOS uses `say`, Linux uses `espeak` if installed. If no voice is available
the calls do nothing: speech is a convenience, never a requirement.
"""
from __future__ import annotations

import atexit
import os
import shutil
import subprocess
import sys

_PS_LOOP = (
    "Add-Type -AssemblyName System.Speech; "
    "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; $s.Rate = 0; "
    "while (($l = [Console]::In.ReadLine()) -ne $null) "
    "{ $s.SpeakAsyncCancelAll(); if ($l) { $s.SpeakAsync($l) | Out-Null } }"
)


class Voice:
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self._proc: subprocess.Popen | None = None
        self._last: subprocess.Popen | None = None
        if enabled and os.name == "nt":
            try:
                self._proc = subprocess.Popen(
                    ["powershell", "-NoProfile", "-NonInteractive", "-Command", _PS_LOOP],
                    stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
                atexit.register(self.close)
            except Exception:
                self._proc = None

    def say(self, text: str) -> None:
        """Speak without blocking. Never raises."""
        if not self.enabled:
            return
        text = " ".join(str(text).split())
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.stdin.write(text + "\n")
                self._proc.stdin.flush()
            elif sys.platform == "darwin" and shutil.which("say"):
                self._one_shot(["say", text])
            elif os.name != "nt" and shutil.which("espeak"):
                self._one_shot(["espeak", text])
        except Exception:
            self.enabled = False

    def _one_shot(self, cmd: list[str]) -> None:
        if self._last is not None and self._last.poll() is None:
            self._last.terminate()
        self._last = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def close(self) -> None:
        try:
            if self._proc is not None and self._proc.poll() is None:
                self._proc.stdin.close()
                self._proc.terminate()
        except Exception:
            pass
