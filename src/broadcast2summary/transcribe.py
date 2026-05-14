from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import json


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str

    def timestamp(self) -> str:
        return _fmt_hms(self.start)


@dataclass(frozen=True)
class TranscriptionResult:
    language: str
    segments: list[Segment]

    def full_text(self) -> str:
        return "".join(s.text for s in self.segments)

    def chunked_for_summary(self, *, max_chars: int = 6000) -> list[str]:
        """Chunk for the summarizer. Each chunk preserves timestamp markers."""
        chunks: list[str] = []
        buf: list[str] = []
        buf_len = 0
        for s in self.segments:
            line = f"[{_fmt_hms(s.start)}] {s.text.strip()}\n"
            if buf_len + len(line) > max_chars and buf:
                chunks.append("".join(buf))
                buf, buf_len = [], 0
            buf.append(line)
            buf_len += len(line)
        if buf:
            chunks.append("".join(buf))
        return chunks


class TranscribeBackend(Protocol):
    def transcribe(self, audio_path: Path) -> TranscriptionResult: ...


class StubBackend:
    """Loads a pre-recorded transcript JSON. Used in tests and `test` CLI mode."""

    def __init__(self, fixture_path: Path):
        self.fixture_path = fixture_path

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        data = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        return TranscriptionResult(
            language=data.get("language", "zh"),
            segments=[Segment(**s) for s in data["segments"]],
        )


class FasterWhisperBackend:
    """Real backend. Imports faster_whisper lazily so tests don't need CTranslate2 runtime."""

    def __init__(self, model_size: str = "large-v3-turbo", device: str = "cpu",
                 compute_type: str = "int8", language_hint: str | None = None):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language_hint = language_hint
        self._model = None

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel  # lazy import
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        model = self._load()
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=self.language_hint,
            vad_filter=True,
        )
        segs = [Segment(start=s.start, end=s.end, text=s.text) for s in segments_iter]
        return TranscriptionResult(language=info.language, segments=segs)


def transcribe_audio(audio_path: Path, *, backend: TranscribeBackend) -> TranscriptionResult:
    return backend.transcribe(audio_path)


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
