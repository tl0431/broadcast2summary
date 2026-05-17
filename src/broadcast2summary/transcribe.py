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
    translation: str | None = None
    speaker_id: str | None = None
    speaker_name: str | None = None

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
            line = f"{format_segment_line_for_summary(s)}\n"
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

    def __init__(self, *, cheap: bool = False, language_hint: str | None = None,
                 device: str = "cpu", compute_type: str = "int8",
                 batch_size: int = 8, convert_traditional: bool = True):
        self.model_size = "small" if cheap else "large-v3-turbo"
        self.device = device
        self.compute_type = compute_type
        self.language_hint = language_hint
        self.batch_size = batch_size
        self.convert_traditional = convert_traditional
        self._model = None
        self._batched = None
        self._cc = None

    def _load(self):
        if self._batched is None:
            from faster_whisper import WhisperModel, BatchedInferencePipeline
            self._model = WhisperModel(
                self.model_size, device=self.device, compute_type=self.compute_type
            )
            self._batched = BatchedInferencePipeline(model=self._model)
        return self._batched

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        import sys
        pipeline = self._load()
        segments_iter, info = pipeline.transcribe(
            str(audio_path),
            language=self.language_hint,
            batch_size=self.batch_size,
            vad_filter=True,
        )
        segs: list[Segment] = []
        for i, s in enumerate(segments_iter):
            segs.append(Segment(start=s.start, end=s.end, text=s.text))
            if i > 0 and i % 20 == 0:
                pct = (s.end / info.duration * 100) if getattr(info, "duration", 0) else 0
                print(
                    f"[transcribe] {i} segs, {s.end:.0f}s/{info.duration:.0f}s ({pct:.0f}%)",
                    file=sys.stderr,
                    flush=True,
                )

        info_lang = getattr(info, "language", None)
        if self.convert_traditional and (info_lang == "zh" or self.language_hint == "zh"):
            if self._cc is None:
                from opencc import OpenCC
                self._cc = OpenCC("t2s")
            segs = [
                Segment(start=s.start, end=s.end, text=self._cc.convert(s.text),
                        translation=s.translation)
                for s in segs
            ]
            # Punctuation restoration (zh only; en skipped; ImportError graceful)
            from .punctuate import punctuate_segments
            segs = punctuate_segments(segs, info_lang or "")

        return TranscriptionResult(language=info_lang or "", segments=segs)


class WhisperCppBackend:
    """Apple Metal-accelerated transcription via whisper.cpp (pywhispercpp)."""

    def __init__(
        self,
        *,
        cheap: bool = False,
        language_hint: str | None = None,
        n_threads: int = 4,
        convert_traditional: bool = True,
    ):
        self.model_size = "small" if cheap else "large-v3-turbo"
        self.language_hint = language_hint
        self.n_threads = n_threads
        self.convert_traditional = convert_traditional
        self._model = None
        self._cc = None

    def _load(self):
        if self._model is None:
            from pywhispercpp.model import Model

            self._model = Model(self.model_size, n_threads=self.n_threads)
        return self._model

    def transcribe(self, audio_path: Path) -> TranscriptionResult:
        model = self._load()
        raw_segs = model.transcribe(
            str(audio_path),
            language=self.language_hint or "auto",
        )
        segs = [
            Segment(start=s.t0 / 100.0, end=s.t1 / 100.0, text=s.text.strip())
            for s in raw_segs
        ]
        info_lang = self.language_hint or "zh"
        if self.convert_traditional and info_lang == "zh":
            if self._cc is None:
                from opencc import OpenCC

                self._cc = OpenCC("t2s")
            segs = [
                Segment(
                    start=s.start,
                    end=s.end,
                    text=self._cc.convert(s.text),
                    translation=s.translation,
                )
                for s in segs
            ]
            from .punctuate import punctuate_segments

            segs = punctuate_segments(segs, info_lang)
        return TranscriptionResult(language=info_lang, segments=segs)


def transcribe_audio(audio_path: Path, *, backend: TranscribeBackend) -> TranscriptionResult:
    return backend.transcribe(audio_path)


def format_segment_line_for_summary(seg: Segment) -> str:
    """One transcript line for the summarizer (speaker_id only, before identity resolution)."""
    ts = _fmt_hms(seg.start)
    if seg.speaker_id:
        return f"[{ts}] [{seg.speaker_id}] {seg.text.strip()}"
    return f"[{ts}] {seg.text.strip()}"


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
