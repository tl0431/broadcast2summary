from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import json
import logging

logger = logging.getLogger(__name__)


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
    def transcribe(self, audio_path: Path, *, language: str | None = None) -> TranscriptionResult: ...


class StubBackend:
    """Loads a pre-recorded transcript JSON. Used in tests and `test` CLI mode."""

    def __init__(self, fixture_path: Path):
        self.fixture_path = fixture_path

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> TranscriptionResult:
        data = json.loads(self.fixture_path.read_text(encoding="utf-8"))
        return TranscriptionResult(
            language=data.get("language", "zh"),
            segments=[Segment(**s) for s in data["segments"]],
        )


def _apply_zh_postprocess(
    segments: list[Segment],
    *,
    info_lang: str,
    convert_traditional: bool,
    language_hint: str | None,
) -> list[Segment]:
    """OpenCC + funasr punctuation for CJK-dominant segments only."""
    from .mixed_language import is_cjk_dominant

    zh_mode = convert_traditional and (info_lang == "zh" or language_hint == "zh")
    if not zh_mode or not segments:
        return segments

    cc = None
    converted: list[Segment] = []
    for s in segments:
        text = s.text
        if is_cjk_dominant(text):
            if cc is None:
                from opencc import OpenCC
                cc = OpenCC("t2s")
            text = cc.convert(text)
        converted.append(Segment(
            start=s.start, end=s.end, text=text,
            translation=s.translation,
            speaker_id=s.speaker_id, speaker_name=s.speaker_name,
        ))

    from .punctuate import punctuate_segments
    to_punct = [s for s in converted if is_cjk_dominant(s.text)]
    if not to_punct:
        return converted
    punctuated = punctuate_segments(to_punct, "zh")
    by_id = {id(s): p for s, p in zip(to_punct, punctuated)}
    return [by_id.get(id(s), s) for s in converted]


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

    def release(self) -> None:
        import gc
        self._model = None
        self._batched = None
        gc.collect()
        logger.info("FasterWhisper model released from memory")

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> TranscriptionResult:
        import sys
        pipeline = self._load()
        whisper_lang = language if language is not None else self.language_hint
        segments_iter, info = pipeline.transcribe(
            str(audio_path),
            language=whisper_lang,
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

        info_lang = getattr(info, "language", None) or ""
        segs = _apply_zh_postprocess(
            segs,
            info_lang=info_lang,
            convert_traditional=self.convert_traditional,
            language_hint=language if language is not None else self.language_hint,
        )
        return TranscriptionResult(language=info_lang or whisper_lang or "", segments=segs)


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

    def _resolve_language(self, model, audio_path: Path, *, override: str | None) -> tuple[str, str]:
        """Return (transcribe_language, info_language) for whisper.cpp."""
        if override:
            return override, override
        if self.language_hint:
            return self.language_hint, self.language_hint
        audio_str = str(audio_path)
        detect = getattr(model, "auto_detect_language", None)
        if detect is not None:
            try:
                (lang_str, _prob), _probs = detect(audio_str)
                if lang_str:
                    return lang_str, lang_str
            except Exception:
                logger.warning(
                    "WhisperCpp language auto-detect failed for %s; falling back to auto",
                    audio_path,
                    exc_info=True,
                )
        return "auto", ""

    def release(self) -> None:
        import gc
        self._model = None
        self._cc = None
        gc.collect()
        logger.info("WhisperCpp model released from memory")

    def transcribe(self, audio_path: Path, *, language: str | None = None) -> TranscriptionResult:
        model = self._load()
        transcribe_lang, info_lang = self._resolve_language(
            model, audio_path, override=language,
        )
        raw_segs = model.transcribe(str(audio_path), language=transcribe_lang)
        segs = [
            Segment(start=s.t0 / 100.0, end=s.t1 / 100.0, text=s.text.strip())
            for s in raw_segs
        ]
        segs = _apply_zh_postprocess(
            segs,
            info_lang=info_lang,
            convert_traditional=self.convert_traditional,
            language_hint=language if language is not None else self.language_hint,
        )
        return TranscriptionResult(language=info_lang or "", segments=segs)


def resolve_whisper_language(primary_language: str) -> str | None:
    """Map feed/episode primary language to Whisper language argument."""
    lang = (primary_language or "zh").lower()
    if lang == "mixed":
        return None
    if lang in ("zh", "en"):
        return lang
    return None


def transcribe_audio(
    audio_path: Path,
    *,
    backend: TranscribeBackend,
    primary_language: str = "zh",
) -> TranscriptionResult:
    """Transcribe audio; repair English sections mis-decoded under zh-primary mode."""
    whisper_lang = resolve_whisper_language(primary_language)
    result = backend.transcribe(audio_path, language=whisper_lang)

    primary = (primary_language or "zh").lower()
    if primary in ("zh", "mixed"):
        from .mixed_language import repair_mixed_language_segments
        segments = repair_mixed_language_segments(audio_path, result.segments, backend)
        if segments is not result.segments:
            result = TranscriptionResult(language=result.language, segments=segments)

    return result


def format_segment_line_for_summary(seg: Segment) -> str:
    """One transcript line for the summarizer (speaker_id only, before identity resolution)."""
    ts = _fmt_hms(seg.start)
    if seg.speaker_id:
        return f"[{ts}] [{seg.speaker_id}] {seg.text.strip()}"
    return f"[{ts}] {seg.text.strip()}"


def _fmt_hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"
