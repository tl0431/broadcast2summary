"""Consolidate phantom speaker clusters that the LLM named identically.

pyannote can over-segment a single voice into multiple clusters (especially on
English broadcast audio with varying mic/energy). When this happens, the
downstream LLM speaker-id step usually assigns the same canonical name to all
those phantom clusters. This module rewrites the speaker_id of such segments
so a single voice has a single id, fixing transcript output and quality checks
without re-running diarization.
"""
from __future__ import annotations

from dataclasses import dataclass

from .transcribe import Segment


_CONFIDENCE_FLOOR = 0.6  # mirror apply_speaker_names: only confident names merge


@dataclass(frozen=True)
class MergeReport:
    merged_pairs: dict[str, str]  # phantom_sid -> canonical_sid
    canonical_to_name: dict[str, str]  # canonical_sid -> display name
    clusters_before: int
    clusters_after: int


def merge_duplicate_named_speakers(
    segments: list[Segment],
    speaker_names: dict[str, dict | str | None],
) -> tuple[list[Segment], MergeReport]:
    """Merge speaker_ids whose LLM-assigned name is identical (and confident).

    Canonical id = the earliest-appearing speaker_id for each name (stable).
    Segments whose speaker_id maps to a phantom are rewritten to the canonical.
    Low-confidence or anonymous clusters are left untouched.
    """
    name_to_canonical: dict[str, str] = {}
    canonical_to_name: dict[str, str] = {}
    merged: dict[str, str] = {}
    sids_before: set[str] = set()

    for seg in segments:
        sid = seg.speaker_id
        if not sid:
            continue
        sids_before.add(sid)
        if sid in merged or sid in canonical_to_name:
            continue
        entry = speaker_names.get(sid)
        name = _confident_name(entry)
        if not name:
            continue
        key = name.casefold().strip()
        if key in name_to_canonical:
            merged[sid] = name_to_canonical[key]
        else:
            name_to_canonical[key] = sid
            canonical_to_name[sid] = name

    if not merged:
        return segments, MergeReport(
            merged_pairs={},
            canonical_to_name=canonical_to_name,
            clusters_before=len(sids_before),
            clusters_after=len(sids_before),
        )

    rewritten: list[Segment] = []
    for seg in segments:
        sid = seg.speaker_id
        if sid and sid in merged:
            rewritten.append(
                Segment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text,
                    translation=seg.translation,
                    speaker_id=merged[sid],
                    speaker_name=seg.speaker_name,
                )
            )
        else:
            rewritten.append(seg)

    return rewritten, MergeReport(
        merged_pairs=merged,
        canonical_to_name=canonical_to_name,
        clusters_before=len(sids_before),
        clusters_after=len(sids_before) - len(merged),
    )


def _confident_name(entry: dict | str | None) -> str | None:
    """Return the speaker name only if confidence is >= floor; else None."""
    if entry is None:
        return None
    if isinstance(entry, dict):
        name = entry.get("name")
        confidence = float(entry.get("confidence", 0.0))
        if name and confidence >= _CONFIDENCE_FLOOR:
            return str(name)
        return None
    # legacy plain string = confidence 1.0
    return str(entry)
