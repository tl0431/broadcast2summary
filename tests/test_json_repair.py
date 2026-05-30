import json
from pathlib import Path

import pytest

from broadcast2summary.json_repair import repair_unescaped_quotes_in_json_strings
from broadcast2summary.quality import _parse_summary_json, evaluate

from tests.test_quality import GOOD, TRANSCRIPT

_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "quality"
_A16Z_DEEPSEEK = _FIXTURES / "a16z_deepseek_attempt1.json"
_A16Z_CLAUDE = _FIXTURES / "a16z_claude_attempt3.json"

# Long enough for summary/transcript ratio L1 check on fixture payloads.
_FIXTURE_TRANSCRIPT = "revenue growth venture capital exit threshold " * 3500


def test_repair_unescaped_quotes_parses():
    inner = '这一"颠覆者困境"动态将长期影响前沿模型的定价权和市场份额分配。'
    raw = (
        '{"tldr": "' + "x" * 80 + '", '
        '"key_points": ["' + inner + '"], '
        '"quotes": [], '
        '"chapters": [{"ts_start":"00:00:00","ts_end":"00:10:00","title":"a","summary":"b"},'
        '{"ts_start":"00:10:00","ts_end":"00:30:00","title":"c","summary":"d"},'
        '{"ts_start":"00:30:00","ts_end":"00:55:00","title":"e","summary":"f"}], '
        '"guests": ["G"]}'
    )
    repaired = repair_unescaped_quotes_in_json_strings(raw)
    parsed = json.loads(repaired)
    assert "颠覆者困境" in parsed["key_points"][0]


def test_evaluate_accepts_repaired_claude_json():
    inner = '这一"颠覆者困境"动态将长期影响前沿模型的定价权和市场份额分配。' * 2
    payload = {
        **GOOD,
        "key_points": [
            GOOD["key_points"][0],
            GOOD["key_points"][1],
            inner,
            GOOD["key_points"][3],
            GOOD["key_points"][4],
        ],
    }
    raw = json.dumps(payload, ensure_ascii=False)
    escaped_inner = json.dumps(inner, ensure_ascii=False)
    broken = raw.replace(escaped_inner, f'"{inner}"', 1)
    r = evaluate(broken, transcript=TRANSCRIPT, l3_enabled=False)
    assert r.passed is True, r.reason


def test_a16z_deepseek_fixture_passes_l2_placeholder_fix():
    raw = _A16Z_DEEPSEEK.read_text(encoding="utf-8")
    r = evaluate(raw, transcript=_FIXTURE_TRANSCRIPT, l3_enabled=False)
    assert r.passed is True, r.reason


def test_a16z_claude_fixture_json_repair_no_longer_invalid_json():
    """Claude fallback failed L1 on unescaped quotes; repair must clear that."""
    raw = _A16Z_CLAUDE.read_text(encoding="utf-8")
    parsed, err = _parse_summary_json(raw)
    assert err is None, err
    assert parsed is not None
    r = evaluate(raw, transcript=_FIXTURE_TRANSCRIPT, l3_enabled=False)
    assert "invalid json" not in r.reason.lower()
