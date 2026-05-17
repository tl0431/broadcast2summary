import subprocess
from pathlib import Path


def _xml_escape_via_bash(value: str) -> str:
    repo = Path(__file__).resolve().parents[1]
    script = repo / "scripts" / "_plist_helpers.sh"
    cmd = f'source "{script}"; xml_escape "{value}"'
    r = subprocess.run(
        ["bash", "-c", cmd],
        capture_output=True,
        text=True,
        check=True,
    )
    return r.stdout


def test_xml_escape_ampersand():
    assert _xml_escape_via_bash("/foo&bar") == "/foo&amp;bar"


def test_xml_escape_less_than():
    assert _xml_escape_via_bash("/a<b") == "/a&lt;b"
