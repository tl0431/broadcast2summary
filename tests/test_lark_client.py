from broadcast2summary.lark_client import LarkClient, LarkCliError
import pytest


def test_run_invokes_subprocess(monkeypatch):
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        class R:
            returncode = 0
            stdout = '{"ok": true}'
            stderr = ""
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)
    c = LarkClient()
    out = c.run(["im", "send", "--to", "ou_1", "--text", "hi"])
    assert out == '{"ok": true}'
    assert calls[0][0] == "lark-cli"
    assert "--to" in calls[0]


def test_run_raises_on_nonzero(monkeypatch):
    def fake_run(cmd, **kwargs):
        class R:
            returncode = 1
            stdout = ""
            stderr = "boom"
        return R()
    monkeypatch.setattr("subprocess.run", fake_run)
    c = LarkClient()
    with pytest.raises(LarkCliError, match="boom"):
        c.run(["im", "send"])
