import subprocess
import sys


def test_cli_help_lists_subcommands():
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "--help"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    out = r.stdout
    for sub in ["run", "test", "fetch-one", "backfill", "retry-failed",
                "list-failed", "feeds"]:
        assert sub in out


def test_cli_test_subcommand_runs_smoke_path(tmp_path, monkeypatch):
    # `test` subcommand must run end-to-end with fixtures and return 0.
    env = {"DEEPSEEK_API_KEY": "x", "ANTHROPIC_API_KEY": "x",
           "BROADCAST2SUMMARY_HOME": str(tmp_path)}
    r = subprocess.run(
        [sys.executable, "-m", "broadcast2summary", "test"],
        capture_output=True, text=True, env={**env},
    )
    assert r.returncode == 0, r.stderr
    assert "all components OK" in r.stdout
