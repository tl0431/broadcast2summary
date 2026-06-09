from __future__ import annotations
import subprocess


class LarkCliError(RuntimeError):
    pass


class LarkClient:
    """Thin subprocess wrapper over `lark-cli`. Keeps auth out of this codebase."""

    def __init__(self, executable: str = "lark-cli"):
        self.executable = executable

    def run(
        self,
        args: list[str],
        *,
        input_text: str | None = None,
        timeout: int = 120,
        cwd: str | None = None,
    ) -> str:
        cmd = [self.executable, *args]
        result = subprocess.run(
            cmd,
            input=input_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        if result.returncode != 0:
            raise LarkCliError(
                f"lark-cli failed (exit {result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
            )
        return result.stdout
