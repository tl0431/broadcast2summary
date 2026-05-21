"""Patch translation into existing English episode markdown files."""
from __future__ import annotations
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

MD_FILES = [
    '/Users/TL_1/Knowledge/broadcast/archive/Lex Fridman Podcast/2026-03-23-#494 – Jensen Huang_ NVIDIA – The $4 Trillion Company & the AI Revolution.md',
    '/Users/TL_1/Knowledge/broadcast/archive/All-In Podcast/2026-05-15-Trump-Xi Summit, Benioff_ _Not My First SaaSpocalypse,_ OpenAI vs Apple, Multi-S.md',
    '/Users/TL_1/Knowledge/broadcast/archive/The a16z Show/2026-05-20-Marc Andreessen on AI, California, and the Future of America _ Joe Rogan.md',
]

BATCH = 30  # segments per DeepSeek call


def _load_deepseek_key() -> str:
    for f in [os.path.expanduser('~/.bashrc_claude'), os.path.expanduser('~/.bashrc')]:
        try:
            for line in open(f):
                m = re.match(r'export DEEPSEEK_API_KEY=["\']?([^"\'\\n]+)', line.strip())
                if m:
                    return m.group(1)
        except OSError:
            pass
    return os.environ.get('DEEPSEEK_API_KEY', '')


_NUMBERED_RE = re.compile(r'^(\d+)[.、]\s*(.+)', re.MULTILINE)


def _parse_numbered(raw: str, expected: int) -> list[str]:
    result: dict[int, str] = {}
    for m in _NUMBERED_RE.finditer(raw):
        idx = int(m.group(1))
        if 1 <= idx <= expected:
            result[idx] = m.group(2).strip()
    return [result.get(i + 1, "") for i in range(expected)]


def _translate_batch(client, texts: list[str]) -> list[str]:
    """Translate a batch using numbered plain-text format (immune to JSON encoding issues)."""
    if not texts:
        return []
    numbered = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(texts))
    prompt = (
        f"将以下 {len(texts)} 段英文播客逐段翻译成中文。\n"
        "按序输出，每段一行，格式为「序号. 译文」，不要其他内容：\n\n"
        + numbered
    )
    raw = client.complete(prompt, temperature=0.1)
    return _parse_numbered(raw, len(texts))


def patch_file(path: str, client) -> None:
    text = open(path, encoding='utf-8').read()
    lines = text.splitlines()

    ts_start = None
    for i, line in enumerate(lines):
        if line.strip().startswith('## 完整转写'):
            ts_start = i + 1
            break
    if ts_start is None:
        print(f"  No transcript section: {path}")
        return

    # Collect lines that need translation (no [译] already following them)
    need_translation: list[tuple[int, str]] = []
    i = ts_start
    while i < len(lines):
        m = re.match(r'^\[(\d{2}:\d{2}:\d{2})\]\s+\[([^\]]+)\]\s+(.+)$', lines[i])
        if m:
            text_content = m.group(3).strip()
            next_i = i + 1
            while next_i < len(lines) and lines[next_i].strip() == '':
                next_i += 1
            if next_i >= len(lines) or not lines[next_i].startswith('[译]'):
                need_translation.append((i, text_content))
        i += 1

    if not need_translation:
        print(f"  Already fully translated: {os.path.basename(path)}")
        return

    total = len(need_translation)
    print(f"  {total} segments to translate in {os.path.basename(path)}")

    all_translations: list[str] = []
    num_batches = (total + BATCH - 1) // BATCH
    for batch_idx in range(num_batches):
        start = batch_idx * BATCH
        batch_texts = [t for _, t in need_translation[start:start + BATCH]]
        print(f"    batch {batch_idx + 1}/{num_batches} ({len(batch_texts)} segs)...")
        try:
            all_translations.extend(_translate_batch(client, batch_texts))
        except Exception as e:
            print(f"    batch {batch_idx + 1} failed: {e} — filling with empty")
            all_translations.extend([""] * len(batch_texts))

    # Insert [译] lines backwards to preserve line indices
    for (line_idx, _), translation in zip(reversed(need_translation), reversed(all_translations)):
        if translation:
            lines.insert(line_idx + 1, f"[译] {translation}")

    open(path, 'w', encoding='utf-8').write('\n'.join(lines) + '\n')
    print(f"  Patched: {os.path.basename(path)}")


def main() -> None:
    from broadcast2summary.summarize import DeepSeekClient
    client = DeepSeekClient(api_key=_load_deepseek_key())
    for md in MD_FILES:
        print(f"\n→ {os.path.basename(md)}")
        patch_file(md, client)
    print("\nDone.")


if __name__ == '__main__':
    main()
