"""Patch translation into existing English episode markdown files."""
from __future__ import annotations
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

MD_FILES = [
    '/Users/TL_1/Knowledge/broadcast/archive/Lex Fridman Podcast/2026-03-23-#494 – Jensen Huang_ NVIDIA – The $4 Trillion Company & the AI Revolution.md',
    '/Users/TL_1/Knowledge/broadcast/archive/All-In Podcast/2026-05-15-Trump-Xi Summit, Benioff_ _Not My First SaaSpocalypse,_ OpenAI vs Apple, Multi-S.md',
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


def _translate_batch(client, texts: list[str]) -> list[str]:
    prompt = (
        "将以下英文播客段落逐段翻译成中文。\n"
        "严格按 JSON 数组返回，顺序与输入一致，每条只有 \"t\" 字段:\n\n"
        f"{json.dumps(texts, ensure_ascii=False)}\n\n"
        "返回格式: [{\"t\": \"译文1\"}, {\"t\": \"译文2\"}, ...]"
    )
    raw = client.complete_json(prompt, temperature=0.1)
    try:
        result = json.loads(raw)
        # unwrap if DeepSeek returned {"translations": [...]} style object
        if isinstance(result, dict):
            result = next((v for v in result.values() if isinstance(v, list)), [])
        return [r.get("t", "") if isinstance(r, dict) else "" for r in result]
    except json.JSONDecodeError as e:
        print(f"    JSON parse error: {e} — batch skipped")
        return [""] * len(texts)


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
