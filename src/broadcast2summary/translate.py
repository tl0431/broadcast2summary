from __future__ import annotations
import json
from .transcribe import Segment


def translate_segments(segments: list[Segment], deepseek_client) -> list[Segment]:
    """Batch-translate English segments to Chinese via DeepSeek.

    Sends all segment texts in ONE API call as a JSON array.
    Returns new Segment list with .translation field populated.
    """
    if not segments:
        return segments

    texts = [s.text for s in segments]
    prompt = (
        "将以下英文播客转写片段翻译成中文。\n"
        "严格按 JSON 数组返回,顺序与输入一致,每条只有 \"t\" 字段:\n\n"
        f"{json.dumps(texts, ensure_ascii=False)}\n\n"
        "返回格式: [{\"t\": \"译文1\"}, {\"t\": \"译文2\"}, ...]"
    )
    raw = deepseek_client.complete(prompt, temperature=0.1)
    translations = json.loads(raw)
    return [
        Segment(start=s.start, end=s.end, text=s.text,
                translation=t.get("t", ""))
        for s, t in zip(segments, translations)
    ]
