from __future__ import annotations

"""Repair common LLM JSON defects before quality evaluation.

Claude (no json_object mode) often emits ASCII double-quotes inside string values,
e.g. 这一"颠覆者困境"动态, which breaks json.loads.
"""


def repair_unescaped_quotes_in_json_strings(raw: str) -> str:
    """Escape interior \" that appear inside JSON string values."""
    out: list[str] = []
    i = 0
    in_string = False
    n = len(raw)
    while i < n:
        ch = raw[i]
        if not in_string:
            out.append(ch)
            if ch == '"':
                in_string = True
            i += 1
            continue
        if ch == "\\":
            out.append(ch)
            if i + 1 < n:
                out.append(raw[i + 1])
                i += 2
            else:
                i += 1
            continue
        if ch == '"':
            j = i + 1
            while j < n and raw[j] in " \t\n\r":
                j += 1
            if j >= n or raw[j] in ",:}]":
                out.append(ch)
                in_string = False
                i += 1
            else:
                out.append('\\"')
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)
