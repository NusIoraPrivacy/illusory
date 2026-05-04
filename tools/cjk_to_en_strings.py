#!/usr/bin/env python3
"""Translate Chinese in Python string literals / docstrings / f-string text (libcst + MyMemory API)."""
from __future__ import annotations

import ast
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import libcst as cst

REPO = Path(__file__).resolve().parents[1]
SKIP_DIRS = frozenset({".git", "__pycache__", ".venv", "venv", "node_modules"})
_MAX_Q = 480
_CJK = re.compile(r"[\u4e00-\u9fff]")
_two = frozenset(
    {"fr", "Fr", "fR", "FR", "rf", "Rf", "rF", "RF", "br", "Br", "bR", "BR", "rb", "Rb", "rB", "RB"}
)
_one = frozenset("rRuUbBfFUu")

_cache: dict[str, str] = {}


def translate_mymemory_fragment(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    if text in _cache:
        return _cache[text]
    q = urllib.parse.quote(text[:2000])
    url = f"https://api.mymemory.translated.net/get?q={q}&langpair=zh-CN|en"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "illusory-cjk-translate/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode())
            out = (data.get("responseData") or {}).get("translatedText") or text
            _cache[text] = out
            time.sleep(0.12)
            return out
        except Exception as e:
            print(f"[warn] translate attempt {attempt}: {e!r}", file=sys.stderr)
            time.sleep(1.5 * (attempt + 1))
    _cache[text] = text
    return text


def translate_text(s: str) -> str:
    if not s or not _CJK.search(s):
        return s
    if s in _cache:
        return _cache[s]
    # Chunk long strings for URL safety and API limits
    if len(s) <= _MAX_Q and s.count("\n") < 40:
        out = translate_mymemory_fragment(s)
        _cache[s] = out
        return out
    parts: list[str] = []
    buf = ""
    for line in s.split("\n"):
        if len(buf) + len(line) + 1 > _MAX_Q and buf:
            parts.append(buf)
            buf = line
        else:
            buf = (buf + "\n" + line) if buf else line
    if buf:
        parts.append(buf)
    merged = []
    for p in parts:
        merged.append(translate_mymemory_fragment(p) if _CJK.search(p) else p)
    out = "\n".join(merged)
    _cache[s] = out
    return out


def split_string_literal(raw: str) -> tuple[str, bool, bool]:
    if len(raw) >= 2 and raw[:2] in _two:
        pfx, rest = raw[:2], raw[2:]
    elif raw and raw[0] in _one:
        pfx, rest = raw[:1], raw[1:]
    else:
        pfx, rest = "", raw

    if rest.startswith('"""'):
        return pfx, True, True
    if rest.startswith("'''"):
        return pfx, True, False
    if rest.startswith('"'):
        return pfx, False, True
    if rest.startswith("'"):
        return pfx, False, False
    raise ValueError(f"Bad string literal: {raw[:50]!r}")


def encode_simple_inner(new_inner: str, *, is_triple: bool, prefer_double: bool) -> str:
    if is_triple:
        q = '"""' if prefer_double else "'''"
        esc = new_inner.replace("\\", "\\\\")
        if q == '"""':
            esc = esc.replace('"""', '\\"\\"\\"')
        else:
            esc = esc.replace("'''", "\\'\\'\\'")
        return f"{q}{esc}{q}"
    return json.dumps(new_inner, ensure_ascii=False)


def rebuild_simple_string(node: cst.SimpleString, new_inner: str) -> cst.SimpleString:
    raw = node.value
    pfx, is_triple, prefer_double = split_string_literal(raw)
    encoded = encode_simple_inner(new_inner, is_triple=is_triple, prefer_double=prefer_double)
    return node.with_changes(value=pfx + encoded)


class TranslateStrings(cst.CSTTransformer):
    def leave_SimpleString(self, original_node: cst.SimpleString, updated_node: cst.SimpleString) -> cst.SimpleString:
        try:
            inner = updated_node.evaluated_value
        except Exception:
            try:
                inner = ast.literal_eval(updated_node.value)
            except Exception:
                return updated_node
        if not isinstance(inner, str) or not inner or not _CJK.search(inner):
            return updated_node
        new_inner = translate_text(inner)
        if new_inner == inner:
            return updated_node
        try:
            return rebuild_simple_string(updated_node, new_inner)
        except Exception as e:
            print(f"[warn] rebuild SimpleString failed: {e}", file=sys.stderr)
            return updated_node

    def leave_FormattedStringText(
        self, original_node: cst.FormattedStringText, updated_node: cst.FormattedStringText
    ) -> cst.FormattedStringText:
        t = updated_node.value
        if not t or not _CJK.search(t):
            return updated_node
        new_t = translate_text(t)
        if new_t == t:
            return updated_node
        return updated_node.with_changes(value=new_t)


def process_file(path: Path) -> bool:
    src = path.read_text(encoding="utf-8")
    try:
        mod = cst.parse_module(src)
    except cst.ParserSyntaxError as e:
        print(f"[skip parse] {path}: {e}", file=sys.stderr)
        return False
    new_mod = mod.visit(TranslateStrings())
    new_src = new_mod.code
    if new_src != src:
        path.write_text(new_src, encoding="utf-8")
        return True
    return False


def main() -> int:
    changed = 0
    for path in sorted(REPO.rglob("*.py")):
        if any(p in SKIP_DIRS for p in path.parts):
            continue
        if path.resolve() == Path(__file__).resolve():
            continue
        if process_file(path):
            changed += 1
            print(path.relative_to(REPO), flush=True)
    print(f"Modified {changed} files.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
