"""
Core utility functions for FlowAgent framework
"""
import json
import re
from json import JSONDecodeError, JSONDecoder
from typing import Any, Dict, Union, List
from pathlib import Path
from flowagent.logger import get_logger

log = get_logger(__name__)


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).resolve().parent.parent


def robust_parse_json(
    text: str,
    *,
    merge_dicts: bool = False,
    strip_double_braces: bool = False
) -> Union[Dict[str, Any], List[Any]]:
    """
    Extract valid JSON from LLM output, logs, jsonl, or Markdown fragments.

    Parameters
    ----------
    text : str
        Input raw text
    merge_dicts : bool, default False
        If multiple dicts are extracted, merge them with dict.update
    strip_double_braces : bool, default False
        Replace '{{' / '}}' with '{' / '}' (for template languages)

    Returns
    -------
    Dict / List / List[Dict | List]
    """
    s = text.strip()

    # ---------- Preprocessing: Remove outer wrappers ----------
    s = _remove_markdown_fence(s)          # ```json ... ```
    s = _remove_outer_triple_quotes(s)     # ''' ... ''' / """ ... """
    s = _remove_leading_json_word(s)       # Leading json/JSON marker

    if strip_double_braces:
        s = s.replace("{{", "{").replace("}}", "}")

    # ---------- Clean comments & trailing commas ----------
    s = _strip_json_comments(s)

    # ---------- Clean illegal control characters ----------
    # Remove all ASCII control characters not allowed in JSON spec
    s = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', s)

    # ---------- Escape unescaped backslashes (fix LaTeX formulas, etc.) ----------
    # Protect already escaped sequences
    s = s.replace('\\\\', '\x00DOUBLE_BACKSLASH\x00')
    s = s.replace('\\n', '\x00NEWLINE\x00')
    s = s.replace('\\r', '\x00RETURN\x00')
    s = s.replace('\\t', '\x00TAB\x00')
    s = s.replace('\\"', '\x00QUOTE\x00')
    s = s.replace('\\/', '\x00SLASH\x00')
    s = s.replace('\\b', '\x00BACKSPACE\x00')
    s = s.replace('\\f', '\x00FORMFEED\x00')

    # Escape all remaining single backslashes
    s = s.replace('\\', '\\\\')

    # Restore protected sequences
    s = s.replace('\x00DOUBLE_BACKSLASH\x00', '\\\\')
    s = s.replace('\x00NEWLINE\x00', '\\n')
    s = s.replace('\x00RETURN\x00', '\\r')
    s = s.replace('\x00TAB\x00', '\\t')
    s = s.replace('\x00QUOTE\x00', '\\"')
    s = s.replace('\x00SLASH\x00', '\\/')
    s = s.replace('\x00BACKSPACE\x00', '\\b')
    s = s.replace('\x00FORMFEED\x00', '\\f')

    log.debug(f'Cleaned content: {s}')

    # ---------- Step-1: Parse entire string ----------
    try:
        result = json.loads(s)
        log.info(f"Parsed successfully, type: {type(result)}")
        return result
    except JSONDecodeError as e:
        log.warning(f"Full parse failed: {e}")

    # ---------- Step-2: Try JSON Lines ----------
    objs = _parse_json_lines(s)
    if objs is not None:
        return _maybe_merge(objs, merge_dicts)

    # ---------- Step-3: Stream extract multiple objects ----------
    objs = _extract_json_objects(s)
    log.warning(f"Extracted {len(objs)} objects")
    if not objs:
        raise ValueError("Unable to locate any valid JSON fragment.")

    return _maybe_merge(objs, merge_dicts)


# ======================================================================
#                            Helper Functions
# ======================================================================

# Regex patterns
_fence_pat = re.compile(r'```[\w-]*\s*([\s\S]*?)```', re.I)
_outer_fence_pat = re.compile(r'^\s*```[\w-]*\s*([\s\S]*?)```\s*$', re.I)


def _remove_markdown_fence(src: str) -> str:
    """Extract text from outer ``` … ``` wrapper; return as-is if not wrapped"""
    match = _outer_fence_pat.match(src)
    if match:
        return match.group(1).strip()
    return src


def _remove_outer_triple_quotes(src: str) -> str:
    """Remove outer triple quotes if present"""
    if (src.startswith("'''") and src.endswith("'''")) or (
        src.startswith('"""') and src.endswith('"""')
    ):
        return src[3:-3].strip()
    return src


def _remove_leading_json_word(src: str) -> str:
    """Remove leading 'json' or 'JSON' marker"""
    return src[4:].lstrip() if src.lower().startswith("json") else src


def _strip_json_comments(src: str) -> str:
    """Remove comments and trailing commas from JSON-like text"""
    # /* ... */ block comments
    src = re.sub(r'/\*[\s\S]*?\*/', '', src)
    # // ... line comments (excluding :// in URLs and // in strings)
    src = re.sub(r'(?<![:\"\'])//.*', '', src)
    # Trailing commas ,}
    src = re.sub(r',\s*([}\]])', r'\1', src)
    return src.strip()


def _parse_json_lines(src: str) -> Union[List[Any], None]:
    """Parse JSON Lines format (one JSON object per line)"""
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]
    if len(lines) <= 1:
        return None

    objs: List[Any] = []
    for ln in lines:
        try:
            objs.append(json.loads(ln))
        except JSONDecodeError:
            return None
    return objs


def _extract_json_objects(src: str) -> List[Any]:
    """Extract multiple JSON objects from text stream"""
    dec = JSONDecoder()
    idx, n = 0, len(src)
    objs: List[Any] = []

    while idx < n:
        m = re.search(r'[{\[]', src[idx:])
        if not m:
            break
        idx += m.start()
        try:
            obj, end = dec.raw_decode(src, idx)
            # Strictness check
            tail = src[end:].lstrip()
            if tail and tail[0] not in ',]}>\n\r':
                idx += 1
                continue
            objs.append(obj)
            idx = end
        except JSONDecodeError:
            idx += 1
    return objs


def _maybe_merge(objs: List[Any], merge_dicts: bool) -> Union[Any, List[Any]]:
    """Merge multiple dicts if requested, otherwise return list"""
    if len(objs) == 1:
        return objs[0]
    if merge_dicts and all(isinstance(o, dict) for o in objs):
        merged: Dict[str, Any] = {}
        for o in objs:
            merged.update(o)
        return merged
    return objs
