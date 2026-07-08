from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class TokenizerFingerprint:
    path: str
    digest: str
    vocab_size: int
    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None
    unk_token_id: int | None


def tokenizer_fingerprints_by_path(
    tokenizer_paths: Iterable[str],
    *,
    trust_remote_code: bool = True,
) -> dict[str, TokenizerFingerprint]:
    """按 tokenizer 路径返回稳定 fingerprint。

    MAPoRL multi-actor 训练允许不同模型路径，但共享 token-id batch
    前必须避免不安全的 token id 混用。只有 vocab 和关键 special token
    id 生成同一个 digest 时，两个 tokenizer 才会被视为兼容。
    """

    fingerprints: dict[str, TokenizerFingerprint] = {}
    for tokenizer_path in dict.fromkeys(str(path) for path in tokenizer_paths):
        fingerprints[tokenizer_path] = _tokenizer_fingerprint(
            tokenizer_path,
            trust_remote_code=trust_remote_code,
        )
    return fingerprints


def assert_compatible_tokenizers(
    tokenizer_paths: Iterable[str],
    *,
    trust_remote_code: bool = True,
) -> dict[str, TokenizerFingerprint]:
    fingerprints = tokenizer_fingerprints_by_path(
        tokenizer_paths,
        trust_remote_code=trust_remote_code,
    )
    digests = {fingerprint.digest for fingerprint in fingerprints.values()}
    if len(digests) > 1:
        summary = {
            path: {
                "digest": fingerprint.digest[:12],
                "vocab_size": fingerprint.vocab_size,
                "bos_token_id": fingerprint.bos_token_id,
                "eos_token_id": fingerprint.eos_token_id,
                "pad_token_id": fingerprint.pad_token_id,
                "unk_token_id": fingerprint.unk_token_id,
            }
            for path, fingerprint in fingerprints.items()
        }
        raise ValueError(f"Tokenizer paths are not compatible for shared token-id training: {summary}")
    return fingerprints


@lru_cache(maxsize=32)
def _tokenizer_fingerprint(tokenizer_path: str, *, trust_remote_code: bool) -> TokenizerFingerprint:
    from transformers import AutoTokenizer

    source = Path(tokenizer_path).expanduser()
    if source.exists():
        tokenizer_source = str(source)
    else:
        tokenizer_source = tokenizer_path

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=trust_remote_code,
    )
    vocab = tokenizer.get_vocab()
    special_ids = {
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "unk_token_id": tokenizer.unk_token_id,
    }
    payload = {
        "vocab": sorted((str(token), int(token_id)) for token, token_id in vocab.items()),
        "special_ids": special_ids,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    return TokenizerFingerprint(
        path=tokenizer_path,
        digest=digest,
        vocab_size=len(vocab),
        bos_token_id=special_ids["bos_token_id"],
        eos_token_id=special_ids["eos_token_id"],
        pad_token_id=special_ids["pad_token_id"],
        unk_token_id=special_ids["unk_token_id"],
    )
