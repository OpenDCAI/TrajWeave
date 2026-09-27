from pathlib import Path

import pytest


def _write_tiny_tokenizer(path: Path, *, pad_token: str = "<pad>") -> None:
    pytest.importorskip("tokenizers")
    pytest.importorskip("transformers")
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {"<unk>": 0, "<eos>": 1, pad_token: 2, "hello": 3, "world": 4}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token=pad_token,
    )
    fast.save_pretrained(path)


def test_tokenizer_compatibility_accepts_identical_tokenizers(tmp_path: Path):
    from trajweave.backends.verl.tokenizer_compat import assert_compatible_tokenizers

    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_tiny_tokenizer(left)
    _write_tiny_tokenizer(right)

    fingerprints = assert_compatible_tokenizers([str(left), str(right)])

    assert len({item.digest for item in fingerprints.values()}) == 1


def test_tokenizer_compatibility_rejects_different_special_ids(tmp_path: Path):
    from trajweave.backends.verl.tokenizer_compat import assert_compatible_tokenizers

    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_tiny_tokenizer(left)
    _write_tiny_tokenizer(right, pad_token="<extra-pad>")

    with pytest.raises(ValueError, match="not compatible"):
        assert_compatible_tokenizers([str(left), str(right)])
