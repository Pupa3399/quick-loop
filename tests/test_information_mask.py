from __future__ import annotations

import re

from ouro_search.training.information_mask import (
    build_information_token_mask,
    information_spans,
)


class WhitespaceTokenizer:
    def __call__(
        self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool
    ) -> dict[str, object]:
        del add_special_tokens, return_offsets_mapping
        matches = list(re.finditer(r"\S+", text))
        return {
            "input_ids": list(range(len(matches))),
            "offset_mapping": [(match.start(), match.end()) for match in matches],
        }


def test_information_region_has_zero_policy_mask() -> None:
    text = (
        "<think> mine </think> <information> external evidence </information> "
        "<answer> mine </answer>"
    )
    mask = build_information_token_mask(WhitespaceTokenizer(), text)
    tokens = text.split()
    assert len(mask) == len(tokens)
    assert [token for token, keep in zip(tokens, mask, strict=True) if not keep] == [
        "<information>",
        "external",
        "evidence",
        "</information>",
    ]


def test_unclosed_information_masks_to_end() -> None:
    text = "generated <information> retrieved forever"
    assert information_spans(text) == [(10, len(text))]
    assert build_information_token_mask(WhitespaceTokenizer(), text) == [1, 0, 0, 0]
