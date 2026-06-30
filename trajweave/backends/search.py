from __future__ import annotations

import re
from dataclasses import dataclass

from trajweave.backends.policy import PolicyRequest, PolicyResponse, StableByteTokenizer


_ANSWER_HINT_RE = re.compile(r"answer hint\s*:\s*(.+)", re.IGNORECASE)


@dataclass
class RuleBasedSearchPolicyBackend:
    tokenizer: StableByteTokenizer = StableByteTokenizer()

    def generate(self, request: PolicyRequest) -> PolicyResponse:
        role = request.agent.role.lower()
        if role == "verifier":
            text = self._verify(request)
        elif role == "searcher":
            query = request.metadata.get("search_query") or request.observation
            text = f"SEARCH: {query}"
        elif role == "answer":
            answer = self._extract_answer_hint(request.team_context) or request.metadata.get("answer") or "unknown"
            text = f"Final answer: {answer}"
        else:
            text = "PASS"
        token_ids = self.tokenizer.encode(text)
        return PolicyResponse(text=text, token_ids=token_ids, logprobs=[0.0] * len(token_ids))

    def _verify(self, request: PolicyRequest) -> str:
        if "Evidence:" in request.team_context and "Answer hint:" in request.team_context:
            return "APPROVED: sufficient evidence has been retrieved."
        return "SEARCH: evidence is missing; call the searcher."

    def _extract_answer_hint(self, text: str) -> str | None:
        matches = _ANSWER_HINT_RE.findall(text)
        if not matches:
            return None
        return matches[-1].strip().splitlines()[0]
