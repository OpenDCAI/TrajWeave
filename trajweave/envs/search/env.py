from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchDocument:
    title: str
    text: str


@dataclass(frozen=True)
class SearchTask:
    task_id: str
    question: str
    answer: str
    documents: tuple[SearchDocument, ...]
    search_query: str | None = None


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"final\s*answer\s*[:=]", " ", text)
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return " ".join(text.split())


def _final_answer_candidate(text: str) -> str:
    matches = re.findall(r"final\s*answer\s*[:=]\s*([^\r\n]*)", text, flags=re.IGNORECASE)
    return matches[-1] if matches else text


class SearchAnswerEnvironment:
    name = "search_answer"

    def initial_observation(self, task: SearchTask) -> str:
        return task.question

    def execute_tool(self, tool_name: str, task: SearchTask, request: str) -> str:
        normalized = normalize_answer(tool_name).replace(" ", "_")
        if normalized not in {"search", "access", "browse", "web_search", "search_and_browse"}:
            raise ValueError(f"Unsupported search environment tool: {tool_name!r}.")
        return self.search(task, request)

    def search(self, task: SearchTask, query: str) -> str:
        query_terms = set(normalize_answer(query).split())
        best_doc = task.documents[0]
        best_score = -1
        for doc in task.documents:
            doc_terms = set(normalize_answer(f"{doc.title} {doc.text}").split())
            score = len(query_terms & doc_terms)
            if score > best_score:
                best_doc = doc
                best_score = score
        return f"Evidence: {best_doc.title}. {best_doc.text}"

    def evaluate(self, task: SearchTask, final_answer: str) -> tuple[float, bool]:
        expected = normalize_answer(task.answer)
        predicted = normalize_answer(_final_answer_candidate(final_answer))
        success = bool(expected) and predicted == expected
        return (1.0 if success else 0.0), success
