from __future__ import annotations

from html.parser import HTMLParser


class _FirstScoreTagParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.depth = 0
        self.finished = False
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        del attrs
        if self.finished:
            return
        if tag.lower() == "score":
            self.depth += 1
        elif self.depth:
            self.depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self.finished or not self.depth:
            return
        self.depth -= 1
        if tag.lower() == "score" and self.depth == 0:
            self.finished = True

    def handle_data(self, data: str) -> None:
        if self.depth and not self.finished:
            self.parts.append(data)


def parse_comas_score(content: str) -> int | None:
    """对齐 CoMAS 原仓库：只接受第一个 ``<score>`` 中的 1、2、3。"""

    parser = _FirstScoreTagParser()
    try:
        parser.feed(str(content))
        parser.close()
        if not parser.finished:
            return None
        score = int("".join(parser.parts).strip())
    except (TypeError, ValueError):
        return None
    return score if score in {1, 2, 3} else None
