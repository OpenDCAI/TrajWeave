from __future__ import annotations

import re

from trajweave.metrics.events import MetricEvent


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def parse_verl_console_metrics(stdout: str, *, run_id: str, source: str = "verl") -> list[MetricEvent]:
    events: list[MetricEvent] = []
    for raw_line in stdout.splitlines():
        line = ANSI_RE.sub("", raw_line).strip()
        if "step:" not in line or " - " not in line:
            continue
        line = line[line.index("step:") :]
        fragments = [fragment.strip() for fragment in line.split(" - ")]
        step: int | None = None
        for fragment in fragments:
            if fragment.startswith("step:"):
                try:
                    step = int(float(fragment.split(":", 1)[1]))
                except ValueError:
                    step = None
                break
        for fragment in fragments:
            if ":" not in fragment:
                continue
            name, raw_value = fragment.split(":", 1)
            name = name.strip()
            if name == "step":
                continue
            value = _parse_metric_value(raw_value.strip())
            if value is None:
                continue
            events.append(MetricEvent(run_id=run_id, name=name, value=value, step=step, source=source))
    return events


def _parse_metric_value(value: str) -> float | int | None:
    value = value.strip()
    if not value:
        return None
    try:
        number = float(value)
    except ValueError:
        return None
    if number.is_integer():
        return int(number)
    return number
