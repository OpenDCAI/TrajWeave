from __future__ import annotations


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    del data_source, extra_info, kwargs
    if ground_truth is None:
        return 0.0
    return 1.0 if str(ground_truth).strip() in str(solution_str) else 0.0
