from __future__ import annotations

import json
from typing import Any

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import required_ground_truth, to_python
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput


class GiGPOEmitterMixin:
    def _build_gigpo_solver_verifier_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        ground_truth = required_ground_truth(prompt)
        prompt_ids = self._encode_prompt(raw_prompt)
        max_steps = self._gigpo_max_steps()
        is_correct = session_id % 2 == 0
        episode_reward = 1.0 if is_correct else 0.0
        metrics = AgentLoopMetrics(generate_sequences=0.0, tool_calls=0.0, compute_score=0.0, num_preempted=-1)
        outputs: list[AgentLoopOutput] = []
        running_prompt_ids = list(prompt_ids)
        feedback = ""
        for step_id in range(max_steps):
            answer = ground_truth if is_correct and step_id == max_steps - 1 else "__wrong__"
            response_ids = self._encode_text(f" Final answer: {answer}")
            anchor = json.dumps(
                {"agent_id": "solver", "task": raw_prompt, "verifier_feedback": feedback},
                ensure_ascii=False,
                sort_keys=True,
                default=str,
            )
            feedback = "APPROVED" if answer == ground_truth else "REVISE"
            outputs.append(
                AgentLoopOutput(
                    prompt_ids=list(running_prompt_ids),
                    response_ids=response_ids,
                    response_mask=[1] * len(response_ids),
                    reward_score=episode_reward,
                    num_turns=step_id + 1,
                    metrics=metrics,
                    extra_fields={
                        "turn_scores": [],
                        "tool_rewards": [],
                        "trajweave_agent_name": "solver",
                        "trajweave_role": "solver",
                        "agent_id": "solver",
                        "policy_group": "shared",
                        "anchor_obs": anchor,
                        "next_obs": feedback,
                        "step_reward": episode_reward if step_id == max_steps - 1 else 0.0,
                        "active_mask": 1.0,
                        "step_id": step_id,
                    },
                )
            )
            running_prompt_ids.extend(response_ids)
        return outputs

    def _gigpo_max_steps(self) -> int:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        gigpo_cfg = config_get(orchestra_cfg, "gigpo", default={}) or {}
        return int(config_get(gigpo_cfg, "max_steps", default=2))
