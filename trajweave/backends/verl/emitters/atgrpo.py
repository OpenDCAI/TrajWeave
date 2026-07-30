from __future__ import annotations

from typing import Any

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput


class ATGRPOEmitterMixin:
    """Deterministic selected-spine emitter used by the synthetic TQ backend."""

    def _build_atgrpo_solver_verifier_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        reward_model = to_python(prompt.get("reward_model", {})) or {}
        ground_truth = str(reward_model.get("ground_truth", "2"))
        root_id = f"{to_python(prompt.get('uid', 'task'))}:tree:{session_id}"
        branch_factor = max(1, int(to_python(prompt.get("__atgrpo_branch_factor__", 1))))
        mixed_reward = self._atgrpo_mixed_reward_settings()
        global_reward = 1.0
        metrics = AgentLoopMetrics(generate_sequences=0.0, tool_calls=0.0, compute_score=0.0, num_preempted=-1)
        running_prompt_ids = self._encode_prompt(raw_prompt)
        observation_text = self._atgrpo_observation_text(raw_prompt)
        parent_node_id = ""
        context: list[str] = []
        outputs: list[AgentLoopOutput] = []
        turn_id = 0

        for solver_round in range(self._atgrpo_max_turns()):
            solver_group_id = f"{root_id}:obs:{parent_node_id or 'root'}:{turn_id}:solver"
            solver_prompt_text = self._atgrpo_prompt_text("solver", observation_text, context)
            selected_solver_text = f"Final answer: {ground_truth}"
            selected_response_ids: list[int] = []
            selected_node_id = ""
            for branch_index in range(branch_factor):
                is_selected = branch_index == 0
                node_id = f"{root_id}:node:{turn_id}:{branch_index}"
                answer = ground_truth if is_selected else "__wrong__"
                text = f" Final answer: {answer}"
                local_score = 1.0 if is_selected else 0.0
                reward_score = (
                    mixed_reward["alpha"] * global_reward + local_score if mixed_reward["enabled"] else global_reward
                )
                response_ids = self._encode_text(text)
                outputs.append(
                    AgentLoopOutput(
                        prompt_ids=list(running_prompt_ids),
                        response_ids=response_ids,
                        response_mask=[1] * len(response_ids),
                        reward_score=reward_score,
                        num_turns=turn_id + 1,
                        metrics=metrics,
                        extra_fields={
                            "turn_scores": [],
                            "tool_rewards": [],
                            "trajweave_agent_name": "solver",
                            "trajweave_role": "solver",
                            "agent_id": "solver",
                            "policy_group": "shared",
                            "traj_uid": root_id,
                            "turn_id": turn_id,
                            "root_id": root_id,
                            "node_id": node_id,
                            "parent_node_id": parent_node_id,
                            "observation_group_id": solver_group_id,
                            "branch_index": branch_index,
                            "selected_for_expansion": is_selected,
                            "local_score": local_score,
                            "local_correct": is_selected,
                            "prompt_text": solver_prompt_text,
                            "response_text": text.strip(),
                            "observation_text": observation_text,
                            "final_answer": selected_solver_text,
                            "workflow_success": True,
                        },
                    )
                )
                if is_selected:
                    selected_response_ids = response_ids
                    selected_node_id = node_id
            running_prompt_ids.extend(selected_response_ids)
            context.append(f"solver: {selected_solver_text}")
            parent_node_id = selected_node_id
            turn_id += 1

            if solver_round == self._atgrpo_max_turns() - 1:
                break

            verifier_group_id = f"{root_id}:obs:{parent_node_id}:{turn_id}:verifier"
            verifier_prompt_text = self._atgrpo_prompt_text("verifier", observation_text, context)
            selected_verifier_approved = False
            selected_response_ids = []
            selected_node_id = ""
            for branch_index in range(branch_factor):
                is_selected = branch_index == 0
                node_id = f"{root_id}:node:{turn_id}:{branch_index}"
                text = " APPROVED" if is_selected else " REVISE"
                local_score = 1.0 if is_selected else -1.0
                role_local_reward = local_score * mixed_reward["verifier_local_reward"]
                reward_score = (
                    mixed_reward["alpha"] * global_reward + role_local_reward
                    if mixed_reward["enabled"]
                    else global_reward
                )
                response_ids = self._encode_text(text)
                outputs.append(
                    AgentLoopOutput(
                        prompt_ids=list(running_prompt_ids),
                        response_ids=response_ids,
                        response_mask=[1] * len(response_ids),
                        reward_score=reward_score,
                        num_turns=turn_id + 1,
                        metrics=metrics,
                        extra_fields={
                            "turn_scores": [],
                            "tool_rewards": [],
                            "trajweave_agent_name": "verifier",
                            "trajweave_role": "verifier",
                            "agent_id": "verifier",
                            "policy_group": "shared",
                            "traj_uid": root_id,
                            "turn_id": turn_id,
                            "root_id": root_id,
                            "node_id": node_id,
                            "parent_node_id": parent_node_id,
                            "observation_group_id": verifier_group_id,
                            "branch_index": branch_index,
                            "selected_for_expansion": is_selected,
                            "local_score": local_score,
                            "model_approved": is_selected,
                            "approved": is_selected,
                            "local_solver_correct": True,
                            "local_decision_correct": is_selected,
                            "prompt_text": verifier_prompt_text,
                            "response_text": text.strip(),
                            "observation_text": observation_text,
                            "final_answer": selected_solver_text,
                            "workflow_success": True,
                        },
                    )
                )
                if is_selected:
                    selected_response_ids = response_ids
                    selected_node_id = node_id
                    selected_verifier_approved = True
            running_prompt_ids.extend(selected_response_ids)
            context.append("verifier: APPROVED")
            parent_node_id = selected_node_id
            turn_id += 1
            if selected_verifier_approved:
                break
        return outputs

    def _build_hf_atgrpo_solver_verifier_outputs(
        self, prompt: dict[str, Any], *, session_id: int = 0
    ) -> list[AgentLoopOutput]:
        from trajweave.backends.verl.workflow_runtime import build_hf_workflow_outputs

        return build_hf_workflow_outputs(
            self,
            recipe="atgrpo_solver_verifier_math",
            prompt=prompt,
            session_id=session_id,
        )

    @staticmethod
    def _atgrpo_observation_text(raw_prompt: Any) -> str:
        if isinstance(raw_prompt, list):
            contents = [str(item.get("content", "")) for item in raw_prompt if isinstance(item, dict)]
            text = "\n".join(content for content in contents if content)
            if text:
                return text
        text = str(raw_prompt).strip()
        return text or "AT-GRPO math task"

    @staticmethod
    def _atgrpo_prompt_text(agent_name: str, observation: str, context: list[str]) -> str:
        rendered_context = "\n".join(context)
        if agent_name == "solver":
            instruction = "Solve the task and return 'Final answer: <number>'."
        else:
            instruction = "Verify the latest solver answer. Return APPROVED or REVISE."
        return f"Task:\n{observation}\n\nTeam context:\n{rendered_context}\n\n{instruction}"

    def _atgrpo_max_turns(self) -> int:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        atgrpo_cfg = config_get(orchestra_cfg, "atgrpo", default={}) or {}
        return int(config_get(atgrpo_cfg, "max_turns", default=3))

    def _atgrpo_mixed_reward_settings(self) -> dict[str, float | bool]:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        atgrpo_cfg = config_get(orchestra_cfg, "atgrpo", default={}) or {}
        mixed_reward_cfg = config_get(atgrpo_cfg, "mixed_reward", default={}) or {}
        return {
            "enabled": bool(config_get(mixed_reward_cfg, "enabled", default=False)),
            "alpha": float(config_get(mixed_reward_cfg, "alpha", default=1.0)),
            "verifier_local_reward": float(config_get(mixed_reward_cfg, "verifier_local_reward", default=1.0)),
        }
