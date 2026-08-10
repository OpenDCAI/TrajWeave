from __future__ import annotations

import hashlib
from typing import Any

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python
from trajweave.envs import CodeExecutionEnvironment
from trajweave.orchestration.tree_search import TreeSearchController
from trajweave.verifiers import VerifierResult
from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput


class MARTIMARS2EmitterMixin:
    def _cleanup_marti_mars2_state(self, uid: str) -> None:
        if hasattr(self, "_trajweave_marti_tree_records"):
            self._trajweave_marti_tree_records.pop(uid, None)
        if hasattr(self, "_trajweave_marti_tree_controllers"):
            self._trajweave_marti_tree_controllers.pop(uid, None)

    def _build_marti_mars2_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        node_id = session_id
        parent_idx = self._marti_select_parent(prompt, node_id=node_id)
        prompt_ids = self._marti_prompt_ids(prompt, node_id=node_id, parent_idx=parent_idx)
        candidate = self._marti_fixture_candidate(node_id) or f" candidate-{node_id}"
        response_ids = self._encode_text(candidate)
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        verification = self._marti_verify(prompt, node_id=node_id, parent_idx=parent_idx, candidate=response_text)
        self._remember_marti_response(
            prompt,
            node_id=node_id,
            response_ids=response_ids,
            parent_idx=parent_idx,
            verification=verification,
        )
        metrics = AgentLoopMetrics(generate_sequences=0.0, tool_calls=0.0, compute_score=1.0, num_preempted=-1)
        return [
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=[1] * len(response_ids),
                response_logprobs=[-0.1 * (node_id + 1)] * len(response_ids),
                reward_score=verification.score,
                num_turns=1,
                metrics=metrics,
                extra_fields=self._marti_tree_fields(
                    prompt,
                    node_id=node_id,
                    parent_idx=parent_idx,
                    verification=verification,
                    source="synthetic_tq",
                ),
            )
        ]

    def _marti_fixture_candidate(self, node_id: int) -> str | None:
        """Return an explicit verifier fixture candidate when configured.

        This path is for multi-actor learning-signal tests only. It preserves
        real verifier execution while making the candidate/token contract
        deterministic; it is never used by the native vLLM fidelity recipe.
        """
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        marti_cfg = config_get(orchestra_cfg, "marti_mars2", default={}) or {}
        candidates = to_python(config_get(marti_cfg, "fixture_candidates", default=None))
        if candidates is None:
            return None
        if isinstance(candidates, dict):
            candidate = candidates.get(str(node_id), candidates.get(node_id))
        else:
            candidate = candidates[node_id] if node_id < len(candidates) else None
        if candidate is None:
            return None
        # JSON/Hydra CLI overrides preserve escaped newlines as literal ``\\n``
        # in some OmegaConf versions. Decode that transport representation so
        # the real verifier receives executable source code.
        candidate_text = str(candidate)
        if "\\n" in candidate_text:
            candidate_text = candidate_text.replace("\\r\\n", "\\n").replace("\\n", "\n")
        return candidate_text

    def _build_hf_marti_mars2_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        node_id = session_id
        parent_idx = self._marti_select_parent(prompt, node_id=node_id)
        prompt_ids = self._marti_prompt_ids(prompt, node_id=node_id, parent_idx=parent_idx)
        _agent_id, policy_group = self._marti_agent_binding(node_id)
        response_ids, response_logprobs = self._generate_local_response_with_logprobs(
            prompt_ids, policy_group=policy_group
        )
        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        verification = self._marti_verify(prompt, node_id=node_id, parent_idx=parent_idx, candidate=response_text)
        self._remember_marti_response(
            prompt,
            node_id=node_id,
            response_ids=response_ids,
            parent_idx=parent_idx,
            verification=verification,
        )
        metrics = AgentLoopMetrics(generate_sequences=1.0, tool_calls=0.0, compute_score=1.0, num_preempted=-1)
        return [
            AgentLoopOutput(
                prompt_ids=prompt_ids,
                response_ids=response_ids,
                response_mask=[1] * len(response_ids),
                response_logprobs=response_logprobs,
                reward_score=verification.score,
                num_turns=1,
                metrics=metrics,
                extra_fields=self._marti_tree_fields(
                    prompt,
                    node_id=node_id,
                    parent_idx=parent_idx,
                    verification=verification,
                    source="hf_local_tq",
                ),
            )
        ]

    def _marti_tree_fields(
        self,
        prompt: dict[str, Any],
        *,
        node_id: int,
        parent_idx: int,
        verification: VerifierResult,
        source: str,
    ) -> dict[str, Any]:
        tree_id = str(to_python(prompt.get("uid", prompt.get("index", "tree"))))
        path = self._marti_path_for_node(prompt, node_id=node_id, parent_idx=parent_idx)
        agent_id, policy_group = self._marti_agent_binding(node_id)
        rollout_global_step = int(to_python(prompt.get("global_steps", 0)) or 0)
        rollout_policy_step = int(to_python(prompt.get("policy_step", rollout_global_step)) or rollout_global_step)
        return {
            "turn_scores": [],
            "tool_rewards": [],
            "trajweave_agent_name": agent_id,
            "trajweave_role": agent_id,
            "agent_id": agent_id,
            "policy_group": policy_group,
            "worker_group": policy_group,
            "rollout_policy_step": rollout_policy_step,
            "rollout_global_step": rollout_global_step,
            "policy_lag": 0,
            "agent_turn_index": node_id,
            "tree_id": tree_id,
            "prompt_id": tree_id,
            "node_id": node_id,
            "parent_idx": parent_idx,
            "path": list(path),
            "raw_score": verification.score,
            "search_stage": "expand" if parent_idx < 0 else "refine",
            "rollout_source": source,
            "verifier_name": verification.metadata.get("verifier_name", "verl_prime_code"),
            "verifier_success": verification.success,
            "verifier_terminal": verification.terminal,
            "search_stop": self._marti_tree_controller(prompt).should_stop(
                success=verification.success, pending_node=False
            ),
            "verifier_feedback": verification.feedback,
            "failure_type": verification.metadata.get("failure_type"),
            "verification_mode": verification.metadata.get("verification_mode"),
            "verifier_metadata": verification.metadata,
        }

    def _marti_agent_binding(self, node_id: int) -> tuple[str, str]:
        """Bind each tree node to a declared agent and policy group.

        Round-robin assignment is only a deterministic identity contract for
        Stage 1E. Independent actor workers and asynchronous buffers remain a
        separate trainer/backend concern.
        """
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        marti_cfg = config_get(orchestra_cfg, "marti_mars2", default={}) or {}
        agent_ids = to_python(config_get(marti_cfg, "agent_ids", default=None))
        model_ids = to_python(config_get(marti_cfg, "model_ids", default=None))
        if not agent_ids:
            agent_ids = to_python(config_get(agent_cfg, "agent_ids", default=["generator"]))
        if not model_ids:
            model_ids = to_python(config_get(agent_cfg, "model_ids", default=["shared"] * len(agent_ids)))
        agent_ids = [str(item) for item in agent_ids]
        model_ids = [str(item) for item in model_ids]
        if len(model_ids) != len(agent_ids):
            raise ValueError("MARTI agent_ids and model_ids must have matching lengths.")
        index = int(node_id) % len(agent_ids)
        return agent_ids[index], model_ids[index]

    def _marti_prompt_ids(self, prompt: dict[str, Any], *, node_id: int, parent_idx: int | None = None) -> list[int]:
        prompt_ids = self._encode_prompt(to_python(prompt.get("raw_prompt", [])))
        if parent_idx is None:
            parent_idx = self._marti_select_parent(prompt, node_id=node_id)
        if parent_idx < 0:
            return prompt_ids
        tree_state = self._marti_tree_records().get(self._marti_tree_id(prompt), {})
        parent_record = tree_state.get(parent_idx)
        if not parent_record:
            return prompt_ids
        parent_response_ids = parent_record["response_ids"]
        parent_text = self.tokenizer.decode(parent_response_ids, skip_special_tokens=True)
        feedback = str(parent_record.get("feedback", "")).strip()
        refinement_text = (
            f" Previous candidate: {parent_text}"
            f"\nVerifier feedback: {feedback}"
            "\nRefine it and return a complete replacement answer."
        )
        try:
            refinement = self.tokenizer.encode(refinement_text, add_special_tokens=False)
        except TypeError:
            refinement = self.tokenizer.encode(refinement_text)
        return [*prompt_ids, *[int(token_id) for token_id in refinement]][-self.rollout_config.prompt_length :]

    def _remember_marti_response(
        self,
        prompt: dict[str, Any],
        *,
        node_id: int,
        response_ids: list[int],
        parent_idx: int,
        verification: VerifierResult,
    ) -> None:
        state = self._marti_tree_records()
        tree_id = self._marti_tree_id(prompt)
        state.setdefault(tree_id, {})[node_id] = {
            "response_ids": list(response_ids),
            "parent_idx": parent_idx,
            "path": self._marti_path_for_node(prompt, node_id=node_id, parent_idx=parent_idx),
            "reward": float(verification.score),
            "feedback": verification.feedback,
            "success": verification.success,
            "terminal": verification.terminal,
            "metadata": dict(verification.metadata),
        }
        self._marti_tree_controller(prompt).record(
            node_id=node_id,
            parent_idx=parent_idx,
            path=self._marti_path_for_node(prompt, node_id=node_id, parent_idx=parent_idx),
            reward=verification.score,
            feedback=verification.feedback,
            success=verification.success,
            terminal=verification.terminal,
            response_ids=response_ids,
            metadata=verification.metadata,
        )

    def _marti_tree_state(self) -> dict[str, dict[int, list[int]]]:
        # Backward-compatible view used by older callers.
        return {
            tree_id: {node_id: record["response_ids"] for node_id, record in records.items()}
            for tree_id, records in self._marti_tree_records().items()
        }

    def _marti_tree_records(self) -> dict[str, dict[int, dict[str, Any]]]:
        if not hasattr(self, "_trajweave_marti_tree_records"):
            self._trajweave_marti_tree_records = {}
        return self._trajweave_marti_tree_records

    def _marti_tree_controller(self, prompt: dict[str, Any]) -> TreeSearchController:
        if not hasattr(self, "_trajweave_marti_tree_controllers"):
            self._trajweave_marti_tree_controllers = {}
        tree_id = self._marti_tree_id(prompt)
        controllers = self._trajweave_marti_tree_controllers
        if tree_id not in controllers:
            controllers[tree_id] = TreeSearchController(
                max_num_nodes=self._marti_max_num_nodes(),
                initial_candidates=self._marti_initial_candidates(),
                exploration_constant=self._marti_exploration_constant(),
                stop_on_success=self._marti_stop_on_success(),
            )
        return controllers[tree_id]

    def _marti_select_parent(self, prompt: dict[str, Any], *, node_id: int) -> int:
        return self._marti_tree_controller(prompt).select_parent(node_id)

    def _marti_path_for_node(self, prompt: dict[str, Any], *, node_id: int, parent_idx: int) -> tuple[int, ...]:
        return self._marti_tree_controller(prompt).path_for(node_id, parent_idx)

    def _marti_verify(
        self,
        prompt: dict[str, Any],
        *,
        node_id: int,
        parent_idx: int,
        candidate: str,
    ) -> VerifierResult:
        candidate_text = candidate.strip()
        environment = CodeExecutionEnvironment()
        task = environment.from_record({key: to_python(value) for key, value in prompt.items()})
        result = environment.verify_candidate(
            task,
            candidate_text,
            node_id=node_id,
            parent_idx=None if parent_idx < 0 else parent_idx,
        )
        controller_stop = self._marti_tree_controller(prompt).should_stop(success=result.success)
        return VerifierResult(
            score=result.score,
            success=result.success,
            terminal=bool(result.terminal and self._marti_stop_on_success()) or (controller_stop and result.success),
            feedback=result.feedback,
            metadata={
                "verifier_name": "verl_prime_code",
                "candidate_sha256": hashlib.sha256(candidate_text.encode("utf-8")).hexdigest(),
                "candidate_preview": candidate_text[:512],
                **result.metadata,
            },
        )

    def _marti_stop_on_success(self) -> bool:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        marti_cfg = config_get(orchestra_cfg, "marti_mars2", default={}) or {}
        return bool(config_get(marti_cfg, "stop_on_success", default=False))

    def _marti_tree_id(self, prompt: dict[str, Any]) -> str:
        return str(to_python(prompt.get("uid", prompt.get("index", "tree"))))

    def _marti_node_path(self, node_id: int) -> tuple[int, tuple[int, ...]]:
        initial_candidates = self._marti_initial_candidates()
        if node_id < initial_candidates:
            return -1, (node_id,)
        parent_idx = node_id - initial_candidates
        _grandparent, parent_path = self._marti_node_path(parent_idx)
        return parent_idx, (*parent_path, node_id)

    def _marti_max_num_nodes(self) -> int:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        marti_cfg = config_get(orchestra_cfg, "marti_mars2", default={}) or {}
        return int(config_get(marti_cfg, "max_num_nodes", default=2))

    def _marti_initial_candidates(self) -> int:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        marti_cfg = config_get(orchestra_cfg, "marti_mars2", default={}) or {}
        value = int(config_get(marti_cfg, "initial_candidates", default=2))
        return max(1, min(value, self._marti_max_num_nodes()))

    def _marti_exploration_constant(self) -> float:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        marti_cfg = config_get(orchestra_cfg, "marti_mars2", default={}) or {}
        return float(config_get(marti_cfg, "exploration_constant", default=1.0))
