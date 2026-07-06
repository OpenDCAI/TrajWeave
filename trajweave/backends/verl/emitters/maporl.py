from __future__ import annotations

from typing import Any

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput

from trajweave.backends.verl.runtime_config import config_get
from trajweave.backends.verl.schema import to_python


class MAPoRLEmitterMixin:
    def _build_maporl_debate_math_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        reward_model = to_python(prompt.get("reward_model", {})) or {}
        ground_truth = str(reward_model.get("ground_truth", "2"))
        prompt_ids = self._encode_prompt(raw_prompt)
        is_correct = session_id % 2 == 0
        reward = 1.0 if is_correct else 0.0
        agent_ids = self._maporl_agent_ids()
        model_ids = self._maporl_model_ids(default_agent_ids=agent_ids)
        max_rounds = self._maporl_max_rounds()
        consensus_threshold = self._maporl_consensus_threshold(default=len(agent_ids))
        early_stop = self._maporl_early_stop()
        reward_feedback = self._maporl_reward_feedback()
        consensus_answer = ground_truth if is_correct else "__wrong__"
        consensus_reached = bool(is_correct)
        finished_round = 0 if consensus_reached and early_stop else -1
        outputs: list[AgentLoopOutput] = []
        running_prompt_ids = list(prompt_ids)
        metrics = AgentLoopMetrics(generate_sequences=0.0, tool_calls=0.0, compute_score=0.0, num_preempted=-1)
        for round_id in range(max_rounds):
            round_answer = ground_truth if is_correct else "__wrong__"
            for agent_index, agent_id in enumerate(agent_ids):
                text = f" Agent {agent_id} round {round_id}. Final answer: {round_answer}"
                response_ids = self._encode_text(text)
                outputs.append(
                    AgentLoopOutput(
                        prompt_ids=list(running_prompt_ids),
                        response_ids=response_ids,
                        response_mask=[1] * len(response_ids),
                        reward_score=reward,
                        num_turns=round_id + 1,
                        metrics=metrics,
                        extra_fields={
                            "turn_scores": [],
                            "tool_rewards": [],
                            "trajweave_agent_name": agent_id,
                            "trajweave_role": "solver",
                            "agent_id": agent_id,
                            "policy_group": model_ids[agent_index],
                            "worker_group": model_ids[agent_index],
                            "worker_group_model_path": self._maporl_worker_group_model_path(model_ids[agent_index]),
                            "agent_answer": round_answer,
                            "raw_score": reward,
                            "correctness": reward,
                            "consensus_answer": consensus_answer,
                            "consensus_reached": consensus_reached,
                            "communication_graph": "fully_connected",
                            "aggregation": "consensus",
                            "reward_feedback": reward_feedback,
                            "round_id": round_id,
                            "agent_index": agent_index,
                            "consensus_threshold": consensus_threshold,
                            "finished_round": finished_round,
                        },
                    )
                )
                running_prompt_ids = running_prompt_ids + response_ids
            if consensus_reached and early_stop:
                break

        if outputs:
            outputs[-1].extra_fields["final_consensus_turn"] = True
        return outputs

    def _build_hf_maporl_debate_math_outputs(
        self,
        prompt: dict[str, Any],
        *,
        session_id: int = 0,
    ) -> list[AgentLoopOutput]:
        raw_prompt = to_python(prompt.get("raw_prompt", []))
        prompt_ids = self._encode_prompt(raw_prompt)
        is_correct = session_id % 2 == 0
        reward = 1.0 if is_correct else 0.0
        agent_ids = self._maporl_agent_ids()
        model_ids = self._maporl_model_ids(default_agent_ids=agent_ids)
        max_rounds = self._maporl_max_rounds()
        consensus_threshold = self._maporl_consensus_threshold(default=len(agent_ids))
        early_stop = self._maporl_early_stop()
        reward_feedback = self._maporl_reward_feedback()
        reward_model = to_python(prompt.get("reward_model", {})) or {}
        ground_truth = str(reward_model.get("ground_truth", "2"))
        round_answer = ground_truth if is_correct else "__wrong__"
        consensus_answer = ground_truth if is_correct else "__wrong__"
        consensus_reached = bool(is_correct)
        finished_round = 0 if consensus_reached and early_stop else -1
        outputs: list[AgentLoopOutput] = []
        running_prompt_ids = list(prompt_ids)
        metrics = AgentLoopMetrics(generate_sequences=1.0, tool_calls=0.0, compute_score=0.0, num_preempted=-1)
        for round_id in range(max_rounds):
            for agent_index, agent_id in enumerate(agent_ids):
                policy_group = model_ids[agent_index]
                response_ids = self._generate_local_response_ids(running_prompt_ids, policy_group=policy_group)
                outputs.append(
                    AgentLoopOutput(
                        prompt_ids=list(running_prompt_ids),
                        response_ids=response_ids,
                        response_mask=[1] * len(response_ids),
                        reward_score=reward,
                        num_turns=round_id + 1,
                        metrics=metrics,
                        extra_fields={
                            "turn_scores": [],
                            "tool_rewards": [],
                            "trajweave_agent_name": agent_id,
                            "trajweave_role": "solver",
                            "agent_id": agent_id,
                            "policy_group": policy_group,
                            "worker_group": policy_group,
                            "worker_group_model_path": self._maporl_worker_group_model_path(policy_group),
                            "agent_answer": round_answer,
                            "raw_score": reward,
                            "correctness": reward,
                            "consensus_answer": consensus_answer,
                            "consensus_reached": consensus_reached,
                            "communication_graph": "fully_connected",
                            "aggregation": "consensus",
                            "reward_feedback": reward_feedback,
                            "round_id": round_id,
                            "agent_index": agent_index,
                            "consensus_threshold": consensus_threshold,
                            "finished_round": finished_round,
                            "rollout_source": "hf_local_tq",
                        },
                    )
                )
                running_prompt_ids = running_prompt_ids + response_ids
            if consensus_reached and early_stop:
                break

        if outputs:
            outputs[-1].extra_fields["final_consensus_turn"] = True
        return outputs

    def _maporl_agent_ids(self) -> list[str]:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        ids = to_python(config_get(agent_cfg, "agent_ids", default=None))
        if ids:
            return [str(item) for item in ids]
        return ["agent_0", "agent_1"]

    def _maporl_model_ids(self, *, default_agent_ids: list[str]) -> list[str]:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        ids = to_python(config_get(agent_cfg, "model_ids", default=None))
        if ids:
            model_ids = [str(item) for item in ids]
            if len(model_ids) != len(default_agent_ids):
                raise ValueError("agent.model_ids must have the same length as agent.agent_ids.")
            return model_ids
        return ["shared" for _ in default_agent_ids]

    def _maporl_max_rounds(self) -> int:
        maporl_cfg = self._maporl_orchestra_config()
        return int(config_get(maporl_cfg, "max_rounds", default=2))

    def _maporl_consensus_threshold(self, *, default: int) -> int:
        maporl_cfg = self._maporl_orchestra_config()
        return int(config_get(maporl_cfg, "consensus_threshold", default=default))

    def _maporl_early_stop(self) -> bool:
        maporl_cfg = self._maporl_orchestra_config()
        return bool(config_get(maporl_cfg, "early_stop", default=True))

    def _maporl_reward_feedback(self) -> bool:
        maporl_cfg = self._maporl_orchestra_config()
        return bool(config_get(maporl_cfg, "reward_feedback", default=False))

    def _maporl_orchestra_config(self) -> Any:
        agent_cfg = config_get(self.config, "agent", default={}) or {}
        orchestra_cfg = config_get(agent_cfg, "orchestra", default={}) or {}
        return config_get(orchestra_cfg, "maporl", default={}) or {}
