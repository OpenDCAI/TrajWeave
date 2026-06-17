from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from trajweave.backends.trainable_tiny_math import TrainableTinyMathPolicyBackend
from trajweave.credit.doctor_mas import DoctorMASCreditAssigner
from trajweave.envs.math import MathTask, SolverVerifierMathEnvironment
from trajweave.orchestration.solver_verifier import SolverVerifierOrchestra
from trajweave.core.specs import AgentSpec, PolicyGroupSpec, TeamSpec
from trajweave.rollout.engine import RolloutEngine


@dataclass
class TrainConfig:
    steps: int = 120
    batch_tasks: int = 8
    rollouts_per_task: int = 8
    max_turns: int = 2
    eval_interval: int = 10
    lr: float = 0.03
    entropy_coef: float = 0.02
    seed: int = 7
    device: str = "cpu"
    output_dir: str = "outputs/doctor_mas_tiny_train"
    task_mode: str = "fixed"
    num_solver_candidates: int = 3
    train_verifier: bool = False


def build_train_engine(config: TrainConfig) -> tuple[RolloutEngine, TrainableTinyMathPolicyBackend]:
    backend = TrainableTinyMathPolicyBackend(
        device=config.device,
        seed=config.seed,
        num_solver_candidates=config.num_solver_candidates,
        train_verifier=config.train_verifier,
    )
    team = TeamSpec(
        name="doctor_mas_tiny_train_math",
        agents=(
            AgentSpec(name="solver", role="solver", policy_group="tiny_solver", trainable=True),
            AgentSpec(name="verifier", role="verifier", policy_group="rule_verifier", trainable=config.train_verifier),
        ),
        policy_groups=(
            PolicyGroupSpec(name="tiny_solver", backend="tiny-torch", trainable=True),
            PolicyGroupSpec(name="rule_verifier", backend="rule", trainable=config.train_verifier),
        ),
        orchestra="solver_verifier",
        reward="math_exact_match",
        credit="doctor_mas_agent_wise_grpo",
        max_turns=config.max_turns,
    )
    engine = RolloutEngine(
        team=team,
        orchestra=SolverVerifierOrchestra(),
        environment=SolverVerifierMathEnvironment(),
        policy_backend=backend,
        credit_assigner=DoctorMASCreditAssigner(),
    )
    return engine, backend


def make_math_tasks(prefix: str, count: int, rng: random.Random) -> list[MathTask]:
    tasks = []
    ops = ["+", "*", "-"]
    for idx in range(count):
        left = rng.randint(0, 9)
        right = rng.randint(0, 9)
        op = rng.choice(ops)
        if op == "+":
            answer = left + right
        elif op == "-":
            answer = left - right
        else:
            answer = left * right
        tasks.append(MathTask(task_id=f"{prefix}_{idx}_{left}_{op}_{right}", question=f"What is {left} {op} {right}?", answer=answer))
    return tasks


def fixed_math_tasks(prefix: str) -> list[MathTask]:
    specs = [
        (1, "+", 1),
        (2, "*", 3),
        (4, "+", 5),
        (8, "-", 3),
        (3, "*", 3),
        (9, "-", 6),
        (2, "+", 7),
        (5, "*", 2),
    ]
    tasks = []
    for idx, (left, op, right) in enumerate(specs):
        if op == "+":
            answer = left + right
        elif op == "-":
            answer = left - right
        else:
            answer = left * right
        tasks.append(MathTask(task_id=f"{prefix}_{idx}_{left}_{op}_{right}", question=f"What is {left} {op} {right}?", answer=answer))
    return tasks


def evaluate(engine: RolloutEngine, backend: TrainableTinyMathPolicyBackend, tasks: list[MathTask]) -> float:
    was_sampling = backend.sample_actions
    backend.eval()
    result = engine.run(tasks, rollouts_per_task=1)
    backend.sample_actions = was_sampling
    if was_sampling:
        backend.train()
    return result.success_rate


def run_training(config: TrainConfig) -> list[dict]:
    import torch

    rng = random.Random(config.seed)
    torch.manual_seed(config.seed)
    engine, backend = build_train_engine(config)
    optimizer = torch.optim.Adam(backend.parameters(), lr=config.lr)
    eval_tasks = (
        fixed_math_tasks("eval")
        if config.task_mode == "fixed"
        else make_math_tasks("eval", 64, random.Random(config.seed + 10_000))
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    metrics_history: list[dict] = []
    with log_path.open("w", encoding="utf-8") as log_file:
        for step in range(config.steps + 1):
            if config.task_mode == "fixed":
                train_tasks = fixed_math_tasks("train")
            elif config.task_mode == "random":
                train_tasks = make_math_tasks(f"train_step_{step}", config.batch_tasks, rng)
            else:
                raise ValueError(f"Unknown task_mode: {config.task_mode}")
            backend.train()
            result = engine.run(train_tasks, rollouts_per_task=config.rollouts_per_task)
            loss, loss_metrics = backend.policy_loss(result.samples, entropy_coef=config.entropy_coef)

            if step > 0:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(backend.parameters(), 1.0)
                optimizer.step()

            should_eval = step == 0 or step % config.eval_interval == 0 or step == config.steps
            eval_success = evaluate(engine, backend, eval_tasks) if should_eval else None
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "train_success_rate": result.success_rate,
                "train_trajectories": len(result.trajectories),
                "train_samples": len(result.samples),
                "eval_success_rate": eval_success,
                **loss_metrics,
            }
            metrics_history.append(row)
            log_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            log_file.flush()
            if should_eval:
                print(
                    f"step={step:04d}",
                    f"loss={row['loss']:.4f}",
                    f"train_success={row['train_success_rate']:.3f}",
                    f"eval_success={eval_success:.3f}",
                    f"entropy={row.get('mean_entropy', 0.0):.3f}",
                )

    torch.save({"solver": backend.solver.state_dict(), "verifier": backend.verifier.state_dict()}, output_dir / "tiny_policy.pt")
    return metrics_history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny DrMAS-style Solver-Verifier math policy.")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--batch-tasks", type=int, default=8)
    parser.add_argument("--rollouts-per-task", type=int, default=8)
    parser.add_argument("--max-turns", type=int, default=2)
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--entropy-coef", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", default="outputs/doctor_mas_tiny_train")
    parser.add_argument("--task-mode", choices=["fixed", "random"], default="fixed")
    parser.add_argument("--num-solver-candidates", type=int, default=3)
    parser.add_argument("--train-verifier", action="store_true")
    args = parser.parse_args()
    run_training(TrainConfig(**vars(args)))


if __name__ == "__main__":
    main()
