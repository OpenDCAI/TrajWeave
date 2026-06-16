# Claude / Agent Notes for TrajWeave

This project is no longer a direct VERL mirror. It keeps VERL as a backend and is being cleaned for Multi-Agent LLM RL development.

Follow `AGENTS.md` for repository rules. In short:

- Keep `verl/` stable as the training backend.
- Put new MASRL orchestration, trajectory, reward, credit assignment, and backend adapters in a TrajWeave layer.
- Do not restore upstream-only recipe, Docker, CI, or docs bulk unless it directly supports TrajWeave.
- Preserve upstream copyright headers in copied source files.
