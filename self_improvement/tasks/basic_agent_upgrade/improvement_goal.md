# Basic Agent Upgrade

`agent_v0` is a weak baseline. It mostly copies each task's starter code into
`solution.py` and emits a generic trace with no real algorithm discovery.

The candidate agent should improve:

- algorithm selection,
- generated `solution.py` quality,
- `trace.json` quality,
- generalization from training failures to held-out algorithm discovery tasks.

The candidate should generate valid submissions for tasks. It should not
hard-code hidden behavior, inspect private task files, or modify evaluator files.
It should rely on visible task materials such as `problem.md`, `metadata.json`,
and `starter.py`.

For this v0.3.1 MVP, the held-out list intentionally contains small known tasks
so the self-improvement evaluator can be tested deterministically. In v0.3.1,
training and held-out task lists must be disjoint: training failure logs are only
provided for training tasks, while held-out tasks are used only for evaluation.
