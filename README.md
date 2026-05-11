# ADC-bench

**ADC-bench** — **Algorithm Discovery Complexity Benchmark** — is a local Python benchmark for evaluating whether AI coding agents can **discover, repair, transfer, and certify algorithmic structure**, not merely pass visible unit tests.

ADC-bench is built around a simple research question:

> Can an AI agent identify the hidden algorithmic structure required to solve a problem under correctness, complexity, transfer, and anti-cheating constraints?

Unlike ordinary coding benchmarks, ADC-bench does not only ask whether code passes tests. It also asks whether the agent understands the algorithmic idea, rejects weaker approaches, explains the invariant, and satisfies the expected complexity.

---

## 1. Motivation

Most programming benchmarks evaluate whether a model can produce code that passes unit tests.

ADC-bench focuses on a deeper distinction:

```text
Instance solving       = solve one concrete input
Code patching          = produce code that passes tests
Algorithm discovery    = infer the hidden algorithmic structure
Certification          = explain why the algorithm works
Transfer               = apply the same structure to related variants
```

ADC-bench is inspired by software engineering benchmarks, but it is not a SWE-bench clone. SWE-style benchmarks usually ask whether an agent can patch real software issues. ADC-bench asks whether an agent can discover the underlying algorithmic structure behind a task.

The central principle is:

```text
Passing tests is not the same as discovering an algorithm.
```

---

## 2. Core Rule of the Game

For each task, an agent receives a problem specification and starter code.

The agent must produce two files:

```text
solution.py
trace.json
```

The evaluator then checks:

1. whether the solution passes hidden tests,
2. whether it satisfies complexity requirements,
3. whether the trace correctly identifies the algorithm,
4. whether the solution transfers across related task families,
5. whether it is robust to adversarial traps,
6. whether it avoids cheating.

The benchmark therefore evaluates both:

```text
code correctness
```

and

```text
algorithmic discovery quality
```

---

## 3. Repository Structure

```text
adc-bench/
  README.md
  LICENSE
  requirements.txt
  pyproject.toml

  adc_bench/
    __init__.py
    cli.py
    evaluator.py
    scoring.py
    sandbox.py
    schemas.py
    trace_validation.py
    anti_cheat.py
    report.py

  tasks/
    known/
    composed/
    synthetic/
    adversarial/

  agents/
    agent_v0/
      README.md
      agent.py

  submissions/
    .gitkeep

  reports/
    .gitkeep

  examples/
    run_baseline.py
```

---

## 4. Task Format

Each task lives in:

```text
tasks/<split>/<task_id>/
```

Each task folder must contain:

```text
problem.md
starter.py
reference_solution.py
public_tests.py
hidden_tests.py
stress_tests.py
expected_trace.json
metadata.json
forbidden_shortcuts.md
```

Example:

```text
tasks/known/two_sum_hash/
  problem.md
  starter.py
  reference_solution.py
  public_tests.py
  hidden_tests.py
  stress_tests.py
  expected_trace.json
  metadata.json
  forbidden_shortcuts.md
```

---

## 5. Files Visible to the Agent

The agent may use:

```text
problem.md
starter.py
metadata.json
public_tests.py
forbidden_shortcuts.md
```

These files define the task, the starter implementation, public examples, and forbidden shortcuts.

---

## 6. Files Hidden from the Agent

The agent must not read or depend on:

```text
hidden_tests.py
stress_tests.py
reference_solution.py
expected_trace.json
```

These files are used only by the evaluator.

Reading or hard-coding against these files is considered cheating.

---

## 7. Required Submission Format

For a task with ID:

```text
two_sum_hash
```

the submission should be placed at:

```text
submissions/two_sum_hash/
```

and must contain:

```text
solution.py
trace.json
```

---

## 8. `solution.py`

`solution.py` must define a function named:

```python
def solve(...):
    ...
```

The function signature must match the task description in `problem.md`.

Example:

```python
def solve(nums, target):
    seen = set()
    for x in nums:
        if target - x in seen:
            return True
        seen.add(x)
    return False
```

---

## 9. `trace.json`

`trace.json` is a structured technical explanation of the discovered algorithm.

It is not private chain-of-thought. It should be a concise, inspectable explanation of the algorithmic decision.

Required schema:

```json
{
  "chosen_algorithm": "...",
  "hypotheses": ["...", "..."],
  "rejected_algorithms": [
    {
      "name": "...",
      "reason": "..."
    }
  ],
  "invariant": "...",
  "complexity_time": "...",
  "complexity_space": "...",
  "edge_cases": ["...", "..."],
  "counterexample_for_wrong_approach": "optional string"
}
```

Example:

```json
{
  "chosen_algorithm": "hash set lookup",
  "hypotheses": [
    "try all pairs",
    "sort and use two pointers",
    "use a hash set of previously seen values"
  ],
  "rejected_algorithms": [
    {
      "name": "brute force",
      "reason": "O(n^2) time violates the expected O(n) complexity"
    }
  ],
  "invariant": "After processing index i, the set contains exactly the elements before i.",
  "complexity_time": "O(n)",
  "complexity_space": "O(n)",
  "edge_cases": [
    "empty list",
    "duplicate values",
    "negative numbers"
  ],
  "counterexample_for_wrong_approach": ""
}
```

---

## 10. What Counts as Algorithm Discovery?

A high-quality ADC-bench submission should do more than pass tests.

It should identify the intended structure behind the problem.

Examples:

| Task Type | Expected Discovery |
|---|---|
| Two Sum | Hash set lookup |
| Valid Parentheses | Stack invariant |
| Graph Reachability | BFS or DFS traversal |
| Positive Weighted Shortest Path | Dijkstra relaxation |
| Sliding Window Maximum | Monotonic deque |
| XOR-SAT | Gaussian elimination over GF(2) |
| 2-SAT | Implication graph + strongly connected components |
| Time-dependent Reachability | BFS over expanded state `(node, time_mod)` |

A strong submission should explain:

1. why the chosen algorithm works,
2. why weaker approaches fail,
3. what invariant is maintained,
4. what the time and space complexity are,
5. which edge cases matter.

---

## 11. Scoring

ADC-bench uses the following score:

```text
ADCScore =
    0.35 * correctness
  + 0.20 * complexity
  + 0.15 * trace_quality
  + 0.15 * transfer
  + 0.10 * robustness
  + 0.05 * anti_cheat
```

Each component is normalized to the range:

```text
[0, 1]
```

---

## 12. Correctness

Correctness measures whether `solution.py` passes hidden tests.

```text
correctness = hidden_tests_passed / hidden_tests_total
```

Public tests are useful for debugging, but they are not enough to establish correctness.

---

## 13. Complexity

Complexity measures whether the implementation satisfies the expected asymptotic behavior.

This is tested using:

```text
stress_tests.py
timeouts
large inputs
scaling-sensitive cases
```

Example:

If a task expects `O(n)` but the submission uses `O(n^2)`, then it may pass public tests but fail stress tests.

---

## 14. Trace Quality

Trace quality measures whether `trace.json` correctly identifies the algorithmic structure.

The validator checks:

```text
chosen_algorithm
hypotheses
rejected_algorithms
invariant
complexity_time
complexity_space
edge_cases
counterexample_for_wrong_approach
```

The current trace validator is heuristic. It is not a formal proof checker.

---

## 15. Transfer

Transfer measures whether the agent performs well across related tasks in the same family.

Example:

```text
shortest_path_family/
  dijkstra_positive_weights
  wrong_hint_shortest_path
  path_reconstruction_variant
```

An agent receives a higher transfer score if the discovered strategy generalizes across variants rather than only solving one task by accident.

---

## 16. Robustness

Robustness measures performance on adversarial or diagonal tasks.

These tasks may contain:

```text
misleading hints
small public tests that allow brute force
hidden complexity traps
ambiguous edge cases
wrong algorithm suggestions
```

A robust agent should not blindly follow misleading hints.

Example: the problem says "This probably needs BFS", but the graph has positive weights, so the correct method is Dijkstra.

---

## 17. Anti-Cheat Rules

ADC-bench includes a lightweight local anti-cheat scanner.

Forbidden behavior includes:

```text
reading hidden_tests.py
reading stress_tests.py
reading reference_solution.py
reading expected_trace.json
modifying evaluator files
hard-coding test outputs
using eval or exec to bypass restrictions
using subprocess to inspect benchmark files
using filesystem introspection to discover hidden files
```

Suspicious strings may include:

```text
hidden_tests
stress_tests
reference_solution
expected_trace
os.listdir
__file__
inspect
subprocess
eval(
exec(
```

Important limitation:

```text
The local sandbox is not secure against malicious code.
```

The current anti-cheat system is designed to catch obvious violations in a research prototype, not to provide strong security isolation.

---

## 18. Task Splits

ADC-bench v0.2 contains four task splits:

```text
tasks/
  known/
  composed/
  synthetic/
  adversarial/
```

### 18.1 Known Tasks

Canonical algorithmic tasks.

Examples:

```text
two_sum_hash
valid_parentheses_stack
graph_reachability_bfs
dijkstra_positive_weights
```

These tasks test whether an agent can recognize and implement standard algorithmic structures.

### 18.2 Composed Tasks

Tasks requiring composition of known techniques.

Examples:

```text
sliding_window_max_deque
xor_sat_f2
two_sat_scc
```

These tasks test whether an agent can combine multiple ideas or choose the correct abstraction.

### 18.3 Synthetic Tasks

Original or less-standard problems designed to reduce pure memorization.

Examples:

```text
parity_interval_merge
rotating_key_reachability
```

These tasks test whether an agent can analyze a new problem rather than only recall a known template.

### 18.4 Adversarial Tasks

Tasks designed to expose shortcut behavior.

Example:

```text
wrong_hint_shortest_path
```

These tasks may include misleading hints, hidden complexity traps, or cases where a common algorithmic guess fails.

---

## 19. Current Initial Task Set

ADC-bench v0.2 includes 10 initial tasks:

```text
known/two_sum_hash
known/valid_parentheses_stack
known/graph_reachability_bfs
known/dijkstra_positive_weights

composed/sliding_window_max_deque
composed/xor_sat_f2
composed/two_sat_scc

synthetic/parity_interval_merge
synthetic/rotating_key_reachability

adversarial/wrong_hint_shortest_path
```

---

## 20. Installation

Clone the repository:

```bash
git clone https://github.com/chi-binh-ta/adc-bench.git
cd adc-bench
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 21. Running the Benchmark

List all tasks:

```bash
python -m adc_bench.cli list
```

Run the weak baseline agent:

```bash
python examples/run_baseline.py
```

Run one task:

```bash
python -m adc_bench.cli run-task --task tasks/known/two_sum_hash --submission submissions/two_sum_hash
```

Run all tasks:

```bash
python -m adc_bench.cli run-all --submissions submissions
```

Results are written to:

```text
reports/results.json
reports/summary.md
```

---

## 22. Result Format

Each evaluated task produces a result like:

```json
{
  "task_id": "known/two_sum_hash",
  "correctness": 1.0,
  "complexity": 1.0,
  "trace_quality": 0.8,
  "transfer": 0.7,
  "robustness": 1.0,
  "anti_cheat": 1.0,
  "adc_score": 0.91,
  "passed_public": true,
  "passed_hidden": true,
  "passed_stress": true,
  "errors": []
}
```

---

## 23. Adding a New Task

To add a new task, create a new folder:

```text
tasks/<split>/<task_id>/
```

Required files:

```text
problem.md
starter.py
reference_solution.py
public_tests.py
hidden_tests.py
stress_tests.py
expected_trace.json
metadata.json
forbidden_shortcuts.md
```

Minimum `metadata.json`:

```json
{
  "task_id": "my_new_task",
  "family": "my_algorithm_family",
  "split": "known",
  "target_algorithm": "hash set",
  "expected_complexity": "O(n)",
  "difficulty": "easy",
  "requires_invariant": true,
  "requires_counterexample": false,
  "anti_cheat": true
}
```

A good ADC-bench task should have:

1. a hidden algorithmic structure,
2. a naive solution that fails stress tests,
3. hidden tests that check edge cases,
4. a clear expected invariant,
5. a meaningful complexity requirement,
6. at least one way for a shallow shortcut to fail.

---

## 24. Adding a Submission

Create a submission folder:

```text
submissions/<task_id>/
```

Add:

```text
solution.py
trace.json
```

Example:

```text
submissions/two_sum_hash/
  solution.py
  trace.json
```

Then run:

```bash
python -m adc_bench.cli run-task --task tasks/known/two_sum_hash --submission submissions/two_sum_hash
```

---

## 25. Baseline Agent

The repository includes a weak baseline agent:

```text
agents/agent_v0/
  agent.py
```

This agent mostly copies starter code and emits a weak trace. It is intentionally limited.

Its purpose is to verify that the evaluator works, weak agents do not score too highly, and hidden/stress tests catch naive solutions.

Run it with:

```bash
python examples/run_baseline.py
```

---

## 26. GitHub Actions Smoke Test

This repository includes a smoke test workflow:

```text
.github/workflows/smoke-test.yml
```

The workflow checks that dependencies install, tasks can be listed, and the baseline agent can run. This is not a full benchmark run; it is a lightweight sanity check for repository health.

---

## 27. Design Philosophy

ADC-bench is based on the following principle:

```text
A model should not be rewarded only for passing tests.
It should be rewarded for discovering the algorithmic reason why the solution works.
```

A standard coding benchmark asks:

```text
Can the model produce code that passes unit tests?
```

ADC-bench asks:

```text
Can the model identify, implement, transfer, and explain the algorithmic structure required by the task?
```

---

## 28. What ADC-bench Is Not

ADC-bench is not:

```text
a secure sandbox
a full formal verification system
a replacement for SWE-bench
a replacement for HumanEval
a large-scale production benchmark yet
```

ADC-bench is currently a research prototype for studying algorithm discovery behavior in AI coding agents.

---

## 29. Limitations

ADC-bench v0.2 has several known limitations:

1. The local sandbox is not secure against malicious code.
2. Trace scoring is heuristic.
3. Transfer scoring is currently approximate.
4. The task set is small.
5. Some tasks may still resemble known competitive-programming patterns.
6. The benchmark does not yet fully evaluate self-modifying agents.
7. The anti-cheat system catches obvious violations, not sophisticated attacks.

---

## 30. Roadmap

Planned improvements:

```text
v0.2.1 — Better verifier module and cleaner reports
v0.3   — Self-improving agent layer
v0.4   — Stronger synthetic task generation
v0.5   — Human-verified task subset
v1.0   — Stable benchmark release
```

---

## 31. Future Direction: Self-Improving Agent Layer

A future version of ADC-bench may include tasks where an agent receives:

```text
agent_v0 source code
failure logs
held-out benchmark tasks
```

The agent must produce:

```text
agent_v1
```

The evaluator then checks whether:

```text
Score(agent_v1) > Score(agent_v0)
```

on held-out tasks. This would move ADC-bench from algorithm discovery toward recursive algorithm improvement.

---

## 32. Summary

ADC-bench evaluates five levels of capability:

```text
Level 1 — Code correctness
Level 2 — Complexity awareness
Level 3 — Algorithmic structure discovery
Level 4 — Transfer across problem families
Level 5 — Robustness and future self-improvement
```

The goal is not to reward code that merely passes visible tests.

The goal is to evaluate whether an AI agent can discover, repair, transfer, certify, and robustly apply algorithmic structure.
