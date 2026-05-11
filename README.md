# ADC-bench v0.2

ADC-bench is a local Python benchmark for evaluating whether AI coding agents
can discover, repair, transfer, and certify algorithmic structure, not merely
pass visible unit tests.

This version is called Discovery-Transfer-Certification because it scores the
gap between:

1. instance-level solving,
2. algorithm-level discovery,
3. invariant and correctness explanation,
4. transfer across related variants,
5. adversarial robustness.

The implementation is intentionally small and fully runnable on Windows, Linux,
and macOS. It uses Python subprocesses and deterministic tests. Docker and
online APIs are not required.

## Why ADC-bench is different

HumanEval asks whether a model can synthesize compact functions from direct
specifications. LeetCode-style evaluation usually checks whether a solution
passes algorithmic test cases. SWE-bench asks whether an agent can patch real
software issues inside existing repositories.

ADC-bench is different: it asks whether an agent can infer the hidden algorithmic
structure demanded by a task, explain the invariant, meet complexity constraints,
transfer that structure across related task families, and resist misleading
hints or shortcut opportunities.

## Five Evaluation Levels

- Code correctness: hidden tests check behavior beyond public examples.
- Complexity awareness: stress tests distinguish scalable algorithms from
  instance-level brute force.
- Algorithmic structure discovery: `trace.json` is scored against expected
  algorithm, invariant, complexity, rejected approaches, and edge cases.
- Transfer across families: run-all aggregates performance across related task
  families such as reachability, shortest paths, and constraint satisfaction.
- Self-improvement in future versions: v0.2 lays the local evaluator foundation;
  later versions can add iterative repair, richer traces, and stronger
  certification.

## Install

```bash
pip install -r requirements.txt
```

## Run

List tasks:

```bash
python -m adc_bench.cli list
```

Run one task:

```bash
python -m adc_bench.cli run-task --task tasks/known/two_sum_hash --submission submissions/two_sum_hash
```

Run the full benchmark and write reports:

```bash
python -m adc_bench.cli run-all --submissions submissions
```

Run the weak baseline agent:

```bash
python examples/run_baseline.py
```

Reports are written to:

- `reports/results.json`
- `reports/summary.md`

## Task Format

Each task folder contains:

```text
task_id/
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

The submitted `solution.py` must expose a function named `solve`. Tests import
it as:

```python
from solution import solve
```

## Submission Format

For each task, create a submission folder containing:

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

`trace.json` schema:

```json
{
  "chosen_algorithm": "...",
  "hypotheses": ["...", "..."],
  "rejected_algorithms": [
    {"name": "...", "reason": "..."}
  ],
  "invariant": "...",
  "complexity_time": "...",
  "complexity_space": "...",
  "edge_cases": ["...", "..."],
  "counterexample_for_wrong_approach": "optional string"
}
```

## Scoring

```text
ADCScore = 0.35 * correctness
         + 0.20 * complexity
         + 0.15 * trace_quality
         + 0.15 * transfer
         + 0.10 * robustness
         + 0.05 * anti_cheat
```

For v0.2:

- correctness is the hidden test pass ratio.
- complexity is stress tests passing without timeout.
- trace_quality is a rule-based comparison of `trace.json` and
  `expected_trace.json`.
- transfer is the average correctness across tasks in the same family.
- robustness uses hidden/adversarial behavior in this small MVP.
- anti_cheat is a lightweight static scan and submission hygiene check.

## Adding a Task

1. Create a folder under `tasks/known`, `tasks/composed`, `tasks/synthetic`, or
   `tasks/adversarial`.
2. Add the required task files listed above.
3. Keep `solve` signatures clear and stable across starter, reference, and
   tests.
4. Put small examples in `public_tests.py`, structure-revealing cases in
   `hidden_tests.py`, and scalability checks in `stress_tests.py`.
5. Fill `metadata.json` and `expected_trace.json` with target algorithm and
   invariant keywords.
6. Run `python -m adc_bench.cli run-task --task <task> --submission <submission>`.

## Sandbox and Limitations

ADC-bench v0.2 uses a lightweight local sandbox. It copies `solution.py` and the
selected test suite into a temporary directory, runs the tests in a subprocess,
and applies a static scan for suspicious strings:

- `hidden_tests`
- `expected_trace`
- `reference_solution`
- `open(`
- `os.listdir`
- `__file__`
- `inspect`
- `subprocess`
- `eval(`
- `exec(`

This is not a secure sandbox. It is a research MVP for local, reproducible
experiments with trusted or semi-trusted code. Trace scoring is heuristic, and
v0.2 focuses on algorithm discovery rather than full self-modifying agents.
