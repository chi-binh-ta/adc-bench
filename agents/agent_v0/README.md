# Agent v0

`agent_v0` is a deliberately weak baseline. It copies each task's `starter.py`
to `solution.py` and writes a shallow `trace.json` with generic guesses.

It exists to test the evaluator plumbing, report generation, sandbox behavior,
and anti-cheat checks. It should not solve most ADC-bench tasks.

Run it indirectly:

```bash
python examples/run_baseline.py
```

Or for one task:

```bash
python agents/agent_v0/agent.py --task tasks/known/two_sum_hash --submission submissions/two_sum_hash
python -m adc_bench.cli run-task --task tasks/known/two_sum_hash --submission submissions/two_sum_hash
```

For self-improvement evaluation, the same agent also supports the shared agent
interface:

```bash
python agents/agent_v0/agent.py --task tasks/known/two_sum_hash --out submissions/two_sum_hash
```
