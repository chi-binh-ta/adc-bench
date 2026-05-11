# Self-Improvement Tasks

ADC-bench v0.3.0 adds an offline self-improvement layer. A self-improvement task
packages:

- a weak baseline agent,
- failure evidence from training tasks,
- a held-out task list,
- an improvement goal,
- an expected improvement trace rubric.

The evaluator does not run live recursive self-modification. Instead, it
compares a fixed baseline agent directory and a fixed candidate agent directory
on the held-out tasks.

Run the sample:

```bash
python -m adc_bench.cli self-improve-eval \
  --baseline-agent agents/agent_v0 \
  --candidate-agent examples/sample_self_improve \
  --task self_improvement/tasks/basic_agent_upgrade \
  --timeout 5 \
  --format text
```
