\# ADC-bench



\*\*ADC-bench\*\* — \*\*Algorithm Discovery Complexity Benchmark\*\* — is a local Python benchmark for evaluating whether AI coding agents can \*\*discover, repair, transfer, and certify algorithmic structure\*\*, not merely pass visible unit tests.



ADC-bench is built around a simple research question:



> Can an AI agent identify the hidden algorithmic structure required to solve a problem under correctness, complexity, transfer, and anti-cheating constraints?



Unlike ordinary coding benchmarks, ADC-bench does not only ask whether code passes tests. It also asks whether the agent understands the algorithmic idea, rejects weaker approaches, explains the invariant, and satisfies the expected complexity.



\---



\## 1. Motivation



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



\---



\## 2. Core Rule of the Game



For each task, an agent receives a problem specification and starter code.



The agent must produce two files:



```text

solution.py

trace.json

```



The evaluator then checks:



1\. whether the solution passes hidden tests,

2\. whether it satisfies complexity requirements,

3\. whether the trace correctly identifies the algorithm,

4\. whether the solution transfers across related task families,

5\. whether it is robust to adversarial traps,

6\. whether it avoids cheating.



The benchmark therefore evaluates both:



```text

code correctness

```



and



```text

algorithmic discovery quality

```



\---



\## 3. Repository Structure



```text

adc-bench/

&#x20; README.md

&#x20; LICENSE

&#x20; requirements.txt

&#x20; pyproject.toml



&#x20; adc\_bench/

&#x20;   \_\_init\_\_.py

&#x20;   cli.py

&#x20;   evaluator.py

&#x20;   scoring.py

&#x20;   sandbox.py

&#x20;   schemas.py

&#x20;   trace\_validation.py

&#x20;   anti\_cheat.py

&#x20;   report.py



&#x20; tasks/

&#x20;   known/

&#x20;   composed/

&#x20;   synthetic/

&#x20;   adversarial/



&#x20; agents/

&#x20;   agent\_v0/

&#x20;     README.md

&#x20;     agent.py



&#x20; submissions/

&#x20;   .gitkeep



&#x20; reports/

&#x20;   .gitkeep



&#x20; examples/

&#x20;   run\_baseline.py

```



\---



\## 4. Task Format



Each task lives in:



```text

tasks/<split>/<task\_id>/

```



Each task folder must contain:



```text

problem.md

starter.py

reference\_solution.py

public\_tests.py

hidden\_tests.py

stress\_tests.py

expected\_trace.json

metadata.json

forbidden\_shortcuts.md

```



Example:



```text

tasks/known/two\_sum\_hash/

&#x20; problem.md

&#x20; starter.py

&#x20; reference\_solution.py

&#x20; public\_tests.py

&#x20; hidden\_tests.py

&#x20; stress\_tests.py

&#x20; expected\_trace.json

&#x20; metadata.json

&#x20; forbidden\_shortcuts.md

```



\---



\## 5. Files Visible to the Agent



The agent may use:



```text

problem.md

starter.py

metadata.json

public\_tests.py

forbidden\_shortcuts.md

```



These files define the task, the starter implementation, public examples, and forbidden shortcuts.



\---



\## 6. Files Hidden from the Agent



The agent must not read or depend on:



```text

hidden\_tests.py

stress\_tests.py

reference\_solution.py

expected\_trace.json

```



These files are used only by the evaluator.



Reading or hard-coding against these files is considered cheating.



\---



\## 7. Required Submission Format



For a task with ID:



```text

two\_sum\_hash

```



the submission should be placed at:



```text

submissions/two\_sum\_hash/

```



and must contain:



```text

solution.py

trace.json

```



\---



\## 8. `solution.py`



`solution.py` must define a function named:



```python

def solve(...):

&#x20;   ...

```



The function signature must match the task description in `problem.md`.



Example:



```python

def solve(nums, target):

&#x20;   seen = set()

&#x20;   for x in nums:

&#x20;       if target - x in seen:

&#x20;           return True

&#x20;       seen.add(x)

&#x20;   return False

```



\---



\## 9. `trace.json`



`trace.json` is a structured technical explanation of the discovered algorithm.



It is not private chain-of-thought. It should be a concise, inspectable explanation of the algorithmic decision.



Required schema:



```json

{

&#x20; "chosen\_algorithm": "...",

&#x20; "hypotheses": \["...", "..."],

&#x20; "rejected\_algorithms": \[

&#x20;   {

&#x20;     "name": "...",

&#x20;     "reason": "..."

&#x20;   }

&#x20; ],

&#x20; "invariant": "...",

&#x20; "complexity\_time": "...",

&#x20; "complexity\_space": "...",

&#x20; "edge\_cases": \["...", "..."],

&#x20; "counterexample\_for\_wrong\_approach": "optional string"

}

```



Example:



```json

{

&#x20; "chosen\_algorithm": "hash set lookup",

&#x20; "hypotheses": \[

&#x20;   "try all pairs",

&#x20;   "sort and use two pointers",

&#x20;   "use a hash set of previously seen values"

&#x20; ],

&#x20; "rejected\_algorithms": \[

&#x20;   {

&#x20;     "name": "brute force",

&#x20;     "reason": "O(n^2) time violates the expected O(n) complexity"

&#x20;   }

&#x20; ],

&#x20; "invariant": "After processing index i, the set contains exactly the elements before i.",

&#x20; "complexity\_time": "O(n)",

&#x20; "complexity\_space": "O(n)",

&#x20; "edge\_cases": \[

&#x20;   "empty list",

&#x20;   "duplicate values",

&#x20;   "negative numbers"

&#x20; ],

&#x20; "counterexample\_for\_wrong\_approach": ""

}

```



\---



\## 10. What Counts as Algorithm Discovery?



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

| Time-dependent Reachability | BFS over expanded state `(node, time\_mod)` |



A strong submission should explain:



1\. why the chosen algorithm works,

2\. why weaker approaches fail,

3\. what invariant is maintained,

4\. what the time and space complexity are,

5\. which edge cases matter.



\---



\## 11. Scoring



ADC-bench uses the following score:



```text

ADCScore =

&#x20;   0.35 \* correctness

&#x20; + 0.20 \* complexity

&#x20; + 0.15 \* trace\_quality

&#x20; + 0.15 \* transfer

&#x20; + 0.10 \* robustness

&#x20; + 0.05 \* anti\_cheat

```



Each component is normalized to the range:



```text

\[0, 1]

```



\---



\## 12. Correctness



Correctness measures whether `solution.py` passes hidden tests.



```text

correctness = hidden\_tests\_passed / hidden\_tests\_total

```



Public tests are useful for debugging, but they are not enough to establish correctness.



\---



\## 13. Complexity



Complexity measures whether the implementation satisfies the expected asymptotic behavior.



This is tested using:



```text

stress\_tests.py

timeouts

large inputs

scaling-sensitive cases

```



Example:



If a task expects:



```text

O(n)

```



but the submission uses:



```text

O(n^2)

```



then it may pass public tests but fail stress tests.



\---



\## 14. Trace Quality



Trace quality measures whether `trace.json` correctly identifies the algorithmic structure.



The validator checks:



```text

chosen\_algorithm

hypotheses

rejected\_algorithms

invariant

complexity\_time

complexity\_space

edge\_cases

counterexample\_for\_wrong\_approach

```



The current trace validator is heuristic. It is not a formal proof checker.



\---



\## 15. Transfer



Transfer measures whether the agent performs well across related tasks in the same family.



Example:



```text

shortest\_path\_family/

&#x20; dijkstra\_positive\_weights

&#x20; wrong\_hint\_shortest\_path

&#x20; path\_reconstruction\_variant

```



An agent receives a higher transfer score if the discovered strategy generalizes across variants rather than only solving one task by accident.



\---



\## 16. Robustness



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



Example:



```text

The problem says: "This probably needs BFS."

But the graph has positive weights, so the correct method is Dijkstra.

```



\---



\## 17. Anti-Cheat Rules



ADC-bench includes a lightweight local anti-cheat scanner.



Forbidden behavior includes:



```text

reading hidden\_tests.py

reading stress\_tests.py

reading reference\_solution.py

reading expected\_trace.json

modifying evaluator files

hard-coding test outputs

using eval or exec to bypass restrictions

using subprocess to inspect benchmark files

using filesystem introspection to discover hidden files

```



Suspicious strings may include:



```text

hidden\_tests

stress\_tests

reference\_solution

expected\_trace

os.listdir

\_\_file\_\_

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



\---



\## 18. Task Splits



ADC-bench v0.2 contains four task splits:



```text

tasks/

&#x20; known/

&#x20; composed/

&#x20; synthetic/

&#x20; adversarial/

```



\---



\### 18.1 Known Tasks



Canonical algorithmic tasks.



Examples:



```text

two\_sum\_hash

valid\_parentheses\_stack

graph\_reachability\_bfs

dijkstra\_positive\_weights

```



These tasks test whether an agent can recognize and implement standard algorithmic structures.



\---



\### 18.2 Composed Tasks



Tasks requiring composition of known techniques.



Examples:



```text

sliding\_window\_max\_deque

xor\_sat\_f2

two\_sat\_scc

```



These tasks test whether an agent can combine multiple ideas or choose the correct abstraction.



\---



\### 18.3 Synthetic Tasks



Original or less-standard problems designed to reduce pure memorization.



Examples:



```text

parity\_interval\_merge

rotating\_key\_reachability

```



These tasks test whether an agent can analyze a new problem rather than only recall a known template.



\---



\### 18.4 Adversarial Tasks



Tasks designed to expose shortcut behavior.



Example:



```text

wrong\_hint\_shortest\_path

```



These tasks may include misleading hints, hidden complexity traps, or cases where a common algorithmic guess fails.



\---



\## 19. Current Initial Task Set



ADC-bench v0.2 includes 10 initial tasks:



```text

known/two\_sum\_hash

known/valid\_parentheses\_stack

known/graph\_reachability\_bfs

known/dijkstra\_positive\_weights



composed/sliding\_window\_max\_deque

composed/xor\_sat\_f2

composed/two\_sat\_scc



synthetic/parity\_interval\_merge

synthetic/rotating\_key\_reachability



adversarial/wrong\_hint\_shortest\_path

```



\---



\## 20. Installation



Clone the repository:



```bash

git clone https://github.com/chi-binh-ta/adc-bench.git

cd adc-bench

```



Install dependencies:



```bash

pip install -r requirements.txt

```



\---



\## 21. Running the Benchmark



List all tasks:



```bash

python -m adc\_bench.cli list

```



Run the weak baseline agent:



```bash

python examples/run\_baseline.py

```



Run one task:



```bash

python -m adc\_bench.cli run-task --task tasks/known/two\_sum\_hash --submission submissions/two\_sum\_hash

```



Run all tasks:



```bash

python -m adc\_bench.cli run-all --submissions submissions

```



Results are written to:



```text

reports/results.json

reports/summary.md

```



\---



\## 22. Result Format



Each evaluated task produces a result like:



```json

{

&#x20; "task\_id": "known/two\_sum\_hash",

&#x20; "correctness": 1.0,

&#x20; "complexity": 1.0,

&#x20; "trace\_quality": 0.8,

&#x20; "transfer": 0.7,

&#x20; "robustness": 1.0,

&#x20; "anti\_cheat": 1.0,

&#x20; "adc\_score": 0.91,

&#x20; "passed\_public": true,

&#x20; "passed\_hidden": true,

&#x20; "passed\_stress": true,

&#x20; "errors": \[]

}

```



\---



\## 23. Adding a New Task



To add a new task, create a new folder:



```text

tasks/<split>/<task\_id>/

```



Required files:



```text

problem.md

starter.py

reference\_solution.py

public\_tests.py

hidden\_tests.py

stress\_tests.py

expected\_trace.json

metadata.json

forbidden\_shortcuts.md

```



Minimum `metadata.json`:



```json

{

&#x20; "task\_id": "my\_new\_task",

&#x20; "family": "my\_algorithm\_family",

&#x20; "split": "known",

&#x20; "target\_algorithm": "hash set",

&#x20; "expected\_complexity": "O(n)",

&#x20; "difficulty": "easy",

&#x20; "requires\_invariant": true,

&#x20; "requires\_counterexample": false,

&#x20; "anti\_cheat": true

}

```



A good ADC-bench task should have:



1\. a hidden algorithmic structure,

2\. a naive solution that fails stress tests,

3\. hidden tests that check edge cases,

4\. a clear expected invariant,

5\. a meaningful complexity requirement,

6\. at least one way for a shallow shortcut to fail.



\---



\## 24. Adding a Submission



Create a submission folder:



```text

submissions/<task\_id>/

```



Add:



```text

solution.py

trace.json

```



Example:



```text

submissions/two\_sum\_hash/

&#x20; solution.py

&#x20; trace.json

```



Then run:



```bash

python -m adc\_bench.cli run-task --task tasks/known/two\_sum\_hash --submission submissions/two\_sum\_hash

```



\---



\## 25. Baseline Agent



The repository includes a weak baseline agent:



```text

agents/agent\_v0/

&#x20; agent.py

```



This agent mostly copies starter code and emits a weak trace.



It is intentionally limited.



Its purpose is to verify that:



1\. the evaluator works,

2\. weak agents do not score too highly,

3\. hidden and stress tests catch naive solutions.



Run it with:



```bash

python examples/run\_baseline.py

```



\---



\## 26. GitHub Actions Smoke Test



This repository includes a smoke test workflow:



```text

.github/workflows/smoke-test.yml

```



The workflow checks that:



```text

dependencies install

tasks can be listed

the baseline agent can run

```



This is not a full benchmark run. It is a lightweight sanity check for repository health.



\---



\## 27. Design Philosophy



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



\---



\## 28. What ADC-bench Is Not



ADC-bench is not:



```text

a secure sandbox

a full formal verification system

a replacement for SWE-bench

a replacement for HumanEval

a large-scale production benchmark yet

```



ADC-bench is currently a research prototype for studying algorithm discovery behavior in AI coding agents.



\---



\## 29. Limitations



ADC-bench v0.2 has several known limitations:



1\. The local sandbox is not secure against malicious code.

2\. Trace scoring is heuristic.

3\. Transfer scoring is currently approximate.

4\. The task set is small.

5\. Some tasks may still resemble known competitive-programming patterns.

6\. The benchmark does not yet fully evaluate self-modifying agents.

7\. The anti-cheat system catches obvious violations, not sophisticated attacks.



\---



\## 30. Roadmap



Planned improvements:



```text

v0.2.1 — Better verifier module and cleaner reports

v0.3   — Self-improving agent layer

v0.4   — Stronger synthetic task generation

v0.5   — Human-verified task subset

v1.0   — Stable benchmark release

```



\---



\## 31. Future Direction: Self-Improving Agent Layer



A future version of ADC-bench may include tasks where an agent receives:



```text

agent\_v0 source code

failure logs

held-out benchmark tasks

```



The agent must produce:



```text

agent\_v1

```



The evaluator then checks whether:



```text

Score(agent\_v1) > Score(agent\_v0)

```



on held-out tasks.



This would move ADC-bench from algorithm discovery toward recursive algorithm improvement.



\---



\## 32. Summary



ADC-bench evaluates five levels of capability:



```text

Level 1 — Code correctness

Level 2 — Complexity awareness

Level 3 — Algorithmic structure discovery

Level 4 — Transfer across problem families

Level 5 — Robustness and future self-improvement

```



The goal is not to reward code that merely passes visible tests.



The goal is to evaluate whether an AI agent can:



```text

discover

repair

transfer

certify

and robustly apply

```



algorithmic structure.

