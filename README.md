# IBM Agentic Data Scientist Tutorial

This repository is a local implementation and adaptation of the IBM Developer tutorial on building an **agentic data scientist**.

The project is being used to study how a data-science workflow can be decomposed into specialized agents, orchestrated with LangGraph, executed locally with an Ollama model, and instrumented for reproducibility and observability.

The current stack includes:

- **LangGraph** for workflow orchestration
- **LangChain** for LLM prompting and abstractions
- **Ollama** for running a local language model
- **Gradio** for the user interface
- **pandas** and the scientific Python ecosystem for data analysis
- a custom **run tracing layer** for execution history and metrics

The original IBM tutorial uses IBM watsonx. In this repository, the architecture is being tested with a local Ollama model instead.

## Current Architecture

```text
User / Gradio UI
       ↓
Input Agent
       ↓
Planner
       ↓
Coder
       ↓
Executor
   │
   ├── success → END
   │
   └── error → Reviewer
                  │
                  ├── corrected code → Executor
                  └── human suggestion → END
```

The agents currently have the following responsibilities:

- **Input Agent**: loads the uploaded dataset into a pandas DataFrame and builds basic dataset metadata.
- **Planner**: creates a concise analysis plan from the user task.
- **Coder**: generates executable Python code that operates directly on the existing DataFrame.
- **Executor**: executes the generated Python code, captures standard output, and records exceptions.
- **Reviewer**: diagnoses execution failures and either returns corrected code or requests human intervention.

The workflow includes a maximum number of execution attempts to prevent infinite retry loops.

## Application Structure

```text
src/
├── main.py            # Gradio UI / application entry point
├── run_service.py     # run lifecycle, model setup, trace lifecycle
├── workflow.py        # LangGraph construction, routing, node instrumentation
├── state.py           # shared GraphState definition
├── run_trace.py       # observability, event logging, run summaries
│
├── input_agent.py     # dataset loading and dataset metadata
├── planner.py         # planning agent
├── coder.py           # code-generation agent
├── executor.py        # Python execution
└── reviewer.py        # execution-error review and recovery
```

This separation keeps the UI independent from the workflow definition and makes it easier to reuse the agentic workflow later from other interfaces such as a CLI, benchmark harness, or automated test suite.

## Run Lifecycle

`run_service.py` coordinates a complete execution:

```text
UI request
   ↓
run_workflow(...)
   ↓
create RunTrace
   ↓
build_workflow(trace, llm)
   ↓
invoke LangGraph
   ↓
determine run status
   ↓
persist trace + summary
   ↓
return formatted result to the UI
```

The current run status values are:

- `success`: execution finished without an unresolved Python error
- `failed`: execution ended with an unresolved error
- `needs_human`: the Reviewer determined that human intervention is required

A `success` status currently means **execution success**, not necessarily analytical or scientific correctness.

## Observability and Run Tracing

Each run creates its own trace directory under:

```text
runs/YYYY-MM-DD/<run_id>/
```

A run currently produces:

```text
events.jsonl
summary.json
```

### `events.jsonl`

The event log records the execution trajectory in sequence. Typical events include:

```text
run_started
node_started
node_completed
execution_succeeded
execution_failed
reviewer_corrected_code
reviewer_suggested_human
run_completed
```

The trace captures operational information such as task, dataset name, model configuration, Git commit, planner instructions, generated code, execution output or traceback, reviewer result, and attempt number.

The trace is intentionally focused on observable workflow state and outputs rather than hidden model reasoning.

### `summary.json`

The summary stores metadata, the final run status, metrics, and the final state.

Current counters include:

```text
planner_calls
coder_calls
executor_calls
reviewer_calls
execution_errors
```

This makes it possible to compare runs and later use the same instrumentation for benchmarks and evaluation.

## Local Model

The current model is:

```text
qwen2.5:3b-instruct-q4_K_M
```

It is served locally through Ollama.

The model configuration currently lives in `src/run_service.py`:

```python
MODEL = "qwen2.5:3b-instruct-q4_K_M"
TEMPERATURE = 0.0
```

The same configuration is recorded in the run trace for reproducibility.

## Environment Setup

Create a dedicated Conda environment:

```bash
conda create -n ibm-agentic-tutorial python=3.12 -y
conda activate ibm-agentic-tutorial
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Make sure Ollama is installed and the configured model is available locally:

```bash
ollama list
```

## Running the Application

```bash
conda activate ibm-agentic-tutorial
python src/main.py
```

Gradio should start a local server, typically:

```text
http://127.0.0.1:7860
```

Open this address in a browser.

## Example Tasks

A simple inspection task:

```text
How many rows and columns are in the dataset?
```

A more complex modeling task:

```text
Train a linear regression model to predict ViolentCrimesPerPop using all suitable
predictor columns. Handle missing values appropriately, encode categorical
variables if needed, split the data into training and test sets, and report the
RMSE and R2 on the test set.
```

The system will plan the requested analysis, generate Python code, execute it, route execution failures to the Reviewer, retry corrected code when possible, stop with a human suggestion when automatic recovery is not appropriate, and persist an execution trace and summary.

## Current Limitations

The system currently evaluates execution much more strongly than analytical correctness.

A Python program may execute successfully while still being scientifically weak, incomplete, inefficient, or inconsistent with the user objective.

Important limitations under investigation include:

- detecting non-standard missing-value sentinels such as `?`
- verifying whether generated code fully satisfies the user objective
- distinguishing execution correctness from analytical correctness
- improving Reviewer diagnosis of execution errors
- preserving richer lineage across original and corrected code
- controlling excessively large execution outputs
- recording explicit routing decisions
- evaluating efficiency as well as correctness

The current Reviewer is primarily an **execution-error debugger**, not yet a semantic verifier.

## Planned Architecture Evolution

A likely next extension is to introduce a semantic verification stage after successful execution:

```text
Executor
   │
   ├── error → Reviewer / Debugger → Executor
   │
   └── success → Verifier
                    │
                    ├── sufficient → END
                    └── insufficient → Planner / Coder
```

This would separate two different questions:

```text
Did the code run?
```

from:

```text
Did the analysis actually answer the task correctly and sufficiently?
```

Future work may also include explicit route tracing, artifact handling, benchmark integration, and more rigorous evaluation of autonomous data-science behavior.

## Project Purpose

This repository is primarily an educational and experimental project for studying agentic AI architecture in data science.

It is also being used to compare two broad approaches:

```text
Predefined semantic tools
vs.
Dynamic Python code generation and execution
```

The broader goal is to explore how autonomous agents can make effective use of scientific Python libraries such as pandas, NumPy, SciPy, and scikit-learn without requiring every analytical operation to be manually exposed as a dedicated tool.

The project is intentionally evolving incrementally so that architecture, observability, error recovery, verification, and evaluation can be understood and tested independently.

## Reference

Based on the IBM Developer tutorial:

**Agentic Data Scientists**

https://developer.ibm.com/articles/agentic-data-scientists/

## Status

Work in progress.
