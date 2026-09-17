# IBM Agentic Data Scientist Tutorial

This repository is a local implementation and adaptation of the IBM Developer tutorial on building an **agentic data scientist**.

The goal of this project is to study and reproduce an agentic data-science workflow based on:

* LangGraph for workflow orchestration
* LangChain for LLM prompting and abstractions
* Ollama for running a local language model
* Gradio for the user interface
* pandas for dataset manipulation and analysis

The original IBM tutorial uses IBM watsonx. In this repository, the architecture is being tested with a local Ollama model instead.

## Current Architecture

The current workflow is:

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

* **Input Agent**: loads the uploaded dataset into a pandas DataFrame.
* **Planner**: creates a concise analysis plan from the user task.
* **Coder**: generates Python code that operates on the existing DataFrame.
* **Executor**: executes the generated Python code and captures output or exceptions.
* **Reviewer**: attempts to correct code when execution fails.

The workflow includes a maximum number of execution attempts to prevent infinite retry loops.

## Local Model

The current model is:

```text
qwen2.5:3b-instruct-q4_K_M
```

It is served locally through Ollama.

The model can be changed in:

```python
MODEL = "qwen2.5:3b-instruct-q4_K_M"
```

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

Activate the environment:

```bash
conda activate ibm-agentic-tutorial
```

Then run:

```bash
python src/agentic_data_scientist.py
```

Gradio should start a local server, typically:

```text
http://127.0.0.1:7860
```

Open this address in a browser.

## Example Task

Upload a CSV or Excel file and enter a task such as:

```text
Inspect the dataset and report the number of rows, columns, and missing values.
```

The system will:

1. Plan the analysis.
2. Generate Python code.
3. Execute the code.
4. Attempt to review and correct execution errors.
5. Return the final output.

## Current Limitations

The current system mainly evaluates whether generated code executes successfully.

A successful execution does not necessarily mean that the analysis is scientifically or semantically correct.

Examples of issues still under investigation include:

* detecting non-standard missing-value sentinels such as `?`
* verifying whether generated code fully follows the Planner instructions
* distinguishing execution correctness from analytical correctness
* improving Reviewer diagnosis of execution errors
* preserving both original and corrected code during retry cycles

A future extension may introduce an analytical critic or quality-review agent.

## Project Purpose

This repository is primarily an educational project for studying agentic AI architecture in data science.

It is also being used to compare two approaches:

```text
Predefined semantic tools
vs.
Dynamic Python code generation and execution
```

The broader goal is to explore how autonomous agents can make effective use of scientific Python libraries such as pandas, NumPy, SciPy, and scikit-learn without requiring every analytical operation to be manually exposed as a dedicated tool.

## Reference

Based on the IBM Developer tutorial:

**Agentic Data Scientists**

https://developer.ibm.com/articles/agentic-data-scientists/

## Status

Work in progress.
