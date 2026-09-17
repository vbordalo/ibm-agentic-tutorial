import io
import traceback
import contextlib
from typing import TypedDict, Literal, Any

import pandas as pd
import gradio as gr
from dotenv import load_dotenv

# LangGraph
from langgraph.graph import END, StateGraph, START

# LangChain
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

# Helper utilities


def _normalise_path(file_obj) -> str:
    if file_obj is None:
        return ""
    if isinstance(file_obj, str):
        return file_obj
    return getattr(file_obj, "name", "")


def load_file(file_obj):
    path = _normalise_path(file_obj)
    if not path:
        return pd.DataFrame()
    ext = path.lower().split(".")[-1]
    try:
        if ext == "csv":
            df = pd.read_csv(path)
        elif ext in ("xlsx", "xls"):
            df = pd.read_excel(path)
        else:
            raise ValueError("Unsupported file type. Please upload CSV or Excel.")
    except Exception as exc:
        raise ValueError(f"Failed to read the uploaded file: {exc}") from exc
    return df.drop(columns="Date", errors="ignore")

def run_workflow(task: str, uploaded_file) -> dict:
    final_state = app.invoke({"task": task, "uploaded_file": uploaded_file})
    result = {
        "planner": final_state.get("instructions", ""),
        "coder": final_state.get("code", ""),
        "executor_output": final_state.get("exec_output", ""),
        "executor_error": final_state.get("exec_error", ""),
        "reviewer": final_state.get("suggestions", ""),
        "final_output": "",
    }
    if final_state.get("exec_error"):
        if final_state.get("suggestions"):
            result["final_output"] = (
                "❗️  The system could not automatically fix the code.\n"
                f"💡  Suggested next step for the human:\n{final_state['suggestions']}"
            )
        else:
            result["final_output"] = (
                "❗️  Execution failed after the maximum number of attempts.\n"
                f"Error was:\n{final_state['exec_error']}"
            )
    else:
        result["final_output"] = "✅  Code ran successfully!  Output:\n" + final_state["exec_output"]
    return result


# Environment and LLM setup

MODEL = "qwen2.5:3b-instruct-q4_K_M"
llm = ChatOllama(
    model=MODEL,
    temperature=0.0,
)


# GraphState definition

class GraphState(TypedDict, total=False):
    task: str
    uploaded_file: Any
    dataset_info: str
    instructions: str
    code: str
    exec_output: str
    exec_error: str
    attempts: int
    suggestions: str

# Input Agent

def ui_input_agent(state: GraphState) -> GraphState:
    global df
    df = load_file(state.get("uploaded_file"))
    if isinstance(df, pd.DataFrame) and not df.empty:
        cols = ", ".join(df.columns.tolist())
        dataset_info = f"The uploaded file contains the following columns: {cols}."
    else:
        dataset_info = "No data was uploaded (empty DataFrame)."
    return {
        "task": state["task"],
        "uploaded_file": state.get("uploaded_file"),
        "dataset_info": dataset_info,
        "attempts": 0,
        "suggestions": "",
    }

# Planner Agent

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Planner in an autonomous data-science workflow.

A pandas DataFrame named `df` is already loaded and available to the workflow.
Do not include steps for loading, opening, or locating the dataset.

Create a concise numbered plan that is sufficient to answer the user's task.

Rules:
- Prefer the simplest plan that fully answers the task.
- Do not add analyses that were not requested.
- Use the dataset information provided to understand the available columns.
- Focus on what the Coder must do with the existing `df`.
- Do not write Python code.
- When assessing data completeness, do not rely only on pandas null detection.
  Consider whether unusual sentinel values or strings such as "?", "NA",
  "N/A", "null", empty strings, or similar values may represent missing data.
- Treat such values as suspicious candidates, not automatically as missing,
  and inspect their occurrence before drawing conclusions about completeness.
"""
        ),
        (
            "human",
            "Task description: {task}\n"
            "Dataset info: {dataset_info}\n\n"
            "Provide the numbered instructions."
        ),
    ]
)


def planner_agent(state: GraphState) -> GraphState:
    prompt = planner_prompt.format_messages(
        task=state["task"],
        dataset_info=state["dataset_info"],
    )
    response = llm.invoke(prompt)
    instructions = response.content.strip()
    return {"instructions": instructions}


# Coder Agent

coder_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are the Coder in an autonomous data-science workflow.

A pandas DataFrame named `df` is already loaded and available in the
execution environment. Never reload the dataset and never invent file paths.

Write Python code that follows the Planner's instructions.

Rules:
- Return ONLY executable Python code.
- Do not use Markdown code fences.
- Do not provide explanations.
- Do not use pip, shell commands, or package installation commands.
- Use the existing `df` DataFrame directly.
- Print the results required by the task.
"""
        ),
        (
            "human",
            "Instructions:\n{instructions}\n\nWrite the Python code:"
        ),
    ]
)

def coder_agent(state: GraphState) -> GraphState:
    prompt = coder_prompt.format_messages(instructions=state["instructions"])
    response = llm.invoke(prompt)
    code = response.content.strip()
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()
    return {"code": code}

# Executor Agent

def executor_agent(state: GraphState) -> GraphState:
    code = state.get("code", "")
    attempts = state.get("attempts", 0) + 1

    if not code:
        return {
            "exec_output": "",
            "exec_error": "No code was provided by the coder.",
            "attempts": attempts,
        }

    exec_namespace = dict(globals())
    exec_namespace.update({"__name__": "__main__"})
    stdout_buf = io.StringIO()

    try:
        with contextlib.redirect_stdout(stdout_buf):
            exec(code, exec_namespace)

        return {
            "exec_output": stdout_buf.getvalue(),
            "exec_error": "",
            "attempts": attempts,
        }

    except Exception:
        tb = traceback.format_exc()

        return {
            "exec_output": "",
            "exec_error": tb,
            "attempts": attempts,
        }

# Reviewer Agent

reviewer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Reviewer in an autonomous data-science workflow.

Your job is to inspect Python execution errors and correct the code.

A pandas DataFrame named `df` is already loaded and available in the
execution environment.

Rules:
- If you can fix the error, return ONLY the corrected executable Python code.
- Do not use Markdown code fences.
- Do not provide explanations together with corrected code.
- Never reload the dataset.
- Never invent file paths.
- Do not use pip, shell commands, or installation commands.
- Preserve the original analytical intent.
- If the error cannot reasonably be fixed automatically, respond exactly
  with:

SUGGEST: <brief recommendation for the human>
"""
        ),
        (
            "human",
            "Execution error:\n{error}\n\n"
            "Original code:\n{code}\n\n"
            "Return corrected code or SUGGEST:"
        ),
    ]
)

def reviewer_agent(state: GraphState) -> GraphState:
    prompt = reviewer_prompt.format_messages(error=state["exec_error"], code=state["code"])
    response = llm.invoke(prompt)
    reply = response.content.strip()
    if reply.upper().startswith("SUGGEST:"):
        suggestion = reply[len("SUGGEST:"):].strip()
        return {"suggestions": suggestion, "code": state["code"]}
    corrected = reply
    if corrected.startswith("```python"):
        corrected = corrected[len("```python"):].strip()
    if corrected.startswith("```"):
        corrected = corrected[3:].strip()
    if corrected.endswith("```"):
        corrected = corrected[:-3].strip()
    return {"code": corrected, "suggestions": ""}

# Workflow definition

MAX_ATTEMPTS = 5
workflow = StateGraph(GraphState)

workflow.add_node("input", ui_input_agent)
workflow.add_node("planner", planner_agent)
workflow.add_node("coder", coder_agent)
workflow.add_node("executor", executor_agent)
workflow.add_node("reviewer", reviewer_agent)

workflow.add_edge(START, "input")
workflow.add_edge("input", "planner")
workflow.add_edge("planner", "coder")
workflow.add_edge("coder", "executor")

def after_executor(state: GraphState) -> Literal["reviewer", END]:
    if state["exec_error"] and state["attempts"] < MAX_ATTEMPTS:
        return "reviewer"
    return END

def after_reviewer(state: GraphState) -> Literal["executor", END]:
    if state.get("suggestions"):
        return END
    return "executor"

workflow.add_conditional_edges("executor", after_executor, {"reviewer": "reviewer", END: END})
workflow.add_conditional_edges("reviewer", after_reviewer, {"executor": "executor", END: END})

app = workflow.compile()

# Gradio user interface

with gr.Blocks() as demo:
    gr.Markdown("""
        # 🤖 Data-Science Assistant (LangGraph + local model)
        1️⃣ Upload a CSV or Excel file.  
        2️⃣ Describe the analysis / model you want in plain English.  
        3️⃣ Press Run - the assistant will plan, code, execute, review and finally give you the result.
    """)

    with gr.Row():
        file_input = gr.File(label="📂 Upload CSV / Excel (optional)", file_types=[".csv", ".xlsx", ".xls"])
        task_input = gr.Textbox(label="📝 Task description", placeholder="e.g. train a linear regression model", lines=3)

    run_btn = gr.Button("🚀 Run", variant="primary")

    planner_box = gr.Textbox(label="🗒️ Planner - numbered steps", lines=6)
    coder_box   = gr.Code(label="👩‍💻 Coder - generated Python code", language="python", lines=12)
    executor_out = gr.Textbox(label="⚙️ Executor - stdout", lines=6)
    executor_err = gr.Textbox(label="❌ Executor - error (if any)", lines=6)
    reviewer_box = gr.Textbox(label="🧐 Reviewer - suggestion (if any)", lines=6)
    final_box    = gr.Textbox(label="🎉 Final output", lines=8)

    def on_click(task, uploaded_file):
        out = run_workflow(task, uploaded_file)
        return (
            out["planner"], out["coder"], out["executor_output"],
            out["executor_error"], out["reviewer"], out["final_output"],
        )

    run_btn.click(fn=on_click, inputs=[task_input, file_input], outputs=[planner_box, coder_box, executor_out, executor_err, reviewer_box, final_box])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Default())