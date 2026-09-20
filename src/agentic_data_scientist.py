from typing import Literal
from pathlib import Path
import gradio as gr

# LangGraph
from langgraph.graph import END, StateGraph, START

# LangChain
from langchain_ollama import ChatOllama

from coder import coder_agent
from executor import executor_agent
from input_agent import ui_input_agent
from planner import planner_agent
from reviewer import reviewer_agent
from state import GraphState
from run_trace import RunTrace

def run_workflow(task: str, uploaded_file) -> dict:

    trace = RunTrace(
        model=MODEL,
        temperature=0.0,
        task=task,
        dataset_name=Path(str(uploaded_file)).name if uploaded_file else None,
    )

    final_state = app.invoke({"task": task, "uploaded_file": uploaded_file})

    if final_state.get("exec_error"):
        if final_state.get("suggestions"):
            run_status = "needs_human"
        else:
            run_status = "failed"
    else:
        run_status = "success"

    trace.finish(
        status=run_status,
        final_state={
            "instructions": final_state.get("instructions", ""),
            "code": final_state.get("code", ""),
            "exec_output": final_state.get("exec_output", ""),
            "exec_error": final_state.get("exec_error", ""),
            "attempts": final_state.get("attempts", 0),
            "suggestions": final_state.get("suggestions", ""),
        },
    )

    trace.finish(
        status=run_status,
        final_state={
            "instructions": final_state.get("instructions", ""),
            "code": final_state.get("code", ""),
            "exec_output": final_state.get("exec_output", ""),
            "exec_error": final_state.get("exec_error", ""),
            "attempts": final_state.get("attempts", 0),
            "suggestions": final_state.get("suggestions", ""),
        },
    )

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


# Workflow definition

MAX_ATTEMPTS = 5
workflow = StateGraph(GraphState)

workflow.add_node("input", ui_input_agent)
workflow.add_node("planner", lambda state: planner_agent(state, llm))
workflow.add_node("coder", lambda state: coder_agent(state, llm))
workflow.add_node("executor", executor_agent)
workflow.add_node("reviewer", lambda state: reviewer_agent(state, llm))

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