from pathlib import Path

# LangChain
from langchain_ollama import ChatOllama

from run_trace import RunTrace
from workflow import build_workflow

# Environment and LLM setup

MODEL = "qwen2.5:3b-instruct-q4_K_M"
TEMPERATURE = 0.0

llm = ChatOllama(
    model=MODEL,
    temperature=TEMPERATURE,
)

def run_workflow(task: str, uploaded_file) -> dict:

    trace = RunTrace(
        model=MODEL,
        temperature=TEMPERATURE,
        task=task,
        dataset_name=Path(str(uploaded_file)).name if uploaded_file else None,
    )

    app = build_workflow(trace, llm)
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
