import shutil
from pathlib import Path

# LangChain
from langchain_ollama import ChatOllama

from run_trace import RunTrace
from workflow import build_workflow

# Environment and LLM setup

MODEL = "qwen2.5:3b-instruct-q4_K_M"
TEMPERATURE = 0.0
SEED = 42
llm = ChatOllama(
    model=MODEL,
    temperature=TEMPERATURE,
    seed=SEED,
)


def prepare_run_workspace(
    run_dir: Path,
    uploaded_files=None,
    source_workspace=None,
) -> Path:
    workspace = run_dir / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)

    if uploaded_files and source_workspace:
        raise ValueError(
            "Provide either uploaded_files or workspace_path, not both."
        )

    if uploaded_files:
        for file_obj in uploaded_files:
            source = Path(str(file_obj))

            if not source.is_file():
                raise FileNotFoundError(
                    f"Uploaded file not found: {source}"
                )

            shutil.copy2(
                source,
                workspace / source.name,
            )

    elif source_workspace:
        source_workspace = Path(source_workspace).expanduser().resolve()

        if not source_workspace.is_dir():
            raise FileNotFoundError(
                f"Workspace not found: {source_workspace}"
            )

        for source in source_workspace.iterdir():
            destination = workspace / source.name

            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)

    return workspace

def run_workflow(
    task: str,
    uploaded_files=None,
    workspace_path=None,
    task_id=None,
) -> dict:

    if uploaded_files:
        dataset_name = ", ".join(
            Path(str(file)).name
            for file in uploaded_files
        )
    elif workspace_path:
        dataset_name = Path(workspace_path).name
    else:
        dataset_name = None

    trace = RunTrace(
        model=MODEL,
        temperature=TEMPERATURE,
        task=task,
        dataset_name=dataset_name,
    )

    run_workspace = prepare_run_workspace(
        run_dir=trace.run_dir,
        uploaded_files=uploaded_files,
        source_workspace=workspace_path,
    )

    app = build_workflow(trace, llm)

    final_state = app.invoke({
        "task": task,
        "workspace_path": str(run_workspace),
        "task_id": task_id,
    })

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
                f"💡  Suggested next step for the human:\n"
                f"{final_state['suggestions']}"
            )
        else:
            result["final_output"] = (
                "❗️  Execution failed after the maximum number of attempts.\n"
                f"Error was:\n{final_state['exec_error']}"
            )
    else:
        result["final_output"] = (
            "✅  Code ran successfully!  Output:\n"
            + final_state["exec_output"]
        )

    return result
