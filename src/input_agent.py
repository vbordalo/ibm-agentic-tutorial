from pathlib import Path


def ui_input_agent(state):
    workspace = Path(state["workspace_path"])

    if not workspace.is_dir():
        raise ValueError(
            f"Workspace does not exist or is not a directory: {workspace}"
        )

    files = sorted(
        p.name
        for p in workspace.iterdir()
        if p.is_file()
    )

    dataset_info = (
        "The task is running in a workspace containing these files:\n"
        + "\n".join(f"- {name}" for name in files)
    )

    return {
        "task": state["task"],
        "task_id": state.get("task_id", ""),
        "workspace_path": str(workspace),
        "workspace_files": files,
        "dataset_info": dataset_info,
        "attempts": 0,
        "suggestions": "",
    }