from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    task: str
    task_id: str
    uploaded_files: list[Any]
    workspace_path: str
    workspace_files: list[str]
    dataset_info: str
    instructions: str
    code: str
    exec_output: str
    exec_error: str
    attempts: int
    suggestions: str
