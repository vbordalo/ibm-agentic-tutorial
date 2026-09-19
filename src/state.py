from typing import TypedDict, Any


class GraphState(TypedDict, total=False):
    task: str
    uploaded_file: Any
    df: Any
    dataset_info: str
    instructions: str
    code: str
    exec_output: str
    exec_error: str
    attempts: int
    suggestions: str
