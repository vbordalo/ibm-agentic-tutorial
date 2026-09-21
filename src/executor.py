import contextlib
import io
import os
import traceback
from pathlib import Path


def executor_agent(state):
    code = state.get("code", "")
    attempts = state.get("attempts", 0) + 1

    if not code:
        return {
            "exec_output": "",
            "exec_error": "No code was provided by the coder.",
            "attempts": attempts,
        }

    workspace = Path(state["workspace_path"]).resolve()

    if not workspace.is_dir():
        return {
            "exec_output": "",
            "exec_error": f"Workspace not found: {workspace}",
            "attempts": attempts,
        }

    exec_namespace = {
        "__name__": "__main__",
    }

    stdout_buf = io.StringIO()
    previous_cwd = Path.cwd()

    try:
        os.chdir(workspace)

        with contextlib.redirect_stdout(stdout_buf):
            exec(code, exec_namespace)

        return {
            "exec_output": stdout_buf.getvalue(),
            "exec_error": "",
            "attempts": attempts,
        }

    except Exception:
        return {
            "exec_output": stdout_buf.getvalue(),
            "exec_error": traceback.format_exc(),
            "attempts": attempts,
        }

    finally:
        os.chdir(previous_cwd)