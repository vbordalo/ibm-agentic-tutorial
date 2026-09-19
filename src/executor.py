import io
import traceback
import contextlib


def executor_agent(state):
    code = state.get("code", "")
    attempts = state.get("attempts", 0) + 1

    if not code:
        return {
            "exec_output": "",
            "exec_error": "No code was provided by the coder.",
            "attempts": attempts,
        }

    exec_namespace = {
        "__name__": "__main__",
        "df": state["df"],
    }

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
        return {
            "exec_output": "",
            "exec_error": traceback.format_exc(),
            "attempts": attempts,
        }