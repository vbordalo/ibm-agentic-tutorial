from typing import Any

import pandas as pd


def _normalise_path(file_obj: Any) -> str:
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
            raise ValueError(
                "Unsupported file type. Please upload CSV or Excel."
            )

    except Exception as exc:
        raise ValueError(
            f"Failed to read the uploaded file: {exc}"
        ) from exc

    return df.drop(columns="Date", errors="ignore")


def ui_input_agent(state):
    df = load_file(state.get("uploaded_file"))

    if isinstance(df, pd.DataFrame) and not df.empty:
        cols = ", ".join(df.columns.tolist())

        dataset_info = (
            f"The uploaded file contains the following columns: {cols}."
        )

    else:
        dataset_info = "No data was uploaded (empty DataFrame)."

    return {
        "task": state["task"],
        "uploaded_file": state.get("uploaded_file"),
        "dataset_info": dataset_info,
        "attempts": 0,
        "suggestions": "",
        "df": df,
    }