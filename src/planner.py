from langchain_core.prompts import ChatPromptTemplate

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
- Never invent, assume, or infer column names.
- Refer only to columns explicitly listed in the dataset information.
- For exploratory feature selection, do not preselect predictors based only
  on their names or assumed meaning.
- Plan an empirical investigation of the available features before deciding
  which features are relevant.
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


def planner_agent(state, llm):
    prompt = planner_prompt.format_messages(
        task=state["task"],
        dataset_info=state["dataset_info"],
    )

    response = llm.invoke(prompt)
    instructions = response.content.strip()

    return {"instructions": instructions}