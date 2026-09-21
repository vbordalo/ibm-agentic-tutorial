from langchain_core.prompts import ChatPromptTemplate

planner_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Planner in an autonomous data-science workflow.

You are working inside a data-science workspace that may contain one or
more datasets, documentation files, templates, configuration files, or
other resources.

Create a concise numbered plan that is sufficient to answer the user's task.

Rules:
- Prefer the simplest plan that fully answers the task.
- Do not add analyses that were not requested.
- Use the workspace information provided to identify the available resources.
- Plan inspection of files, schemas, documentation, or other resources when
  their contents are needed to complete the task.
- Do not assume the role or contents of a file solely from its name.
- Do not write Python code.
- Never invent, assume, or infer column names, schemas, file contents, or
  relationships that have not been inspected.
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
            "Task description:\n{task}\n\n"
            "Workspace information:\n{dataset_info}\n\n"
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