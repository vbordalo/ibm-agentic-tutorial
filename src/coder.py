from langchain_core.prompts import ChatPromptTemplate

coder_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Coder in an autonomous data-science workflow.

A pandas DataFrame named `df` is already loaded and available in the
execution environment. Never reload the dataset and never invent file paths.

Write Python code that follows the Planner's instructions.

Rules:
- Return ONLY executable Python code.
- Your entire response must be valid Python source code.
- Do not use Markdown code fences.
- Do not provide explanations, headings, or numbered steps.
- Do not use pip, shell commands, or package installation commands.
- Use the existing `df` DataFrame directly.
- Never invent, assume, or infer column names.
- Use only columns explicitly present in the dataset information.
- If the task requires discovering useful predictors, inspect the actual
  DataFrame programmatically instead of assuming which variables matter.
- Print the results required by the task.
"""
        ),
        (
            "human",
            "Dataset information:\n{dataset_info}\n\n"
            "Planner instructions:\n{instructions}\n\n"
            "Write the Python code:"
        ),
    ]
)


def coder_agent(state, llm):
    prompt = coder_prompt.format_messages(
        instructions=state["instructions"],
        dataset_info=state["dataset_info"],
    )

    response = llm.invoke(prompt)
    code = response.content.strip()

    # Defensive cleanup in case the model still returns Markdown fences.
    if code.startswith("```python"):
        code = code[len("```python"):].strip()
    if code.startswith("```"):
        code = code[3:].strip()
    if code.endswith("```"):
        code = code[:-3].strip()

    return {"code": code}