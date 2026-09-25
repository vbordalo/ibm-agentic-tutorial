from langchain_core.prompts import ChatPromptTemplate

from output_parsing import extract_python_code

coder_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Coder in an autonomous data-science workflow.

You are working inside a data-science workspace that may contain one or
more datasets, documentation files, templates, configuration files, or
other resources.

Write Python code that follows the Planner's instructions and completes
the user's task.

Rules:
- Return ONLY executable Python code.
- Your entire response must be valid Python source code.
- Do not use Markdown code fences.
- Do not provide explanations, headings, or numbered steps.
- Do not use pip, shell commands, or package installation commands.
- Use the available workspace resources as needed.
- Inspect file contents, schemas, and data programmatically when they are
  not already known.
- Never invent file names, column names, schemas, values, or relationships.
- Do not assume the role or contents of a file solely from its name.
- If the task requires discovering useful predictors, inspect the actual
  data programmatically instead of assuming which variables matter.
- Create or modify output files when explicitly required by the task.
- Preserve any required output format or template.
- Print relevant results so that execution can be inspected.
"""
        ),
        (
            "human",
            "Workspace information:\n{dataset_info}\n\n"
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
    code = extract_python_code(response.content)

    return {"code": code}