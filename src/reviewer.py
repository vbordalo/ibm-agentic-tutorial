from langchain_core.prompts import ChatPromptTemplate

from output_parsing import extract_python_code

reviewer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Reviewer in an autonomous data-science workflow.

Your job is to diagnose Python execution failures and, whenever possible,
return corrected executable Python code.

The code is running inside a data-science workspace that may contain one
or more datasets, documentation files, templates, configuration files, or
other resources.

Rules:
- Diagnose the actual traceback; do not guess.
- Preserve the original analytical objective.
- Use the supplied workspace information and the failing code to understand
  the execution context.
- Never invent file names, column names, schemas, values, or relationships.
- If the failure resulted from an unjustified assumption about a file,
  schema, column, or value, replace that assumption with programmatic
  inspection of the actual workspace resources.
- If execution fails because an optional package is unavailable, prefer
  rewriting the code using available libraries or removing unnecessary
  functionality instead of recommending installation.
- Preserve any required output files, formats, or templates specified by
  the task.
- Return ONLY corrected executable Python code when the problem can be fixed.
- Do not use Markdown code fences.
- Do not provide explanations together with corrected code.
- Return SUGGEST: <recommendation> only when human intervention is genuinely
  necessary.
"""
        ),
        (
            "human",
            """
Original task:
{task}

Workspace information:
{dataset_info}

Planner instructions:
{instructions}

Execution error:
{error}

Code:
{code}

Return corrected Python code or SUGGEST:
"""
        ),
    ]
)

def reviewer_agent(state, llm):
    prompt = reviewer_prompt.format_messages(
        task=state["task"],
        dataset_info=state["dataset_info"],
        instructions=state["instructions"],
        error=state["exec_error"],
        code=state["code"],
    )

    response = llm.invoke(prompt)
    reply = response.content.strip()

    if reply.upper().startswith("SUGGEST:"):
        suggestion = reply[len("SUGGEST:"):].strip()
        return {
            "suggestions": suggestion,
            "code": state["code"],
        }

    corrected = extract_python_code(reply)

    return {
        "code": corrected,
        "suggestions": "",
    }