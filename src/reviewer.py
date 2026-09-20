from langchain_core.prompts import ChatPromptTemplate

reviewer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are the Reviewer in an autonomous data-science workflow.

Your job is to diagnose Python execution failures and, whenever possible,
return corrected executable Python code.

A pandas DataFrame named `df` is already loaded and available in the
execution environment.

Rules:
- Diagnose the actual traceback; do not guess.
- Preserve the original analytical objective.
- Never invent column names.
- Use the supplied dataset information to correct invalid column references.
- If the failure resulted from an unjustified analytical assumption,
  replace that assumption with programmatic inspection of the actual data.
- If execution fails because an optional package is unavailable, prefer
  rewriting the code using available libraries or removing unnecessary
  functionality instead of recommending installation.
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

Dataset information:
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

    corrected = reply

    # Defensive cleanup in case the model ignores the output contract.
    if corrected.startswith("```python"):
        corrected = corrected[len("```python"):].strip()

    if corrected.startswith("```"):
        corrected = corrected[3:].strip()

    if corrected.endswith("```"):
        corrected = corrected[:-3].strip()

    return {
        "code": corrected,
        "suggestions": "",
    }