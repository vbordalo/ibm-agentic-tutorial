


def extract_python_code(text: str) -> str:
    text = text.strip()

    if "```" not in text:
        return text

    parts = text.split("```")

    # Prefer fenced blocks explicitly marked as Python.
    for i in range(1, len(parts), 2):
        block = parts[i].strip()

        if block.lower().startswith("python"):
            return block[len("python"):].lstrip()

    # Otherwise, return the first fenced block.
    if len(parts) >= 3:
        return parts[1].strip()

    return text
