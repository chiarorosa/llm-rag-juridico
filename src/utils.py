# src/utils.py


def build_prompt(prompt_template, chunks, query):
    """
    Constrói um prompt formatado com os chunks e a query.

    Args:
        prompt_template (str): Template do prompt.
        chunks (str): Chunks formatados do banco vetorial.
        query (str): Query do usuário.

    Returns:
        str: Prompt formatado.
    """
    return prompt_template.format(chunks=chunks, query=query)
