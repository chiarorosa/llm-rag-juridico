# src/utils.py

import json


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


def cleaned_json(response):
    """
    Limpa e processa uma string contendo JSON, removendo trechos desnecessários 
    e realizando o parse para um objeto Python.

    Parâmetros:
        response (str): Uma string contendo possivelmente JSON, com trechos desnecessários como delimitadores de código (e.g., "```json", """ """).

    Retorna:
        dict: O JSON parseado como um dicionário Python.

    """
    try:
        # Remove trechos desnecessários, como a indicação de ser código
        cleaned_data = response.strip().strip('"""').strip("```json").strip("```")

        # Parse do JSON
        parsed_data = json.loads(cleaned_data)
        return parsed_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Erro ao fazer o parse do JSON: {e}")
