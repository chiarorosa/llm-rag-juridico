# src/llm_interface.py

import logging

from ollama import chat


def get_completion(prompt, system_prompt, model):
    """
    Obtém uma resposta do modelo de linguagem usando o Ollama.

    Args:
        prompt (str): O prompt principal a ser enviado ao modelo.
        system_prompt (str): O prompt do sistema que define o comportamento do modelo.
        model (str): Nome do modelo a ser usado.

    Returns:
        str: A resposta gerada pelo modelo ou None em caso de erro.
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    try:
        logging.info("Enviando prompt ao modelo de linguagem...")
        response = chat(model=model, messages=messages)
        return response["message"]["content"]
    except Exception as e:
        logging.error(f"Erro ao obter resposta do modelo: {e}")
        return None


def parse_response(response):
    try:
        # Se a resposta for uma string JSON, tenta carregá-la como dicionário
        if isinstance(response, str):
            response = json.loads(response)

        # Inicializa a lista para coletar todos os termos
        termos_combinados = []

        # Função auxiliar para processar dicionários recursivamente
        def extrair_termos(dados):
            for key, value in dados.items():
                if isinstance(value, list):  # Adiciona diretamente listas
                    termos_combinados.extend(value)
                elif isinstance(value, dict):  # Explora dicionários aninhados
                    extrair_termos(value)

        # Processa o dicionário inicial
        extrair_termos(response)

        # Remove duplicados e retorna como lista de sentenças
        return list(set(termos_combinados))

    except Exception as e:
        logging.error(f"Erro ao parsear a resposta: {e}")
        return []
