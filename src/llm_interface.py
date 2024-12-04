# src/llm_interface.py

import json
import logging
import re

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
    """
    Limpa e tenta decodificar uma resposta JSON.

    Args:
        response (str): Resposta do modelo.

    Returns:
        dict: Objeto JSON decodificado ou None se houver erro.
    """
    if not response:
        logging.warning("A resposta do modelo está vazia.")
        return None

    # Remove marcadores de código e espaços em branco
    response = response.strip().strip("```").strip()

    # Procura pelo primeiro objeto JSON na resposta
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        json_str = match.group()
        # Tenta decodificar o JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logging.error(f"Erro ao decodificar JSON: {e}")
            logging.info("Tentando corrigir a formatação do JSON...")

            # Tenta corrigir aspas simples para aspas duplas
            json_str_fixed = json_str.replace("'", '"')
            try:
                return json.loads(json_str_fixed)
            except json.JSONDecodeError as e2:
                logging.error(f"Erro ao decodificar JSON após correção: {e2}")
                logging.error(f"JSON recebido: {json_str}")
                return None
    else:
        logging.error("Não foi possível encontrar um objeto JSON na resposta.")
        logging.error(f"Resposta recebida: {response}")
        return None
