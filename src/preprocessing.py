# src/preprocessing.py

import re

import nltk


def merge_lines(text, tokenizer_path):
    """
    Pré-processa o texto, removendo espaços extras e separando por sentenças completas.

    Args:
        text (str): Texto completo extraído do PDF.
        tokenizer_path (str): Caminho para o tokenizer do NLTK.

    Returns:
        list: Lista de sentenças.
    """
    # Remove espaços em branco extras
    text = re.sub(r"\s+", " ", text)
    # Carrega o tokenizer do NLTK
    tokenizer = nltk.data.load(tokenizer_path)
    # Tokeniza o texto em sentenças
    sentences = tokenizer.tokenize(text)
    return sentences
