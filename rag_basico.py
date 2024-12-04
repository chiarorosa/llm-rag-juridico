# -*- coding: utf-8 -*-
"""
rag_basico.ipynb - implementação original em Jupyter Notebook desenvolvida por @ricardoaraujo na unidade de RAG do PPGC
rag_basico.py - implementação em Python adaptada para uso do ollama por @chiarorosa
"""

import json

import chromadb
import yaml
from ollama import chat
from PyPDF2 import PdfReader
from tqdm import tqdm


def read_pdf(file_path):
    """Lê o conteúdo de um arquivo PDF e retorna o texto completo."""
    reader = PdfReader(file_path)
    return "".join([page.extract_text() + "\n" for page in reader.pages])


def merge_lines(text_lines):
    """
    Pré-processa o texto, unindo linhas e separando por sentenças.

    Args:
        text_lines (list): Lista de strings (linhas do texto).
    Returns:
        list: Lista de sentenças.
    """
    merged, current_sentence = [], ""
    for line in text_lines:
        line = line.strip()
        if not line:
            continue

        if current_sentence:
            line = " " + line
        current_sentence += line

        if ". " in current_sentence:
            parts = current_sentence.split(". ")
            merged.extend([part + "." for part in parts[:-1]])
            current_sentence = parts[-1]

    if current_sentence:
        merged.append(current_sentence)
    return merged


def create_embeddings(collection, sentences):
    """
    Adiciona sentenças a um banco de dados vetorial.

    Args:
        collection (chromadb.Collection): Coleção do banco vetorial.
        sentences (list): Lista de sentenças.
    """
    for idx, sentence in tqdm(enumerate(sentences), total=len(sentences)):
        collection.add(documents=[sentence], ids=[f"sentence_{idx}"])


def query_collection(collection, query_text, n_results=3):
    """
    Consulta uma coleção do banco vetorial com uma query.

    Args:
        collection (chromadb.Collection): Coleção do banco vetorial.
        query_text (str): Texto da query.
        n_results (int): Número de resultados a recuperar.

    Returns:
        list: Lista de documentos e distâncias.
    """
    results = collection.query(query_texts=[query_text], n_results=n_results)
    return zip(results["documents"][0], results["distances"][0])


def get_completion(prompt, system_prompt, model="llama3.2:3b"):
    """
    Obtém uma resposta do modelo de linguagem usando o Ollama.

    Args:
        prompt (str): O prompt principal a ser enviado ao modelo.
        system_prompt (str): O prompt de sistema que define o comportamento do modelo.
        model (str): Nome do modelo a ser usado.

    Returns:
        str: A resposta gerada pelo modelo.
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    response = chat(model=model, messages=messages)
    return response["message"]["content"]


def parse_response(response):
    """
    Limpa e tenta decodificar uma resposta JSON.

    Args:
        response (str): Resposta do modelo.
    Returns:
        dict: Objeto JSON decodificado ou None se houver erro.
    """
    if not response:
        print("A resposta do modelo está vazia.")
        return None

    response = response.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON: {e}")
        print("Resposta recebida:", response)
        return None


def build_prompt(prompt_template, chunks, query):
    """
    Constroi um prompt formatado com os chunks e a query.

    Args:
        prompt_template (str): Template do prompt.
        chunks (str): Chunks formatados do banco vetorial.
        query (str): Query do usuário.

    Returns:
        str: Prompt formatado.
    """
    return prompt_template.format(chunks=chunks, query=query)


# Main pipeline
def main():
    # Ler e pré-processar o texto do PDF
    text = read_pdf("artigo_exemplo.pdf")
    sentences = merge_lines(text.split("\n"))

    # Criar coleção no banco vetorial
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="artigo_exemplo")
    create_embeddings(collection, sentences)

    # Teste simples de recuperação
    query = "What is RAG?"
    results = query_collection(collection, query)
    print(f"Query: {query}\n")
    for doc, distance in results:
        print(f"Document: {doc}")
        print(f"Distance: {distance}\n")

    # Ler os templates de prompt
    with open("prompt_template.yml", "r") as file:
        prompts = yaml.safe_load(file)
    system_prompt = prompts["System_Prompt"]
    prompt_template = prompts["Prompt"]

    # Expansão de query
    user_query = "Que modelos de LLMs são avaliados e qual é o principal resultado do artigo?"
    expansion_prompt = prompts["Prompt_Expansao"].format(query=user_query)
    response = get_completion(expansion_prompt, system_prompt)

    # Processar a resposta
    response_json = parse_response(response)
    queries = response_json.get("termos", []) if response_json else []
    respostas = response_json.get("respostas_ficticias", []) if response_json else []

    print("Queries:", queries)
    print("Respostas fictícias:", respostas)

    # Recuperação e construção do prompt
    all_docs = []
    for query_ in queries + respostas:
        docs = collection.query(query_texts=[query_], n_results=10)
        all_docs.extend(docs["documents"][0])

    formatted_chunks = "\n".join([f"{chunk}\n" for chunk in all_docs])
    final_prompt = build_prompt(prompt_template, formatted_chunks, user_query)
    print("Prompt construído:\n", final_prompt)

    # Geração da resposta final
    final_response = get_completion(final_prompt, system_prompt)
    print("Query original:", user_query)
    print("Resposta gerada:\n", final_response)


if __name__ == "__main__":
    main()
