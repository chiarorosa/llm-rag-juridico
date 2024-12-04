# -*- coding: utf-8 -*-
"""
rag_basico.ipynb - implementação original em Jupyter Notebook desenvolvida por @ricardoaraujo na unidade de RAG do PPGC
rag_basico.py - implementação em Python adaptada para uso do ollama por @chiarorosa
"""

import json

import chromadb
import fitz  # PyMuPDF
import nltk
import torch
import yaml
from ollama import chat
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODELLLM = "llama3.2:3b"  # Modelo de linguagem a ser usado
MODELEMB = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # Sentence-BERT
EMBEDDING_DIM = 384  # Dimensão dos embeddings do modelo Sentence-BERT
TOKENIZER = "tokenizers/punkt/english.pickle"  # Tokenizer do NLTK

nltk.download("punkt")  # Baixar o tokenizer do NLTK


def read_pdf(file_path):
    """Lê o conteúdo de um arquivo PDF e retorna o texto completo usando PyMuPDF."""
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text


def merge_lines(text):
    """
    Pré-processa o texto, unindo linhas quebradas e separando por sentenças completas.

    Args:
        text (str): Texto completo extraído do PDF.
    Returns:
        list: Lista de sentenças.
    """
    text = text.replace("\n", " ")
    tokenizer = nltk.data.load(TOKENIZER)
    sentences = tokenizer.tokenize(text)
    return sentences


def generate_embeddings(sentences, batch_size=16):
    """
    Gera embeddings para uma lista de sentenças usando Sentence-BERT

    Args:
        sentences (list): Lista de sentenças.
    Returns:
        list: Lista contendo os embeddings das sentenças.
    """
    print("Gerando embeddings para as sentenças...")
    embeddings = []
    total_batches = (len(sentences) + batch_size - 1) // batch_size  # Calcula o número total de batches

    for i in tqdm(range(0, len(sentences), batch_size), total=total_batches, desc="Processando batches"):
        batch = sentences[i : i + batch_size]
        batch_embeddings = MODELEMB.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)

    embeddings = torch.cat(embeddings, dim=0)
    print("Embeddings gerados com sucesso.")
    return embeddings


def create_embeddings(collection, sentences):
    embeddings = generate_embeddings(sentences)
    embeddings_list = embeddings.cpu().numpy().tolist()
    ids = [f"sentence_{idx}" for idx in range(len(sentences))]

    # Adiciona todos os dados de uma vez
    collection.add(
        documents=sentences,
        embeddings=embeddings_list,
        ids=ids,
    )


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
    # Gera embedding para a query
    query_embedding = generate_embeddings([query_text])[0].cpu().numpy().tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return zip(results["documents"][0], results["distances"][0])


def get_completion(prompt, system_prompt, model=MODELLLM):
    """
    Obtém uma resposta do modelo de linguagem usando o Ollama.

    Args:
        prompt (str): O prompt principal a ser enviado ao modelo.
        system_prompt (str): O prompt de sistema que define o comportamento do modelo.
        model (str): Nome do modelo a ser usado.

    Returns:
        str: A resposta gerada pelo modelo ou None em caso de erro.
    """
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
    try:
        response = chat(model=model, messages=messages)
        return response["message"]["content"]
    except Exception as e:
        print(f"Erro ao obter resposta do modelo: {e}")
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
        print("A resposta do modelo está vazia.")
        return None

    import re

    # Remover marcadores de código e espaços em branco
    response = response.strip().strip("```").strip()

    # Procurar pelo primeiro objeto JSON na resposta
    match = re.search(r"\{.*\}", response, re.DOTALL)
    if match:
        json_str = match.group()
        # Tentar decodificar o JSON
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"Erro ao decodificar JSON: {e}")
            print("Tentando corrigir a formatação do JSON...")

            # Tentativa de corrigir aspas simples para aspas duplas
            json_str_fixed = json_str.replace("'", '"')
            try:
                return json.loads(json_str_fixed)
            except json.JSONDecodeError as e2:
                print(f"Erro ao decodificar JSON após correção: {e2}")
                print("JSON recebido:", json_str)
                return None
    else:
        print("Não foi possível encontrar um objeto JSON na resposta.")
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
    sentences = merge_lines(text)

    # Criar coleção no banco vetorial
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name="artigo_exemplo", embedding_function=None)
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

    if response is None:
        print("Não foi possível obter uma resposta do modelo para a expansão da query.")
        return

    # Processar a resposta
    response_json = parse_response(response)
    if response_json is None:
        print("Não foi possível parsear a resposta do modelo para a expansão da query.")
        return

    queries = response_json.get("termos", [])
    respostas = response_json.get("respostas_ficticias", [])

    print("Queries:", queries)
    print("Respostas fictícias:", respostas)

    # Verificar se queries e respostas não estão vazias
    if not queries and not respostas:
        print("Nenhuma query ou resposta fictícia disponível para recuperação.")
        return

    # Recuperação e construção do prompt
    query_list = queries + respostas
    if not query_list:
        print("Nenhuma query ou resposta fictícia disponível para recuperação.")
        return

    # Gera embeddings para todas as queries de uma vez
    query_embeddings = generate_embeddings(query_list)
    query_embeddings_list = query_embeddings.cpu().numpy().tolist()

    # Consulta o banco de dados com todas as embeddings
    docs = collection.query(query_embeddings=query_embeddings_list, n_results=10)

    all_docs = []
    if docs and "documents" in docs:
        for doc_list in docs["documents"]:
            all_docs.extend(doc_list)
    else:
        print("Nenhum documento encontrado para as queries fornecidas.")
        return

    if not all_docs:
        print("Nenhum documento relevante encontrado para as queries fornecidas.")
        return

    formatted_chunks = "\n".join([f"{chunk}\n" for chunk in all_docs])
    final_prompt = build_prompt(prompt_template, formatted_chunks, user_query)
    print("Prompt construído:\n", final_prompt)

    # Geração da resposta final
    final_response = get_completion(final_prompt, system_prompt)
    if final_response is None:
        print("Não foi possível obter a resposta final do modelo.")
        return

    print("Query original:", user_query)
    print("Resposta gerada:\n", final_response)


if __name__ == "__main__":
    main()
