# -*- coding: utf-8 -*-
"""
rag_basico.ipynb - implementação original em Jupyter Notebook desenvolvida por @ricardoaraujo na unidade de RAG do PPGC
rag_basico.py - implementação em Python adaptada para uso do ollama por @chiarorosa
"""

import json

import chromadb
import nltk
import torch
import yaml
from ollama import chat
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

MODELLLM = "llama3.2:3b"  # Modelo de linguagem a ser usado
MODELEMB = SentenceTransformer("all-MiniLM-L6-v2")  # Sentence-BERT
EMBEDDING_DIM = 384  # Dimensão dos embeddings do modelo Sentence-BERT
TOKENIZER = "tokenizers/punkt/english.pickle"  # Tokenizer do NLTK


def read_pdf(file_path):
    """Lê o conteúdo de um arquivo PDF e retorna o texto completo."""
    reader = PdfReader(file_path)
    return "".join([page.extract_text() + "\n" for page in reader.pages])


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
    print("Gerando embeddings para as sentenças...")
    """
    Gera embeddings para uma lista de sentenças usando Sentence-BERT.

    Args:
        sentences (list): Lista de sentenças.
    Returns:
        list: Lista contendo os embeddings das sentenças.
    """
    embeddings = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        print(f"Gerando embeddings para batch de {len(batch)} sentenças...")
        batch_embeddings = MODELEMB.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)
    embeddings = torch.cat(embeddings, dim=0)
    print("Embeddings gerados com sucesso.")
    return embeddings


def create_embeddings(collection, sentences):
    """
    Adiciona sentenças e seus embeddings ao banco de dados vetorial.

    Args:
        collection (chromadb.Collection): Coleção do banco vetorial.
        sentences (list): Lista de sentenças.
    """
    embeddings = generate_embeddings(sentences)  # Geração de embeddings com Sentence-BERT

    for idx, (sentence, embedding) in enumerate(zip(sentences, embeddings)):
        collection.add(
            documents=[sentence],
            embeddings=[embedding.cpu().numpy().tolist()],  # Converter tensor para lista
            ids=[f"sentence_{idx}"],
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

    Modelos testados: phi3.5:3.8b, llama3.1:8b, llama3.2:3b

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

    # Limpar a resposta removendo código e formatação desnecessária
    response = response.strip()
    if response.startswith("```json"):
        response = response[7:]
    if response.endswith("```"):
        response = response[:-3]

    response = response.replace("\n", " ").replace("\r", " ").replace("```", "").strip()

    try:
        return json.loads(response)
    except json.JSONDecodeError as e:
        print(f"Erro ao decodificar JSON: {e}")
        try:
            # Tentativa de localizar o início e fim do JSON válido
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end != -1:
                response = response[start:end]
                return json.loads(response)
        except json.JSONDecodeError as e2:
            print(f"Erro adicional ao tentar extrair JSON: {e2}")
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

    # Processar a resposta
    response_json = parse_response(response)
    queries = response_json.get("termos", []) if response_json else []
    respostas = response_json.get("respostas_ficticias", []) if response_json else []

    print("Queries:", queries)
    print("Respostas fictícias:", respostas)

    # Recuperação e construção do prompt
    all_docs = []
    for query_ in queries + respostas:
        docs = collection.query(
            query_embeddings=[generate_embeddings([query_])[0].cpu().numpy().tolist()], n_results=10
        )
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
