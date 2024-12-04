# -*- coding: utf-8 -*-
"""
rag_basico.ipynb - Implementação original em Jupyter Notebook desenvolvida por @ricardoaraujo na unidade de RAG do PPGC.
rag_basico.py - Implementação em Python adaptada para uso do Ollama por @chiarorosa.
"""

import json
import re

import chromadb
import fitz  # PyMuPDF para leitura de PDFs
import nltk
import torch
import yaml
from ollama import chat
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Configurações e constantes
BATCH_SIZE = 32  # Tamanho do batch para geração de embeddings
MODELLLM = "llama3.2:3b"  # Modelo de linguagem a ser usado
MODELEMB = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")  # Modelo de embeddings multilíngue
EMBEDDING_DIM = 384  # Dimensão dos embeddings do modelo Sentence-BERT
TOKENIZER = "tokenizers/punkt/english.pickle"  # Tokenizer do NLTK (inglês)
PROMPT_TEMPLATE_FILE = "prompt_template.yml"  # Arquivo com os templates de prompt

nltk.download("punkt")  # Baixar o tokenizer do NLTK


def read_pdf(file_path):
    """
    Lê o conteúdo de um arquivo PDF e retorna o texto completo usando PyMuPDF,
    incluindo o texto de tabelas e gráficos quando possível.

    Args:
        file_path (str): Caminho para o arquivo PDF.

    Returns:
        str: Texto completo extraído do PDF, incluindo texto de tabelas e gráficos.
    """
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            # Extrai o texto da página como um dicionário estruturado
            page_dict = page.get_text("dict")
            for block in page_dict["blocks"]:
                if block["type"] == 0:  # Bloco de texto
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += span["text"] + " "
                    text += "\n"
    return text


def merge_lines(text):
    """
    Pré-processa o texto, removendo espaços extras e separando por sentenças completas.

    Args:
        text (str): Texto completo extraído do PDF.

    Returns:
        list: Lista de sentenças.
    """
    # Remove espaços em branco extras
    text = re.sub(r"\s+", " ", text)
    # Carrega o tokenizer do NLTK
    tokenizer = nltk.data.load(TOKENIZER)
    # Tokeniza o texto em sentenças
    sentences = tokenizer.tokenize(text)
    return sentences


def generate_embeddings(sentences, batch_size=BATCH_SIZE):
    """
    Gera embeddings para uma lista de sentenças usando Sentence-BERT.

    Args:
        sentences (list): Lista de sentenças.
        batch_size (int): Tamanho do batch para processamento.

    Returns:
        torch.Tensor: Tensor contendo os embeddings das sentenças.
    """
    print("Gerando embeddings para as sentenças...")
    embeddings = []
    # Calcula o número total de batches
    total_batches = (len(sentences) + batch_size - 1) // batch_size

    # Processa os embeddings em batches
    for i in tqdm(range(0, len(sentences), batch_size), total=total_batches, desc="Processando batches"):
        batch = sentences[i : i + batch_size]
        batch_embeddings = MODELEMB.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)

    # Concatena todos os embeddings em um único tensor
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
    embeddings = generate_embeddings(sentences)
    embeddings_list = embeddings.cpu().numpy().tolist()
    ids = [f"sentence_{idx}" for idx in range(len(sentences))]

    # Adiciona todos os dados de uma vez ao ChromaDB
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
        iterator: Iterador de tuplas contendo documentos e distâncias.
    """
    # Gera o embedding para a query
    query_embedding = generate_embeddings([query_text])[0].cpu().numpy().tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=n_results)
    return zip(results["documents"][0], results["distances"][0])


def get_completion(prompt, system_prompt, model=MODELLLM):
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
            print(f"Erro ao decodificar JSON: {e}")
            print("Tentando corrigir a formatação do JSON...")

            # Tenta corrigir aspas simples para aspas duplas
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
    Constrói um prompt formatado com os chunks e a query.

    Args:
        prompt_template (str): Template do prompt.
        chunks (str): Chunks formatados do banco vetorial.
        query (str): Query do usuário.

    Returns:
        str: Prompt formatado.
    """
    return prompt_template.format(chunks=chunks, query=query)


def main():
    """
    Fluxo principal de execução do pipeline RAG.
    """
    # Ler e pré-processar o texto do PDF
    text = read_pdf("artigo_exemplo.pdf")
    sentences = merge_lines(text)

    # Criar cliente e coleção no banco vetorial
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
    try:
        with open(PROMPT_TEMPLATE_FILE, "r") as file:
            prompts = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Arquivo {PROMPT_TEMPLATE_FILE} não encontrado.")
        return

    system_prompt = prompts["System_Prompt"]
    prompt_template = prompts["Prompt"]

    # Realiza a expansão da query do usuário
    user_query = "Que modelos de LLMs são avaliados e qual é o principal resultado do artigo?"
    expansion_prompt = prompts["Prompt_Expansao"].format(query=user_query)
    response = get_completion(expansion_prompt, system_prompt)

    if response is None:
        print("Não foi possível obter uma resposta do modelo para a expansão da query.")
        return

    # Processa a resposta e extrai termos e respostas fictícias
    response_json = parse_response(response)
    if response_json is None:
        print("Não foi possível parsear a resposta do modelo para a expansão da query.")
        return

    queries = response_json.get("termos", [])
    respostas = response_json.get("respostas_ficticias", [])

    print("Queries:", queries)
    print("Respostas fictícias:", respostas)

    # Verifica se as queries e respostas não estão vazias
    if not queries and not respostas:
        print("Nenhuma query ou resposta fictícia disponível para recuperação.")
        return

    # Combina as queries e respostas para recuperação
    query_list = queries + respostas
    if not query_list:
        print("Nenhuma query ou resposta fictícia disponível para recuperação.")
        return

    # Gera embeddings para todas as queries de uma vez
    query_embeddings = generate_embeddings(query_list)
    query_embeddings_list = query_embeddings.cpu().numpy().tolist()

    # Consulta o banco de dados vetorial com as embeddings das queries
    docs = collection.query(query_embeddings=query_embeddings_list, n_results=10)

    # Coleta todos os documentos retornados
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

    # Remove duplicatas mantendo a ordem
    from collections import OrderedDict

    all_docs = list(OrderedDict.fromkeys(all_docs))

    # Constrói o prompt final para o modelo de linguagem
    formatted_chunks = "\n".join([f"{chunk}\n" for chunk in all_docs])
    final_prompt = build_prompt(prompt_template, formatted_chunks, user_query)
    print("Prompt construído:\n", final_prompt)

    # Gera a resposta final usando o modelo de linguagem
    final_response = get_completion(final_prompt, system_prompt)
    if final_response is None:
        print("Não foi possível obter a resposta final do modelo.")
        return

    # Exibe a query original e a resposta gerada
    print("Query original:", user_query)
    print("Resposta gerada:\n", final_response)


if __name__ == "__main__":
    main()
