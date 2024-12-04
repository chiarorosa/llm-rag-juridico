# src/main.py

import os
import sys

# Adiciona o diretório 'src' ao sys.path para permitir imports relativos
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from collections import OrderedDict

import chromadb
import nltk
import yaml

# Importações dos módulos do projeto
from data_loader import read_pdf
from embedding import create_embeddings, generate_embeddings
from llm_interface import get_completion, parse_response
from preprocessing import merge_lines
from retrieval import query_collection
from utils import build_prompt


def main():
    """
    Fluxo principal de execução do pipeline RAG.
    """
    # Carrega as configurações do arquivo config.yaml
    with open("../configs/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    # Configurações específicas
    data_config = config["data"]
    model_config = config["model"]
    tokenizer_config = config["tokenizer"]
    prompts_config = config["prompts"]
    retrieval_config = config["retrieval"]

    # Baixa o tokenizer do NLTK se necessário
    nltk.download("punkt")

    # Ler e pré-processar o texto do PDF
    pdf_file_path = os.path.join("../", data_config["raw_data_path"], data_config["pdf_file"])
    text = read_pdf(pdf_file_path)
    sentences = merge_lines(text, tokenizer_config["path"])

    # Criar cliente e coleção no banco vetorial
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(name=retrieval_config["collection_name"], embedding_function=None)

    # Gera embeddings das sentenças
    embeddings = generate_embeddings(sentences, model_config["embedding_model_name"], model_config["batch_size"])

    # Adiciona as sentenças e embeddings à coleção
    create_embeddings(collection, sentences, embeddings)

    # Teste simples de recuperação
    query_text = "What is RAG?"
    query_embedding = generate_embeddings(
        [query_text], model_config["embedding_model_name"], model_config["batch_size"]
    )
    results = query_collection(collection, query_embeddings=query_embedding.cpu().numpy().tolist(), n_results=3)
    print(f"Query: {query_text}\n")
    for doc, distance in zip(results["documents"][0], results["distances"][0]):
        print(f"Document: {doc}")
        print(f"Distance: {distance}\n")

    # Ler os templates de prompt
    prompt_template_file = os.path.join("../", prompts_config["template_file"])
    try:
        with open(prompt_template_file, "r") as file:
            prompts = yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Arquivo {prompt_template_file} não encontrado.")
        return

    system_prompt = prompts["System_Prompt"]
    prompt_template = prompts["Prompt"]

    # Realiza a expansão da query do usuário
    user_query = "Que modelos de LLMs são avaliados e qual é o principal resultado do artigo?"
    expansion_prompt = prompts["Prompt_Expansao"].format(query=user_query)
    response = get_completion(expansion_prompt, system_prompt, model_config["llm_name"])

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
    query_embeddings = generate_embeddings(
        query_list, model_config["embedding_model_name"], model_config["batch_size"]
    )
    query_embeddings_list = query_embeddings.cpu().numpy().tolist()

    # Consulta o banco de dados vetorial com as embeddings das queries
    docs = query_collection(collection, query_embeddings_list, retrieval_config["n_results"])

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
    all_docs = list(OrderedDict.fromkeys(all_docs))

    # Constrói o prompt final para o modelo de linguagem
    formatted_chunks = "\n".join([f"{chunk}\n" for chunk in all_docs])
    final_prompt = build_prompt(prompt_template, formatted_chunks, user_query)
    print("Prompt construído:\n", final_prompt)

    # Gera a resposta final usando o modelo de linguagem
    final_response = get_completion(final_prompt, system_prompt, model_config["llm_name"])
    if final_response is None:
        print("Não foi possível obter a resposta final do modelo.")
        return

    # Exibe a query original e a resposta gerada
    print("Query original:", user_query)
    print("Resposta gerada:\n", final_response)


if __name__ == "__main__":
    main()
