# src/main.py

import argparse
import glob  # Importar glob para listar arquivos
import logging
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


def parse_arguments():
    """
    Analisa os argumentos de linha de comando.
    """
    parser = argparse.ArgumentParser(description="Execute o pipeline RAG.")
    parser.add_argument("--query", type=str, help="Query do usuário para o pipeline RAG.")
    return parser.parse_args()


def main():
    """
    Fluxo principal de execução do pipeline RAG.
    """
    # Configuração do logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    logging.info("Iniciando a aplicação...")

    # Analisa os argumentos de linha de comando
    args = parse_arguments()
    user_query = (
        args.query
        or "Quais são os elementos necessários para comprovar dano moral em contratos de prestação de serviços turísticos?"
    )

    try:
        # Carrega as configurações do arquivo config.yaml
        config_path = os.path.join(os.path.dirname(__file__), "../configs/config.yaml")
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        # Configurações específicas
        data_config = config["data"]
        model_config = config["model"]
        tokenizer_config = config["tokenizer"]
        prompts_config = config["prompts"]
        retrieval_config = config["retrieval"]

        # Baixa o tokenizer do NLTK se necessário
        nltk.download("punkt", quiet=True)

        # Caminho para a pasta de PDFs
        pdf_folder_path = os.path.join(os.path.dirname(__file__), "../", data_config["raw_data_path"])

        # Lista para armazenar todas as sentenças de todos os PDFs
        all_sentences = []

        # Lista todos os arquivos PDF na pasta
        pdf_files = glob.glob(os.path.join(pdf_folder_path, "*.pdf"))
        num_files = len(pdf_files)
        logging.info(f"Número de arquivos PDF encontrados: {num_files}")

        # Verifica se há arquivos PDF para processar
        if num_files == 0:
            logging.warning("Nenhum arquivo PDF encontrado na pasta especificada.")
            return

        # Percorre todos os arquivos PDF na pasta
        for pdf_file in pdf_files:
            logging.info(f"Lendo e processando o arquivo: {pdf_file}")
            text = read_pdf(pdf_file)
            sentences = merge_lines(text, tokenizer_config["path"])
            logging.info(f"Número de sentenças extraídas do arquivo {pdf_file}: {len(sentences)}")
            all_sentences.extend(sentences)

        # Verifica se coletou alguma sentença
        if not all_sentences:
            logging.warning("Nenhuma sentença foi extraída dos PDFs fornecidos.")
            return

        logging.info(f"Número total de sentenças extraídas: {len(all_sentences)}")

        # Criar cliente e coleção no banco vetorial
        chroma_client = chromadb.Client()
        collection = chroma_client.create_collection(name=retrieval_config["collection_name"], embedding_function=None)

        # Gera embeddings das sentenças
        embeddings = generate_embeddings(
            all_sentences, model_config["embedding_model_name"], model_config["batch_size"]
        )

        # Adiciona as sentenças e embeddings à coleção
        create_embeddings(collection, all_sentences, embeddings)

        # Ler os templates de prompt
        prompt_template_file = os.path.join(os.path.dirname(__file__), "../", prompts_config["template_file"])
        try:
            with open(prompt_template_file, "r") as file:
                prompts = yaml.safe_load(file)
        except FileNotFoundError:
            logging.error(f"Arquivo {prompt_template_file} não encontrado.")
            return

        system_prompt = prompts["System_Prompt"]
        prompt_template = prompts["Prompt"]

        # Realiza a expansão da query do usuário
        expansion_prompt = prompts["Prompt_Expansao"].format(query=user_query)
        response = get_completion(expansion_prompt, system_prompt, model_config["llm_name"])

        if response is None:
            logging.error("Não foi possível obter uma resposta do modelo para a expansão da query.")
            return

        # Processa a resposta e extrai termos e respostas fictícias
        response_json = parse_response(response)
        if response_json is None:
            logging.error("Não foi possível parsear a resposta do modelo para a expansão da query.")
            return

        queries = response_json.get("termos", [])
        respostas = response_json.get("respostas_ficticias", [])

        logging.info(f"Queries: {queries}")
        logging.info(f"Respostas fictícias: {respostas}")

        # Verifica se as queries e respostas não estão vazias
        if not queries and not respostas:
            logging.warning("Nenhuma query ou resposta fictícia disponível para recuperação.")
            return

        # Combina as queries e respostas para recuperação
        query_list = queries + respostas
        if not query_list:
            logging.warning("Nenhuma query ou resposta fictícia disponível para recuperação.")
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
            logging.warning("Nenhum documento encontrado para as queries fornecidas.")
            return

        if not all_docs:
            logging.warning("Nenhum documento relevante encontrado para as queries fornecidas.")
            return

        # Remove duplicatas mantendo a ordem
        all_docs = list(OrderedDict.fromkeys(all_docs))

        # Constrói o prompt final para o modelo de linguagem
        formatted_chunks = "\n".join([f"{chunk}\n" for chunk in all_docs])
        final_prompt = build_prompt(prompt_template, formatted_chunks, user_query)
        logging.info(f"Prompt construído:\n{final_prompt}")

        # Gera a resposta final usando o modelo de linguagem
        final_response = get_completion(final_prompt, system_prompt, model_config["llm_name"])
        if final_response is None:
            logging.error("Não foi possível obter a resposta final do modelo.")
            return

        # Exibe a query original e a resposta gerada
        logging.info(f"Query original: {user_query}")
        logging.info(f"Resposta gerada:\n{final_response}")

    except Exception as e:
        logging.exception("Ocorreu um erro durante a execução do pipeline:")
        logging.exception(e)


if __name__ == "__main__":
    main()
