# src/embedding.py

import logging

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def generate_embeddings(sentences, model_name, batch_size):
    """
    Gera embeddings para uma lista de sentenças usando Sentence-BERT.

    Args:
        sentences (list): Lista de sentenças.
        model_name (str): Nome do modelo de embeddings.
        batch_size (int): Tamanho do batch para processamento.

    Returns:
        torch.Tensor: Tensor contendo os embeddings das sentenças.
    """
    logging.info("Gerando embeddings para as sentenças...")
    embeddings = []
    model = SentenceTransformer(model_name)
    # Calcula o número total de batches
    total_batches = (len(sentences) + batch_size - 1) // batch_size

    # Processa os embeddings em batches
    for i in tqdm(range(0, len(sentences), batch_size), total=total_batches, desc="Processando batches"):
        batch = sentences[i : i + batch_size]
        batch_embeddings = model.encode(batch, convert_to_tensor=True)
        embeddings.append(batch_embeddings)

    # Concatena todos os embeddings em um único tensor
    embeddings = torch.cat(embeddings, dim=0)
    logging.info("Embeddings gerados com sucesso.")
    return embeddings


# src/embedding.py


def store_embeddings(collection, sentences, embeddings, metadatas):
    """
    Adiciona sentenças, seus embeddings e metadados ao banco de dados vetorial.

    Args:
        collection (chromadb.Collection): Coleção do banco vetorial.
        sentences (list): Lista de sentenças.
        embeddings (torch.Tensor): Tensor contendo os embeddings das sentenças.
        metadatas (list): Lista de metadados correspondentes a cada sentença.
    """
    embeddings_list = embeddings.cpu().numpy().tolist()
    ids = [f"sentence_{idx}" for idx in range(len(sentences))]

    # Adiciona todos os dados de uma vez ao ChromaDB
    collection.add(documents=sentences, embeddings=embeddings_list, ids=ids, metadatas=metadatas)
    logging.info(f"Adicionadas {len(sentences)} sentenças e seus embeddings ao banco de dados vetorial.")
