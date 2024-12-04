# src/retrieval.py


def query_collection(collection, query_embeddings, n_results):
    """
    Realiza uma consulta na coleção usando embeddings de query.

    Args:
        collection (chromadb.Collection): Coleção do banco vetorial.
        query_embeddings (list): Lista de embeddings das queries.
        n_results (int): Número de resultados a recuperar.

    Returns:
        dict: Resultados da consulta.
    """
    results = collection.query(query_embeddings=query_embeddings, n_results=n_results)
    return results
