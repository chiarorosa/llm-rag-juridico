# src/data_loader.py

import fitz


def read_pdf_with_metadata(file_path):
    """
    Lê o conteúdo de um arquivo PDF e retorna uma lista de tuplas contendo
    o texto da sentença, o nome do documento e o número da página.

    Args:
        file_path (str): Caminho para o arquivo PDF.

    Returns:
        list: Lista de tuplas (sentença, documento, página).
    """
    sentences_with_metadata = []
    document_name = os.path.basename(file_path)
    with fitz.open(file_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            # Extrai o texto da página como um dicionário estruturado
            page_dict = page.get_text("dict")
            text = ""
            for block in page_dict["blocks"]:
                if block["type"] == 0:  # Bloco de texto
                    for line in block["lines"]:
                        for span in line["spans"]:
                            text += span["text"] + " "
                    text += "\n"
            # Pré-processa o texto da página para obter as sentenças
            sentences = merge_lines(text, tokenizer_path)
            for sentence in sentences:
                sentences_with_metadata.append({"sentence": sentence, "document": document_name, "page": page_num})
    return sentences_with_metadata
