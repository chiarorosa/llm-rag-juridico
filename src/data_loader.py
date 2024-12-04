# src/data_loader.py

import fitz  # PyMuPDF para leitura de PDFs


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
