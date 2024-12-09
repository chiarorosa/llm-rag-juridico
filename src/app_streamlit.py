import os
import time

import requests
import streamlit as st
from PyPDF2 import PdfReader


def consulta(user_query):
    url = "http://127.0.0.1:8000/query"
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    data = {"query": f"{user_query}"}
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}"}


# Função para abrir e exibir PDFs
def abrir_pdf(pdf_path):
    try:
        if not os.path.exists(pdf_path):
            st.error(f"Arquivo não encontrado: {pdf_path}")
            return

        # Ler o PDF
        reader = PdfReader(pdf_path)
        texto = ""
        for page in reader.pages:
            texto += page.extract_text()

        # Exibir conteúdo no Streamlit
        st.text_area(f"Visualizando: {os.path.basename(pdf_path)}", texto, height=400)

        # Botão para baixar o PDF
        with open(pdf_path, "rb") as f:
            st.download_button(
                label="Baixar PDF",
                data=f,
                file_name=os.path.basename(pdf_path),
                mime="application/pdf",
            )
    except Exception as e:
        st.error(f"Erro ao abrir o PDF: {e}")


# Função principal da aplicação
def main():
    st.title("juris.ai")
    st.write("Insira sua query abaixo e clique no botão para consultar a API.")

    user_query = st.text_input("Digite sua query:", "")

    # Inicializar session_state para o PDF selecionado
    if "selected_pdf" not in st.session_state:
        st.session_state.selected_pdf = None

    if st.button("Enviar Query"):
        if user_query.strip() == "":
            st.error("Por favor, insira uma query antes de enviar.")
        else:
            with st.spinner("Consultando a API..."):
                # Simulação da consulta
                response = consulta(user_query)  # Substitua pela função real
                if "consulta" in response:
                    st.session_state.query_result = response
                    st.session_state.selected_pdf = None  # Resetar seleção de PDF
                else:
                    st.error("Não foi possível processar a resposta da API.")

    # Exibir resultados da consulta, se disponíveis
    if "query_result" in st.session_state:
        response = st.session_state.query_result
        for index, result in enumerate(response["consulta"]):
            st.subheader(f"Resultado {index + 1}")

            # Exibir links dos PDFs
            st.markdown("### Documentos Relacionados:")
            for chunk in result["chunks"]:
                pdf_path = f"data/raw/{chunk['documento']}"
                if st.button(
                    f"Abrir {chunk['documento']} (Página {chunk['pag']})", key=f"{index}_{chunk['documento']}"
                ):
                    st.session_state.selected_pdf = pdf_path

            # Exibir conteúdo relacionado
            st.markdown("### Conteúdo Relacionado:")
            st.write(result["conteúdo"])

        # Exibir o JSON completo
        st.markdown("### JSON Completo:")
        st.json(response)

    # Exibir o PDF selecionado
    if st.session_state.selected_pdf:
        st.markdown("---")
        abrir_pdf(st.session_state.selected_pdf)


if __name__ == "__main__":
    main()
