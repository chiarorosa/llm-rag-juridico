import json
import time

import requests
import streamlit as st


def consulta(user_query):
    url = "http://127.0.0.1:8000/query"
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    data = {"query": f"{user_query}"}
    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Request failed with status code {response.status_code}"}


def main():
    st.title("juris.AI")
    st.write("Insira sua query abaixo e clique no botão para consultar a API.")

    # Campo de entrada para a query
    user_query = st.text_input("Digite sua query:", "")

    if st.button("Enviar Query"):
        if user_query.strip() == "":
            st.error("Por favor, insira uma query antes de enviar.")
        else:
            with st.spinner("Consultando a API..."):
                start_time = time.time()
                response = consulta(user_query)
                end_time = time.time()
                elapsed_time = end_time - start_time

                st.success("Consulta concluída!")
                st.write(f"Tempo de resposta: {elapsed_time:.2f} segundos")

                # Pretty print do JSON de resposta
                st.json(response)


if __name__ == "__main__":
    main()
