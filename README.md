# Projeto: Sistema Inteligente para Busca de Jurisprudências com LLM e RAG

###

_Projeto final RAG para disciplina “Grandes Modelos de Linguagem aplicados a Geração Aumentada via Recuperação” do Programa de Pós-Graduação em Computação da Universidade Federal de Pelotas, ministrada em 2024/2 por Ricardo Matsumura Araújo._

_autor: Doutorando Pablo De Chiaro_

## Descrição Geral

Este projeto implementa um **sistema inteligente baseado em LLM (Large Language Models) com RAG (Retrieval-Augmented Generation)**, escrito em Python, para auxiliar usuários na busca por jurisprudências e explicações de decisões judiciais.

O sistema é um **assistente jurídico especializado**, capaz de:

- Encontrar decisões relevantes de tribunais com base na **query** fornecida pelo usuário focando nas Ementas dos Processos (neste caso TJRS - Comarca de Pelotas) ênfase 'tutela antecipada'

## Principais Funcionalidades

1. **Busca de Jurisprudências**:

   - Recupera documentos relevantes usando uma combinação de embeddings e metadados.
   - Expande a consulta do usuário para melhorar a recuperação de informações.

2. **Geração de Respostas**:

   - Analisa os chunks recuperados e gera respostas contextualizadas usando prompts estruturados.

3. **Configuração Personalizável**:
   - Todas as configurações principais estão em `configs/config.yaml` para evitar modificações diretas no código.

## Arquitetura do Projeto

A estrutura do projeto é organizada para garantir modularidade, escalabilidade e fácil manutenção:

```
.
├── README.md                   # Documentação geral
├── configs
│   └── config.yaml             # Configurações gerais do sistema
├── data
│   ├── raw                     # Documentos PDF utilizados como fonte
│   └── responses               # Respostas geradas e logs
├── notebooks
│   └── blank                   # Notebooks para prototipagem
├── prompts
│   └── prompt_template.yml     # Templates de prompts para a LLM
├── requirements.txt            # Dependências do projeto
└── src                         # Código-fonte principal
    ├── __init__.py
    ├── app_streamlit.py        # Interface para visualização com Streamlit
    ├── data_loader.py          # Leitura e manipulação de dados
    ├── embedding.py            # Geração de embeddings
    ├── fast_api.py             # API para integração com FastAPI
    ├── llm_interface.py        # Comunicação com a LLM
    ├── main.py                 # Pipeline principal do sistema
    ├── preprocessing.py        # Pré-processamento dos dados
    ├── retrieval.py            # Recuperação de informações (RAG)
    └── utils.py                # Funções auxiliares
```

## Configurações Gerais

As configurações são centralizadas no arquivo `configs/config.yaml`:

### Exemplos de Configurações:

- **Dados**:

  ```yaml
  data:
    raw_data_path: "data/raw/"
  ```

- **Modelos**:

  ```yaml
  model:
    llm_name: "qwen2.5:3b"
    embedding_model_name: "paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: 32
  ```

- **Tokenização**:

  ```yaml
  tokenizer:
    path: "tokenizers/punkt/portuguese.pickle"
  ```

- **Prompts**:

  ```yaml
  prompts:
    template_file: "prompts/prompt_template.yml"
  ```

- **Recuperação**:
  ```yaml
  retrieval:
    collection_name: "tjrs"
    n_results: 10
  ```

## Fluxo de Execução

1. **Pré-processamento**:

   - Os documentos PDF são carregados de `data/raw/` e processados em chunks utilizando `data_loader.py`.

2. **Geração de Embeddings**:

   - Embeddings são gerados utilizando o modelo definido em `embedding_model_name`.

3. **Recuperação**:

   - A query do usuário é expandida e utilizada para recuperar chunks relevantes do banco vetorial (ChromaDB).

4. **Geração de Resposta**:
   - A LLM utiliza os chunks e a query para gerar respostas fundamentadas, destacando critérios legais e precedentes.

## Principais Bibliotecas Utilizadas

- **ollama**: Integração com modelos LLM externos.
- **sentence-transformers**: Geração de embeddings com modelos pré-treinados.
- **streamlit**: Interface gráfica para visualização e testes do sistema.
- **FastAPI**: Criação de uma API REST para expor as funcionalidades do sistema, permitindo integração com outros sistemas e aplicações externas.
- **chromadb**: Gerenciamento do banco de dados vetorial.
- **nltk**: Tokenização e processamento de linguagem natural.
- **torch**: Backend para modelos baseados em PyTorch.

## Como Executar o Projeto

1. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure os parâmetros em `configs/config.yaml`.

3. (Opcional) Execute o pipeline principal:

   ```bash
   python src/main.py --query "consulta"
   ```

4. Streamlit:

   ```bash
   streamlit run src/app_streamlit.py
   ```

5. Use a API via FastAPI dentro do diretório src/:
   ```bash
   uvicorn fast_api:app --reload
   ```

## Responsabilidade do Sistema

Este sistema é um **assistente jurídico especializado**, limitado a:

- Recuperar jurisprudências relevantes com base na query do usuário.
- Apresentar critérios legais e argumentos baseados em precedentes.

**Nota:** Não substitui um advogado ou consulta jurídica formal.

---

Se precisar de mais informações ou ajustes, por favor, me avise!

## RAG básico referência:

https://github.com/ricardoaraujo/ppgc_rag
