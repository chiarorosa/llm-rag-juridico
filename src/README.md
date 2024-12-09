# Pipeline Principal (`main.py`)

## Funcionalidades Principais

1. **Carregamento de Configurações**:

   - Todas as configurações estão centralizadas no arquivo `configs/config.yaml` para facilitar a personalização.

2. **Processamento de PDFs**:

   - Lê documentos PDF na pasta `data/raw/`.
   - Extrai sentenças e metadados (como número do documento e página).

3. **Geração e Armazenamento de Embeddings**:

   - Gera embeddings para as sentenças extraídas usando modelos configuráveis de `sentence-transformers`.
   - Armazena as sentenças e seus embeddings em um banco vetorial gerenciado pelo **ChromaDB**.

4. **Expansão e Recuperação de Informações**:

   - Recebe uma query do usuário e expande seus termos para melhorar a precisão da recuperação.
   - Consulta o banco vetorial e retorna documentos relevantes baseados nos embeddings da query expandida.

5. **Geração de Resposta Final**:

   - Combina a query original, os chunks recuperados e os templates de prompts para gerar uma resposta final usando a LLM configurada.

6. **Logs e Erros**:
   - Todas as etapas são monitoradas com mensagens detalhadas de log para facilitar o rastreamento de erros e o acompanhamento do processo.

## Estrutura do Código

### Principais Componentes:

1. **Funções Utilizadas:**

   - `parse_arguments`: Analisa os argumentos de linha de comando (opcional).
   - `read_pdf_with_metadata`: Processa PDFs e extrai sentenças com metadados.
   - `generate_embeddings`: Gera embeddings para sentenças ou queries.
   - `store_embeddings`: Armazena embeddings e sentenças no banco vetorial.
   - `query_collection`: Consulta o banco vetorial para recuperar documentos relevantes.
   - `get_completion`: Interage com a LLM para gerar respostas baseadas nos prompts construídos.
   - `build_prompt`: Constrói o prompt final combinando query e chunks recuperados.
   - `parse_response`: Processa e interpreta as respostas JSON da LLM.

2. **Fluxo Principal:**
   - **Carregamento de Configurações:**
     - Lê o arquivo `config.yaml` para configurar caminhos, modelos e parâmetros.
   - **Processamento de PDFs:**
     - Extrai sentenças e metadados dos documentos PDF.
   - **Criação de Banco Vetorial:**
     - Armazena sentenças e embeddings no ChromaDB.
   - **Consulta e Expansão:**
     - Expande a query do usuário e consulta o banco vetorial para recuperar documentos relevantes.
   - **Geração de Resposta Final:**
     - Gera a resposta final com base nos dados recuperados e no template de prompts.

## Dependências

O pipeline utiliza as seguintes bibliotecas principais:

- **chromadb**: Para gerenciar o banco vetorial.
- **nltk**: Para tokenização e manipulação de texto.
- **torch**: Backend para geração de embeddings com `sentence-transformers`.
- **yaml**: Leitura e manipulação de arquivos de configuração.
- **json**: Processamento e validação de respostas no formato JSON.
- **logging**: Registro e acompanhamento de logs detalhados durante o pipeline.

## Como Executar

1. Certifique-se de que as dependências estão instaladas:

   ```bash
   pip install -r requirements.txt
   ```

2. Configure os parâmetros no arquivo `configs/config.yaml`, incluindo:

   - Caminhos para PDFs.
   - Modelos de embeddings e LLM.
   - Parâmetros de recuperação e prompts.

3. Execute o pipeline principal:

   ```bash
   python src/main.py
   ```

4. Para passar uma query personalizada, utilize:
   ```bash
   python src/main.py --query "Sua consulta aqui"
   ```

## Exemplo de Fluxo

### Entrada:

- PDFs armazenados em `data/raw/`.
- Query do usuário: _"Quais são os requisitos legais para a concessão de tutela antecipada?"_

### Saída:

- Resposta final estruturada:
  ```json
  {
    "consulta": [
      {
        "chunks": [
          { "documento": "007.pdf", "pagina": 1 },
          { "documento": "004.pdf", "pagina": 2 }
        ],
        "conteudo": "Os critérios incluem probabilidade do direito e risco ao resultado útil do processo."
      }
    ]
  }
  ```

## Possíveis Ajustes

- Alterar o modelo de embeddings no arquivo `config.yaml`.
- Modificar os templates de prompts em `prompts/prompt_template.yml`.

## Logs e Debug

- Todos os eventos importantes do pipeline são registrados no console.
- Logs detalhados incluem:
  - Número de arquivos PDF processados.
  - Sentenças e embeddings gerados.
  - Documentos recuperados e resultados finais.
