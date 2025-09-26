# app.py

import streamlit as st
import requests

# URL do nosso backend FastAPI
BACKEND_URL = "http://127.0.0.1:8000/analyze"

# --- Configuração da Página do Streamlit ---
st.set_page_config(
    page_title="Agente de Análise de CSV",
    page_icon="🤖",
    layout="centered"
 )

# --- Interface do Usuário ---
st.title("🤖 Agente de Análise de Dados")
st.markdown(
    "Faça o upload de um arquivo CSV e faça uma pergunta em linguagem natural. "
    "O agente irá analisar os dados e gerar uma resposta para você."
)

# Widget para upload do arquivo
uploaded_file = st.file_uploader(
    "**1. Escolha um arquivo CSV**", 
    type=["csv"]
)

# Campo de texto para a pergunta do usuário
question = st.text_input(
    "**2. Digite sua pergunta sobre os dados**",
    placeholder="Ex: Quantas linhas existem? Qual a média da coluna 'Valor'?"
)

# Botão para iniciar a análise
if st.button("Analisar Dados", type="primary"):
    if uploaded_file is not None and question:
        # Mostra uma mensagem de "carregando" enquanto processa
        with st.spinner("O agente está analisando os dados... Isso pode levar um momento. 🧠"):
            try:
                # Prepara os dados para enviar na requisição POST
                files = {'file': (uploaded_file.name, uploaded_file.getvalue(), 'text/csv')}
                data = {'question': question}

                # Envia a requisição para o backend FastAPI
                response = requests.post(BACKEND_URL, files=files, data=data)

                if response.status_code == 200:
                    # Se a resposta for bem-sucedida, exibe o resultado
                    answer = response.json().get("answer", "Nenhuma resposta foi retornada.")
                    st.success("Análise Concluída!")
                    st.markdown("### Resposta do Agente:")
                    st.write(answer)
                else:
                    # Se houver um erro no backend, mostra a mensagem de erro
                    error_message = response.json().get("error", "Erro desconhecido no servidor.")
                    st.error(f"Falha na análise: {error_message}")

            except requests.exceptions.ConnectionError:
                st.error("Erro de Conexão: Não foi possível conectar ao backend. Verifique se o servidor (main.py) está em execução.")
            except Exception as e:
                st.error(f"Ocorreu um erro inesperado no frontend: {e}")
    else:
        # Avisa o usuário se faltar o arquivo ou a pergunta
        st.warning("Por favor, faça o upload de um arquivo CSV e digite uma pergunta antes de analisar.")
