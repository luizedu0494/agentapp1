# app_unified.py

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Importações do LangChain que estavam no main.py
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType
from langchain_experimental.agents.pandas.dataframe.base import create_pandas_dataframe_agent

# --- Configuração da Página e Chaves de API ---

# Carrega as variáveis de ambiente (importante para rodar localmente)
load_dotenv()

# Pega a chave da API do DeepSeek dos "Secrets" do Streamlit Cloud
# Use st.secrets["DEEPSEEK_API_KEY"] para o deploy
# Use os.getenv("DEEPSEEK_API_KEY") para rodar localmente
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY"))

# --- Interface do Usuário (código do seu app.py original) ---

st.set_page_config(
    page_title="Agente de Análise de Dados",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 Agente de Análise de Dados")
st.markdown(
    "Faça o upload de um arquivo CSV e faça uma pergunta em linguagem natural. "
    "O agente irá analisar os dados e gerar uma resposta para você."
)

uploaded_file = st.file_uploader(
    "**1. Escolha um arquivo CSV**", 
    type=["csv"]
)

question = st.text_input(
    "**2. Digite sua pergunta sobre os dados**",
    placeholder="Ex: Quantas linhas existem? Qual a média da coluna 'Valor'?"
)

# --- Lógica do Backend (código do seu main.py adaptado) ---

if st.button("Analisar Dados", type="primary"):
    if uploaded_file is not None and question:
        if not DEEPSEEK_API_KEY:
            st.error("Chave da API do DeepSeek não encontrada! Configure os Secrets no Streamlit Cloud.")
        else:
            with st.spinner("O agente está analisando os dados... Isso pode levar um momento. 🧠"):
                try:
                    # Carrega o CSV em um dataframe do Pandas
                    df = pd.read_csv(uploaded_file)

                    # Inicializa o LLM (usando DeepSeek)
                    llm = ChatOpenAI(
                        model="deepseek-chat", 
                        temperature=0,
                        api_key=DEEPSEEK_API_KEY,
                        base_url="https://api.deepseek.com/v1" 
                     )

                    # Cria o Agente Pandas DataFrame do LangChain
                    agent = create_pandas_dataframe_agent(
                        llm=llm,
                        df=df,
                        verbose=True,
                        agent_type=AgentType.OPENAI_FUNCTIONS,
                        handle_parsing_errors=True,
                    )

                    # Cria um prompt detalhado para guiar o agente
                    prompt = f"""
                    Sua tarefa é atuar como um analista de dados.
                    Use o dataframe fornecido para responder à seguinte pergunta: '{question}'.
                    Pense passo a passo. Execute o código Python para encontrar a resposta.
                    Forneça uma conclusão clara, concisa e em português.
                    """
                    
                    # Executa o agente com o prompt
                    # Usamos agent.invoke() em vez de ainvoike() pois o Streamlit não lida bem com async aqui
                    response = agent.invoke(prompt)
                    
                    # Exibe a resposta final do agente
                    st.success("Análise Concluída!")
                    st.markdown("### Resposta do Agente:")
                    st.write(response["output"])

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a análise: {e}")
    else:
        st.warning("Por favor, faça o upload de um arquivo CSV e digite uma pergunta antes de analisar.")

