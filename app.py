# app.py (versão com pré-visualização dos dados)

import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Importações do LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent # Usando a importação simplificada

# --- Configuração da Página e Chaves de API ---

load_dotenv()
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY"))

# --- Interface do Usuário ---

st.set_page_config(
    page_title="Agente de Análise de Dados",
    page_icon="🤖",
    layout="wide"  # Mudei para "wide" para dar mais espaço para as tabelas
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

# --- NOVA SEÇÃO: PRÉ-VISUALIZAÇÃO DOS DADOS ---
# Este bloco de código será executado assim que um arquivo for carregado.
if uploaded_file is not None:
    try:
        # Carrega o dataframe para a pré-visualização
        df_preview = pd.read_csv(uploaded_file)
        
        st.markdown("---") # Adiciona uma linha divisória
        st.subheader("Pré-visualização dos Dados")
        
        # Mostra as 5 primeiras linhas do arquivo
        st.write("**Primeiras 5 linhas:**")
        st.dataframe(df_preview.head())
        
        # Mostra informações sobre as colunas
        st.write("**Informações das Colunas:**")
        # Cria um dataframe com as informações para exibir de forma organizada
        info_df = pd.DataFrame({
            "Coluna": df_preview.columns,
            "Tipo de Dado": df_preview.dtypes.astype(str),
            "Valores Não Nulos": df_preview.count().values
        })
        st.dataframe(info_df)

    except Exception as e:
        st.error(f"Ocorreu um erro ao tentar ler o arquivo CSV: {e}")
# -------------------------------------------------

st.markdown("---")
st.subheader("2. Faça sua Pergunta ao Agente")

question = st.text_input(
    "Digite sua pergunta sobre os dados:",
    placeholder="Ex: Qual a correlação entre a coluna 'Time' e 'Amount'?"
)

# --- Lógica do Agente (permanece a mesma) ---

if st.button("Analisar Dados", type="primary"):
    if uploaded_file is not None and question:
        if not DEEPSEEK_API_KEY:
            st.error("Chave da API não encontrada! Configure os Secrets no Streamlit Cloud.")
        else:
            with st.spinner("O agente está analisando os dados... Isso pode levar um momento. 🧠"):
                try:
                    # O arquivo precisa ser "rebobinado" para ser lido novamente pelo agente
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)

                    llm = ChatOpenAI(
                        model="deepseek-chat", 
                        temperature=0,
                        api_key=DEEPSEEK_API_KEY,
                        base_url="https://api.deepseek.com/v1" 
                     )

                    agent = create_pandas_dataframe_agent(
                        llm=llm,
                        df=df,
                        verbose=True,
                        agent_type=AgentType.OPENAI_FUNCTIONS,
                        handle_parsing_errors=True,
                        allow_dangerous_code=True
                    )

                    prompt = f"""
                    Sua tarefa é atuar como um analista de dados.
                    Use o dataframe fornecido para responder à seguinte pergunta: '{question}'.
                    Pense passo a passo. Execute o código Python para encontrar a resposta.
                    Forneça uma conclusão clara, concisa e em português.
                    """
                    
                    response = agent.invoke(prompt)
                    
                    st.success("Análise Concluída!")
                    st.markdown("### Resposta do Agente:")
                    st.write(response["output"])

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a análise: {e}")
    else:
        st.warning("Por favor, faça o upload de um arquivo CSV e digite uma pergunta antes de analisar.")
