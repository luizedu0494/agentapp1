# -----------------------------------------------------------------
# Agente de Análise de Dados com Streamlit e LangChain (usando Google Gemini)
# -----------------------------------------------------------------

# --- 1. Importação das Bibliotecas Necessárias ---
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Importações específicas do LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents.agent_types import AgentType
from langchain_agents import create_pandas_dataframe_agent



# --- 2. Configuração da Página e Chaves de API ---
st.set_page_config(
    page_title="Agente de Análise com Gemini",
    page_icon="🤖",
    layout="wide"
)
load_dotenv()
API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY"))

# --- 3. Interface do Usuário (UI) com Streamlit ---
st.title("🤖 Agente de Análise de Dados com Google Gemini")
st.markdown(
    "Faça o upload de um arquivo CSV, visualize os dados e faça perguntas em linguagem natural. "
    "O agente irá analisar o arquivo e gerar uma resposta para você."
)
uploaded_file = st.file_uploader(
    "**1. Escolha um arquivo CSV**",
    type=["csv"]
)

# --- 4. Funcionalidade de Pré-visualização dos Dados ---
if uploaded_file is not None:
    try:
        df_preview = pd.read_csv(uploaded_file)
        st.markdown("---")
        st.subheader("Visualização do Arquivo Carregado")
        st.write("**Conteúdo completo do arquivo CSV:**")
        st.dataframe(df_preview)
        st.write("**Informações das Colunas:**")
        info_df = pd.DataFrame({
            "Coluna": df_preview.columns,
            "Tipo de Dado": df_preview.dtypes.astype(str),
            "Valores Não Nulos": df_preview.count().values
        })
        st.dataframe(info_df)
    except Exception as e:
        st.error(f"Ocorreu um erro ao tentar ler e pré-visualizar o arquivo CSV: {e}")

# --- 5. Seção de Interação com o Agente ---
st.markdown("---")
st.subheader("2. Faça sua Pergunta ao Agente")
question = st.text_input(
    "Digite sua pergunta sobre os dados:",
    placeholder="Ex: Qual a correlação entre a coluna 'Time' e 'Amount'?"
)

if st.button("Analisar Dados", type="primary"):
    if uploaded_file is not None and question:
        if not API_KEY:
            st.error("Chave da API do Google não encontrada! Por favor, configure os 'Secrets' ou seu arquivo .env.")
        else:
            with st.spinner("O Gemini está pensando e analisando os dados... Isso pode levar um momento. 🧠"):
                try:
                    uploaded_file.seek(0)
                    df_for_agent = pd.read_csv(uploaded_file)

                    # Inicializa o LLM com o modelo mais recente do Gemini
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-1.5-pro-latest", # <--- MUDANÇA APLICADA AQUI
                        temperature=0,
                        google_api_key=API_KEY,
                        convert_system_message_to_human=True
                    )

                    # Cria a instância do Agente Pandas do LangChain
                    agent = create_pandas_dataframe_agent(
                        llm=llm,
                        df=df_for_agent,
                        verbose=True,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                        handle_parsing_errors=True,
                        allow_dangerous_code=True
                    )

                    # Cria um prompt detalhado para guiar o agente
                    prompt = f"""
                    Sua tarefa é atuar como um analista de dados sênior.
                    Use o dataframe fornecido para responder à seguinte pergunta: '{question}'.
                    Pense passo a passo. Execute o código Python necessário para encontrar a resposta.
                    Forneça uma conclusão final clara, concisa e em português.
                    """

                    # Executa o agente com o prompt.
                    response = agent.invoke(prompt)

                    # Exibe a resposta final para o usuário
                    st.success("Análise Concluída!")
                    st.markdown("### Resposta do Agente:")
                    st.write(response["output"])

                except Exception as e:
                    st.error(f"Ocorreu um erro durante a análise: {e}")
    else:
        st.warning("Por favor, faça o upload de um arquivo CSV e digite uma pergunta antes de analisar.")