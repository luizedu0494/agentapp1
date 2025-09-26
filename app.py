# -----------------------------------------------------------------
# Agente de Análise de Dados com Streamlit e LangChain
#
# Autor: [Seu Nome]
# Data: 26 de Setembro de 2025
#
# Descrição:
# Esta aplicação permite que os usuários façam o upload de um arquivo CSV,
# visualizem seu conteúdo e façam perguntas em linguagem natural sobre os dados.
# Um agente de IA (construído com LangChain) analisa os dados e fornece
# respostas e conclusões.
# -----------------------------------------------------------------

# --- 1. Importação das Bibliotecas Necessárias ---
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

# Importações específicas do LangChain
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType
from langchain_experimental.agents import create_pandas_dataframe_agent

# --- 2. Configuração da Página e Chaves de API ---

# Configura o layout da página do Streamlit.
# O layout "wide" oferece mais espaço para tabelas e conteúdo.
st.set_page_config(
    page_title="Agente de Análise de Dados",
    page_icon="🤖",
    layout="wide"
)

# Carrega as variáveis de ambiente de um arquivo .env (útil para desenvolvimento local)
load_dotenv()

# Busca a chave da API. Prioriza os "Secrets" do Streamlit Cloud,
# mas usa as variáveis de ambiente como alternativa para rodar localmente.
# NOTA: Substitua 'DEEPSEEK_API_KEY' se estiver usando outra API, como 'GROQ_API_KEY'.
API_KEY = st.secrets.get("DEEPSEEK_API_KEY", os.getenv("DEEPSEEK_API_KEY"))
API_BASE_URL = "https://api.deepseek.com/v1" # Mude se usar outra API
LLM_MODEL = "deepseek-chat" # Mude se usar outra API

# --- 3. Interface do Usuário (UI ) com Streamlit ---

# Título e descrição da aplicação
st.title("🤖 Agente de Análise de Dados")
st.markdown(
    "Faça o upload de um arquivo CSV, visualize os dados e faça perguntas em linguagem natural. "
    "O agente irá analisar o arquivo e gerar uma resposta para você."
)

# Widget para o usuário fazer o upload do arquivo CSV
uploaded_file = st.file_uploader(
    "**1. Escolha um arquivo CSV**",
    type=["csv"]
)

# --- 4. Funcionalidade de Pré-visualização dos Dados ---

# Este bloco é executado assim que um arquivo é carregado pelo usuário
if uploaded_file is not None:
    try:
        # Carrega o arquivo CSV em um dataframe do Pandas para a pré-visualização
        df_preview = pd.read_csv(uploaded_file)

        st.markdown("---")
        st.subheader("Visualização do Arquivo Carregado")

        # Exibe o dataframe completo em uma tabela com barra de rolagem
        st.write("**Conteúdo completo do arquivo CSV:**")
        st.dataframe(df_preview)

        # Exibe informações úteis sobre as colunas do dataframe
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

# Campo de texto para o usuário digitar a pergunta
question = st.text_input(
    "Digite sua pergunta sobre os dados:",
    placeholder="Ex: Qual a correlação entre a coluna 'Time' e 'Amount'?"
)

# Botão para iniciar a análise. A lógica principal é executada quando este botão é clicado.
if st.button("Analisar Dados", type="primary"):
    # Verifica se o arquivo foi carregado e se uma pergunta foi feita
    if uploaded_file is not None and question:
        # Verifica se a chave da API foi configurada corretamente
        if not API_KEY:
            st.error("Chave da API não encontrada! Por favor, configure os 'Secrets' no painel da Streamlit Cloud.")
        else:
            # Mostra uma mensagem de "carregando" enquanto o agente trabalha
            with st.spinner("O agente está pensando e analisando os dados... Isso pode levar um momento. 🧠"):
                try:
                    # Importante: "Rebobina" o arquivo para o início antes de lê-lo novamente.
                    # Necessário porque ele já foi lido na etapa de pré-visualização.
                    uploaded_file.seek(0)
                    df_for_agent = pd.read_csv(uploaded_file)

                    # Inicializa o Modelo de Linguagem (LLM) que o agente usará
                    llm = ChatOpenAI(
                        model=LLM_MODEL,
                        temperature=0,
                        api_key=API_KEY,
                        base_url=API_BASE_URL
                    )

                    # Cria a instância do Agente Pandas do LangChain
                    agent = create_pandas_dataframe_agent(
                        llm=llm,
                        df=df_for_agent,
                        verbose=True, # Imprime o "raciocínio" do agente nos logs do servidor
                        agent_type=AgentType.OPENAI_FUNCTIONS,
                        handle_parsing_errors=True,
                        allow_dangerous_code=True # Permite que o agente execute código Python
                    )

                    # Cria um prompt detalhado para guiar o agente
                    prompt = f"""
                    Sua tarefa é atuar como um analista de dados sênior.
                    Use o dataframe fornecido para responder à seguinte pergunta: '{question}'.
                    Pense passo a passo. Execute o código Python necessário para encontrar a resposta.
                    Forneça uma conclusão final clara, concisa e em português.
                    """

                    # Executa o agente com o prompt.
                    # agent.invoke() é usado em vez de ainvoike() para compatibilidade com o Streamlit.
                    response = agent.invoke(prompt)

                    # Exibe a resposta final para o usuário
                    st.success("Análise Concluída!")
                    st.markdown("### Resposta do Agente:")
                    st.write(response["output"])

                except Exception as e:
                    # Captura e exibe qualquer erro que ocorra durante a análise
                    st.error(f"Ocorreu um erro durante a análise: {e}")
    else:
        # Avisa o usuário se o arquivo ou a pergunta estiverem faltando
        st.warning("Por favor, faça o upload de um arquivo CSV e digite uma pergunta antes de analisar.")
