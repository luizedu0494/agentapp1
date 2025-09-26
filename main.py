# main.py (versão corrigida)

import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
import uvicorn
import pandas as pd
from dotenv import load_dotenv

# Importações do LangChain CORRIGIDAS
from langchain_openai import ChatOpenAI
from langchain.agents import AgentType  # Importação corrigida
# Nova tentativa de importação
from langchain_experimental.agents import create_pandas_dataframe_agent


# Carrega as variáveis de ambiente
load_dotenv()

# Inicializa o aplicativo FastAPI
app = FastAPI(
    title="API do Agente de Análise de Dados",
    description="Uma API que usa LangChain e DeepSeek para analisar arquivos CSV.",
    version="1.0.0"
)

# Define o caminho onde o CSV será salvo temporariamente
CSV_FILE_PATH = "temp_data.csv"

@app.post("/analyze", summary="Analisa um arquivo CSV com uma pergunta")
async def analyze_data(
    question: str = Form(..., description="A pergunta do usuário sobre os dados."), 
    file: UploadFile = File(..., description="O arquivo CSV a ser analisado.")
):
    """
    Recebe um arquivo CSV e uma pergunta, usa um agente LangChain com DeepSeek 
    para analisar os dados e retorna a resposta.
    """
    try:
        # 1. Salva o arquivo CSV enviado no servidor
        with open(CSV_FILE_PATH, "wb") as buffer:
            buffer.write(await file.read())

        # 2. Carrega o CSV em um dataframe do Pandas
        df = pd.read_csv(CSV_FILE_PATH)

        # 3. Inicializa o LLM (usando DeepSeek)
        llm = ChatOpenAI(
            model="deepseek-chat", 
            temperature=0,
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com/v1" 
         )

        # 4. Cria o Agente Pandas DataFrame do LangChain
        agent = create_pandas_dataframe_agent(
            llm=llm,
            df=df,
            verbose=True, # Mostra o "raciocínio" do agente no terminal do backend
            agent_type=AgentType.OPENAI_FUNCTIONS,
            handle_parsing_errors=True,
        )

        # 5. Cria um prompt detalhado para guiar o agente
        prompt = f"""
        Sua tarefa é atuar como um analista de dados.
        Use o dataframe fornecido para responder à seguinte pergunta: '{question}'.
        
        - Pense passo a passo.
        - Execute o código Python para encontrar a resposta.
        - Forneça uma conclusão clara, concisa e em português.
        """
        
        # 6. Executa o agente com o prompt
        response = await agent.ainvoke(prompt)
        
        # 7. Retorna a resposta final do agente
        return JSONResponse(content={"answer": response["output"]})

    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Ocorreu um erro no servidor: {str(e)}"}
        )

# Permite executar a API diretamente com 'python main.py'
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
