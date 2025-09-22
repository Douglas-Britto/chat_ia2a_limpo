import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os

# ✅ Importação corrigida do agente
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.llms import OpenAI

# 🔹 Carregar chave da API
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# 🔹 Carregar CSV
caminho_csv = "creditcard.csv"
df = pd.read_csv(caminho_csv)

# 🔹 Criar agente inteligente com permissão para executar código
agent = create_pandas_dataframe_agent(
    OpenAI(temperature=0, openai_api_key=openai_api_key),
    df,
    verbose=False,
    allow_dangerous_code=True
)

# 🔹 Inicializar histórico
if "historico" not in st.session_state:
    st.session_state.historico = []

if "grupos_salvos" not in st.session_state:
    st.session_state.grupos_salvos = []

# 🔹 Interface principal
st.title("🧠 Agente IA2A — Inteligência sobre Dados")

pergunta = st.text_input("Digite sua pergunta sobre o conjunto de dados:")

# 🔹 Função para dividir perguntas complexas
def dividir_pergunta(pergunta):
    partes = []
    if "variabilidade" in pergunta.lower():
        partes.append("Qual o desvio padrão das variáveis?")
        partes.append("Qual a variância das variáveis?")
    elif "fraudes" in pergunta.lower() and "influência" in pergunta.lower():
        partes.append("Quais variáveis têm maior influência nas fraudes?")
        partes.append("Existe correlação entre tempo e valor nas transações fraudulentas?")
    else:
        partes.append(pergunta)
    return partes

# 🔹 Processar pergunta
if pergunta:
    partes = dividir_pergunta(pergunta)
    respostas = []

    for parte in partes:
        try:
            resposta = agent.run(parte)
        except Exception as e:
            resposta = f"Erro ao processar: {str(e)}"
        st.session_state.historico.append((parte, resposta))

        # 🔹 Exibir resposta
        st.subheader(f"💬 Resposta para: {parte}")
        st.write(resposta)

        # 🔹 Gerar gráfico se aplicável
        if "distribuição" in parte.lower() or "gráfico" in parte.lower():
            colunas_numericas = df.select_dtypes(include=["float64", "int64"]).columns
            for coluna in colunas_numericas:
                if coluna.lower() in parte.lower():
                    st.subheader(f"📊 Distribuição da variável: {coluna}")
                    fig, ax = plt.subplots()
                    sns.histplot(df[coluna], kde=True, ax=ax)
                    st.pyplot(fig)
                    break

    # 🔹 Mostrar histórico atual
    st.subheader("📁 Histórico atual")
    for i, (p, r) in enumerate(st.session_state.historico):
        st.markdown(f"**{i+1}. Pergunta:** {p}")
        st.markdown(f"**Resposta:** {r}")

    # 🔹 Salvar grupo e limpar histórico automaticamente
    st.session_state.grupos_salvos.append(st.session_state.historico.copy())
    st.session_state.historico = []
    st.success("✅ Grupo salvo e histórico limpo para próxima pergunta.")

# 🔹 Botão para gerar resumo final
if st.button("🧠 Gerar resumo final do agente"):
    resumo = "🔍 **Conclusões do agente com base nas perguntas feitas:**\n\n"
    for grupo in st.session_state.grupos_salvos:
        for pergunta, resposta in grupo:
            resumo += f"- **{pergunta}**\n  → {resposta}\n\n"
    st.subheader("🧠 Conclusão Final")
    st.text_area("Resumo gerado:", resumo, height=400)
