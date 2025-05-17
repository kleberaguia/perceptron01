import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Simulação de Dados IoT
def gerar_dados(num_pontos):
    np.random.seed(42)
    temperatura = np.random.uniform(20, 30, num_pontos)
    umidade = np.random.uniform(40, 80, num_pontos)
    # Criando um alvo simples baseado em temperatura e umidade
    alvo = np.where((temperatura > 25) & (umidade > 60), 1, 0) # 1 para "alerta", 0 para "normal"
    df = pd.DataFrame({'temperatura': temperatura, 'umidade': umidade, 'alvo': alvo})
    return df

# 2. "Big Data" Mockado
dados = gerar_dados(20) # Gerando um conjunto maior de dados para simular "Big Data"

# 3. Processamento com Perceptron
X = dados[['temperatura', 'umidade']]
y = dados['alvo']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

perceptron = Perceptron()
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
acuracia = accuracy_score(y_test, y_pred)

# Função para classificar um novo ponto usando o perceptron treinado
def classificar_ponto(temp, umid):
    predicao = perceptron.predict([[temp, umid]])[0]
    return "Alerta" if predicao == 1 else "Normal"

# 4. Frontend Streamlit
st.title("Sistema Simples Integrando IoT, Big Data e Perceptron")
st.subheader("Simulação de Dados de Sensores")
st.dataframe(dados.head())

st.subheader("Classificação com Perceptron")
st.write(f"Acurácia do Perceptron nos dados de teste: {acuracia:.2f}")

st.subheader("Classificar Novo Dado Simulado de Sensor")
temperatura_nova = st.slider("Temperatura:", 15.0, 35.0, 25.0)
umidade_nova = st.slider("Umidade:", 30.0, 90.0, 60.0)

if st.button("Classificar"):
    classificacao = classificar_ponto(temperatura_nova, umidade_nova)
    st.write(f"Para Temperatura = {temperatura_nova:.2f}°C e Umidade = {umidade_nova:.2f}%, a Classificação é: **{classificacao}**")

st.subheader("Visualização dos Dados e da Fronteira de Decisão (Simplificada)")
st.write("Este é um gráfico simplificado para ilustrar a separação feita pelo Perceptron.")

# Plot simples (requer matplotlib, que Streamlit geralmente tem)
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
scatter = plt.scatter(dados['temperatura'], dados['umidade'], c=dados['alvo'], cmap='viridis')
plt.xlabel("Temperatura")
plt.ylabel("Umidade")
plt.colorbar(scatter, label='Alvo (0: Normal, 1: Alerta)')

# Simplificando a visualização da fronteira (linear para o Perceptron)
w = perceptron.coef_[0]
b = perceptron.intercept_[0]
x_min, x_max = dados['temperatura'].min() - 1, dados['temperatura'].max() + 1
y_min, y_max = dados['umidade'].min() - 1, dados['umidade'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = perceptron.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

st.pyplot(plt)