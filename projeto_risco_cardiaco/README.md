# infnet_projetos_ml
Para projetos da pós graduação de Machine Learning da Infnet

# 🫀 Predição de Risco Cardíaco com Machine Learning

## 📋 Sobre o Projeto
Este projeto de Machine Learning tem como objetivo prever o risco de doenças cardiovasculares em pacientes utilizando algoritmos de classificação supervisionada. O foco principal da análise é lidar com os desafios reais do domínio médico, como **desbalanceamento de classes (80/20)** e o **custo assimétrico do erro** (onde falsos negativos custam vidas).

## 📊 O Dataset
* **Origem:** Heart Disease Data for Health Research (Kaggle).
* **Volume:** 10.000 instâncias clínicas reais.
* **Features:** 24 variáveis preditivas (após One-Hot Encoding), incluindo dados de idade, pressão arterial, níveis de Colesterol, PCR e hábitos de vida.
* **Variável-alvo:** `Heart Disease Status` (1 = Yes / 0 = No).

## 🛠️ Tecnologias e Algoritmos
* Python, Scikit-Learn, Pandas, Matplotlib, Seaborn.
* **Perceptron** (Baseline e Otimizado)
* **Decision Tree** (Árvore de Decisão com Pruning)
* **Random Forest** (Modelo Ensemble)
* Otimização via `GridSearchCV` (5-fold cross-validation).

## 📈 Principais Conclusões
1. **O Paradoxo da Acurácia:** O Random Forest apresentou a melhor acurácia sustentável (68%), mas falhou em identificar 85% dos doentes devido ao desbalanceamento. 
2. **Viabilidade Operacional:** O Perceptron Otimizado com ajuste de pesos (`class_weight='balanced'`), apesar da baixa acurácia global (29%), provou-se ser o modelo mais seguro e ético para um cenário de triagem médica inicial, conseguindo identificar **81% dos pacientes em risco**.
3. **Fatores de Risco:** A análise de *Feature Importance* consolidou o **BMI** (Índice de Massa Corporal) e o **CRP Level** (Proteína C-Reativa) como os marcadores não-lineares mais determinantes.
