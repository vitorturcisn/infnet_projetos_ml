# 🏥 Relatório Técnico de MLOps: Diagnóstico de Risco Cardíaco

## 1. Introdução
Este documento detalha a arquitetura, os experimentos e as conclusões obtidas durante o desenvolvimento do modelo de predição de risco cardíaco. O projeto foi estruturado seguindo os princípios de **MLOps (Machine Learning Operations)**, focando em reprodutibilidade, modularidade e rastreabilidade.

---

## 2. Arquitetura do Projeto
A solução foi desenvolvida utilizando uma abordagem modular em Python, organizada da seguinte forma:

* **Ingestão:** Scripts para consumo de dados brutos e conversão para formatos eficientes (Parquet).
* **Pré-processamento:** Pipeline customizável via arquivos YAML para limpeza, imputação e engenharia de atributos.
* **Tracking:** Uso do **MLflow** para registro de hiperparâmetros, métricas e versionamento de modelos.
* **Otimização:** Uso do **Optuna** para busca automatizada dos melhores hiperparâmetros (HPO).
* **Deploy:** Interface em **Streamlit** consumindo o modelo registrado no MLflow Model Registry.

---

## 3. Pipeline de Pré-processamento (V1)
O pipeline foi configurado para tratar os desafios detectados na Análise Exploratória de Dados (EDA):

* **Imputação Clínica:** Preenchimento de valores ausentes em exames (Colesterol, CRP, Homocisteína) utilizando a **Mediana**, preservando a tendência central sem introduzir outliers.
* **Tratamento Categórico:** Criação da categoria `"Not Specified"` para a coluna de consumo de álcool (devido ao alto volume de nulos) e `"Unknown"` para gênero e hábitos.
* **Engenharia de Atributos:**
    * `cholesterol_bmi_ratio`: Razão entre colesterol e IMC.
    * `age_x_bmi`: Interação polinomial para capturar o risco combinado de idade e obesidade.
    * `is_hypertensive` / `is_obese`: Flags binárias baseadas em protocolos clínicos (PA > 140, IMC > 30).
* **Normalização:** Uso do `StandardScaler` para garantir que todas as features numéricas contribuam equitativamente para o cálculo do gradiente nos modelos.

---

## 4. Resultados da Modelagem
O experimento utilizou uma **Arena de Modelos** com validação cruzada estratificada (5-Fold).

### 📈 Comparativo de Performance (ROC-AUC)
| Algoritmo | Baseline CV | Otimizado (Optuna) | Holdout (Teste Final) |
| :--- | :---: | :---: | :---: |
| **Logistic Regression** | **0.5302** | **0.5314** | **0.4870** |
| Random Forest | 0.5159 | 0.5167 | - |
| XGBoost | 0.5102 | 0.5177 | - |
| LightGBM | 0.5127 | 0.5140 | - |

**Melhor Modelo Selecionado:** Regressão Logística (V1).

---

## 5. Análise de Robustez e Conclusões

### O Teto de Performance (AUC 0.53)
A performance observada (ROC-AUC ~0.53) indica que o modelo está operando pouco acima do limite do acaso (0.50). Após múltiplos ciclos de ajuste (V1 a V5), o diagnóstico técnico aponta para:
1.  **Baixa Relação Sinal-Ruído:** As variáveis independentes fornecidas pelo dataset (Kaggle) não possuem correlação estatística forte com o alvo clínico.
2.  **Paradoxos nos Dados:** Foram detectados pacientes com indicadores críticos marcados como saudáveis, sugerindo um dataset com alta variância inexplicada ou origem sintética.

### Valor de Engenharia vs. Valor Científico
Embora a acurácia diagnóstica seja limitada pelo dataset, o projeto atingiu **sucesso total nos requisitos de engenharia**:
* **Pipeline Automático:** O sistema é capaz de re-treinar, otimizar e registrar uma nova versão do modelo em menos de 5 minutos.
* **Versionamento:** Cada tentativa é rastreada no MLflow, permitindo auditoria de qual configuração de pré-processamento foi utilizada.
* **Prontidão para Produção:** A infraestrutura permite que, ao substituir o arquivo de dados por um dataset real (com sinal médico preservado), o modelo atinja alta performance sem alterações no código fonte.

---

## 6. Como Reproduzir
1.  Ativar o ambiente virtual: `.\venv\Scripts\Activate.ps1`
2.  Rodar o pré-processamento: `python notebooks/preprocessamento.py`
3.  Rodar a modelagem: `python notebooks/modelagem.py`
4.  Visualizar no MLflow: `mlflow ui`
5.  Iniciar o App: `streamlit run production_app/app.py`

---
*Relatório finalizado em 14 de Abril de 2026 — Projeto INFNET MLOps.*