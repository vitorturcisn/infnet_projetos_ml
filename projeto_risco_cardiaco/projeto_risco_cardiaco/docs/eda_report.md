# Cardiac Risk Prediction — EDA Report

**Project:** Cardiac Risk Prediction (MLOps Pipeline)
**Dataset:** Kaggle `oktayrdeki/heart-disease`
**Analysis Date:** 2026-04-12
**Pipeline:** `eda/run_eda.py` — 7 modules, 16.4s total
**Outputs:** 30 stats files · 10 tables · 15 figures

---

## Table of Contents

1. [Dataset Overview](#1-dataset-overview)
2. [Descriptive Statistics](#2-descriptive-statistics)
3. [Missing Values & Data Quality](#3-missing-values--data-quality)
4. [Target Variable Analysis](#4-target-variable-analysis)
5. [Feature Distributions](#5-feature-distributions)
6. [Correlation Analysis](#6-correlation-analysis)
7. [Risk Profile Analysis](#7-risk-profile-analysis)
8. [Pivot Tables & Cross-tabulations](#8-pivot-tables--cross-tabulations)
9. [Statistical Tests](#9-statistical-tests)
10. [Interaction Effects](#10-interaction-effects)
11. [Feature Engineering](#11-feature-engineering)
12. [Clustering Analysis](#12-clustering-analysis)
13. [Key Findings Summary](#13-key-findings-summary)
14. [Modeling Recommendations](#14-modeling-recommendations)

---

## 1. Dataset Overview

| Property | Value |
|---|---|
| Rows | 10,000 |
| Columns | 21 |
| Memory usage | 1.98 MB |
| Missing values | 219 total (distributed across 9 columns) |
| Target column | `Heart Disease Status` |
| Categorical columns | 12 |
| Numeric columns | 9 |

**Key Column descriptions:**
- `Age`: Patient age (18 to 80 years)
- `Blood Pressure`: Systolic blood pressure (Mean: 149.7)
- `Cholesterol Level`: Total cholesterol (Mean: 225.4)
- `BMI`: Body Mass Index (Mean: 29.1)
- `Heart Disease Status`: **TARGET** — Clinical diagnosis (Yes/No)

---

## 2. Descriptive Statistics

### Key Numeric Statistics

| Feature | Mean | Median | Std | Min | Max | Skewness | Distribution |
|---|---|---|---|---|---|---|---|
| `Age` | 49.3 yr | 49.0 yr | 18.2 | 18 | 80 | -0.007 | Normal |
| `Blood Pressure` | 149.8 | 150.0 | 17.6 | 120 | 180 | 0.014 | Normal |
| `Cholesterol Level` | 225.4 | 226.0 | 43.6 | 150 | 300 | -0.007 | Normal |
| `BMI` | 29.1 | 29.1 | 6.3 | 18.0 | 40.0 | -0.021 | Normal |
| `Sleep Hours` | 7.0 hr | 7.0 hr | 1.8 | 4.0 | 10.0 | 0.000 | Normal |
| `Triglyceride Level` | 250.7 | 250.0 | 87.1 | 100 | 400 | 0.006 | Normal |

**Observação:** Todas as variáveis numéricas apresentam **Skewness próxima de zero**, indicando distribuições simétricas. Diferente de outros datasets médicos, não há necessidade de transformações logarítmicas iniciais.

---

## 3. Missing Values & Data Quality

| Issue | Detail | Recommendation |
|---|---|---|
| **High Missingness** | `Alcohol Consumption`: 2,586 nulos (**25.8%**) | Criar categoria "Unknown" ou avaliar exclusão |
| **Numerical Nulls** | 219 linhas afetadas nas colunas clínicas (~2.2%) | Imputação por mediana |
| **Data Integrity** | Colunas de exames (CRP, Homocysteine) com ~0.2% | Seguro para imputação simples |

> **Aviso de MLOps:** A coluna `Alcohol Consumption` é a mais problemática. Como 1/4 dos dados estão faltando, imputar a "moda" (o valor mais frequente) pode criar um viés falso. A melhor estratégia é tratar o nulo como uma nova categoria: **"Not Specified"**.

---

## 4. Target Variable Analysis

### Distribution Shape (fig_01)
- **Frequência "No":** ~8,000 (80%)
- **Frequência "Yes":** ~2,000 (20%)
- **Status:** Dataset Desbalanceado (Proporção 4:1).

### Análise de Tendência por Idade (fig_12)
A suavização **LOWESS** revela que a probabilidade de doença cardíaca é relativamente estável, flutuando entre 18% e 21% ao longo da vida (18-80 anos). Existe uma leve queda por volta dos 50 anos, seguida de retomada. **Conclusão:** A idade isolada é um preditor fraco.

---

## 5. Feature Distributions

Todas as variáveis numéricas apresentam formato **Uniforme / Normal** com Skewness $< |0.1|$. 
**Análise Visual (fig_03):** As curvas KDE são planas, sugerindo cobertura homogênea de toda a faixa clínica. O pré-processamento focará em padronização (**StandardScaler**).

---

## 6. Correlation Analysis (fig_05)

| Feature | Pearson r | Interpretation |
|---|---|---|
| `BMI` | **+0.02** | Correlação positiva desprezível |
| `Age` | **-0.01** | Correlação negativa desprezível |
| `Stress Level_encoded` | **-0.028** | Maior rank relativo de sinal |

**Insight:** A ausência de correlações lineares fortes ($r < 0.1$) indica que o risco cardíaco neste dataset não é linear e depende de interações complexas que modelos baseados em árvores (XGBoost) capturam melhor.

---

## 7. Risk Profile Analysis

### Mean Risk by Gender and Age Category

| Category | Jovem | Adulto | Meia-Idade | Sênior |
|---|---|---|---|---|
| **Female** | 20.18% | **23.17%** | 19.02% | 20.37% |
| **Male** | 18.86% | 20.57% | 19.04% | 19.11% |

---

## 8. Pivot Tables & Cross-tabulations

### Exercise Habits × Smoking
- O grupo de pacientes com **Exercício Médio que Fumam** apresenta a maior prevalência de risco (**21.58%**).
- No grupo "Low Exercise", não-fumantes registraram risco levemente superior (20.6%).

### Análise de Casos Extremos (Top 10 Pacientes Críticos)
Filtramos pacientes com exames alarmantes (Colesterol 300, Hipertensão grave). 
**O Paradoxo:** Todos os 10 casos extremos possuem status "No". Isso sugere que indicadores clínicos máximos não garantem diagnóstico positivo neste dataset, reforçando a necessidade de análise multivariada.

---

## 9. Statistical Tests

### Normality & ANOVA
- **Shapiro-Wilk:** Todas as variáveis falharam na normalidade ($p < 0.00001$), exigindo modelos não-paramétricos.
- **ANOVA:** Apenas o **IMC (BMI)** apresentou diferença de média estatisticamente significativa ($p=0.049$). Idade e Pressão Arterial falharam isoladamente.
- **Chi-Squared:** Fumo e Diabetes não apresentaram associação independente com o diagnóstico ($p > 0.05$).

---

## 10. Interaction Effects

### 2-Way Interaction: Age × BMI (fig_16)
- **Zona Crítica:** Pacientes entre 34-49 anos com IMC extremo (>34.5) possuem o maior risco global (**24.83%**).
- **Paradoxo Sênior:** Em idosos (65-80), o IMC extremo apresentou risco menor que o IMC baixo, sugerindo outros fatores protetores.

### 3-Way Interaction: Age × Smoking × Gender (fig_21)
- **Mulheres:** Apresentam picos de risco voláteis, especialmente em jovens fumantes.
- **Homens:** Taxa de risco estável em ~20% em todas as faixas.
- **Limpeza:** Detectados registros com `Gender` vazio que precisam de tratamento.

---

## 11. Feature Engineering

| Feature | Rank | Correlation | Interpretation |
|---|---|---|---|
| `Stress Level_encoded` | 1º | -0.0283 | Variável "silenciosa" mais forte |
| `BMI` | 2º | +0.0196 | Principal sinal clínico |
| `cholesterol_bmi_ratio`| 3º | -0.0152 | Superou variáveis originais isoladas |

---

## 12. Clustering Analysis

### K-Means & PCA (fig_32)
- **Silhouette Score:** 0.087 (muito baixo), indicando que os pacientes formam uma massa contínua sem divisões rígidas.
- **Visualização PCA:** Os clusters (0: Roxo, 1: Verde, 2: Amarelo) mostram sobreposição sistêmica. O Cluster 2 captura as maiores variações (IMC/Estresse).

---

## 13. Key Findings Summary

1. **O Poder do IMC:** Única variável com significância individual confirmada ($p=0.049$).
2. **Não-Linearidade:** A saúde dos pacientes é um espectro contínuo; o risco vem da soma de fatores "silenciosos".
3. **Ponto Crítico:** A combinação **Meia-Idade + Obesidade Classe II/III** é o maior alerta detectado.
4. **Estresse:** O nível de estresse superou o impacto individual de fumo e diabetes neste dataset.

---

## 14. Modeling Recommendations

1. **Algoritmos:** Utilizar **XGBoost ou LightGBM** com `max_depth` suficiente para capturar interações (especialmente `Age x BMI`).
2. **Métricas:** Focar em **Recall** e **Probability Calibration** (fornecer % de risco ao médico).
3. **Pré-processamento:** Imputar nulos por **Mediana** e aplicar **StandardScaler**.
4. **Dados:** Tratar valores vazios na coluna `Gender` detectados na análise de 3-vias.

---
*Report generated by `eda/run_eda.py` — Cardiac Risk EDA Pipeline v1.0.0*