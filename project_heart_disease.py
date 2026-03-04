# -*- coding: utf-8 -*-
"""project_heart_disease.ipynb"""

import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import RobustScaler, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# Configurações iniciais
warnings.filterwarnings('ignore')
RANDOM_STATE = 13

# ==========================================
# 1. CARREGAMENTO DOS DADOS E EDA
# ==========================================
data = pd.read_csv('heart_disease.csv')

# Inspeção inicial
display(data.info())
display(data.isna().sum())

# Análise de nulos (Alcohol Consumption)
print("Contagem por categoria de Álcool:")
display(data['Alcohol Consumption'].value_counts(dropna=False))

print("\nDistribuição Percentual:")
display(data['Alcohol Consumption'].value_counts(dropna=False, normalize=True) * 100)

plt.figure(figsize=(10, 6))
data['Alcohol Consumption'].value_counts(dropna=False).plot(kind='bar', color='skyblue')
plt.title('Distribuição do Consumo de Álcool (incluindo nulos)')
plt.xlabel('Nível de Consumo')
plt.ylabel('Quantidade de Pacientes')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Cálculo de perda potencial de dados
linhas_com_algum_nulo = data.isna().any(axis=1).sum()
total_linhas = len(data)
linhas_restantes = total_linhas - linhas_com_algum_nulo

print(f"Total de linhas original: {total_linhas}")
print(f"Linhas com pelo menos um valor nulo: {linhas_com_algum_nulo}")
print(f"Linhas restantes após drop total: {linhas_restantes}")
print(f"Porcentagem de perda: {(linhas_com_algum_nulo/total_linhas)*100:.2f}%\n")

# Distribuição de variáveis numéricas e outliers
numeric_cols = data.select_dtypes(include=['float64']).columns

plt.figure(figsize=(15, 20))
for i, col in enumerate(numeric_cols):
    plt.subplot(len(numeric_cols), 2, 2*i + 1)
    sns.histplot(data[col], kde=True, color='teal')
    plt.title(f'Distribuição: {col} (Skew: {data[col].skew():.2f})')

    plt.subplot(len(numeric_cols), 2, 2*i + 2)
    sns.boxplot(x=data[col], color='lightcoral')
    plt.title(f'Outliers: {col}')

plt.tight_layout()
plt.show()

# Distribuição da Variável Alvo
print("Contagem de Heart Disease Status:")
print(data['Heart Disease Status'].value_counts())
print("\nProporção Percentual:")
print(data['Heart Disease Status'].value_counts(normalize=True) * 100)

plt.figure(figsize=(8, 5))
sns.countplot(x='Heart Disease Status', data=data, palette='viridis')
plt.title('Distribuição da Variável Alvo (Target)')
plt.show()

# ==========================================
# 2. PRÉ-PROCESSAMENTO
# ==========================================
y = data['Heart Disease Status'].map({'Yes': 1, 'No': 0})
X = data.drop(columns=['Heart Disease Status'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

num_features = X.select_dtypes(include=['float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object']).columns.tolist()

# Transformadores padrão (StandardScaler)
num_transformer_std = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor_std = ColumnTransformer(transformers=[
    ('num', num_transformer_std, num_features),
    ('cat', cat_transformer, cat_features)
])

# ==========================================
# 3. MODELO BASELINE (PERCEPTRON ORIGINAL)
# ==========================================
baseline_model = Pipeline(steps=[
    ('preprocessor', preprocessor_std),
    ('classifier', Perceptron(random_state=RANDOM_STATE))
])

baseline_model.fit(X_train, y_train)
y_pred = baseline_model.predict(X_test)

print("--- Relatório de Classificação - Perceptron Baseline ---")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Disease', 'Heart Disease'])
disp.plot(cmap='Blues')
plt.title('Matriz de Confusão - Perceptron Baseline')
plt.show()

# ==========================================
# 4. PERCEPTRON OTIMIZADO
# ==========================================
# Transformador numérico com RobustScaler
num_transformer_rob = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

preprocessor_rob = ColumnTransformer(transformers=[
    ('num', num_transformer_rob, num_features),
    ('cat', cat_transformer, cat_features)
])

optimized_perceptron = Pipeline(steps=[
    ('preprocessor', preprocessor_rob),
    ('classifier', Perceptron(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        max_iter=1000,
        tol=1e-3
    ))
])

optimized_perceptron.fit(X_train, y_train)
y_pred_opt = optimized_perceptron.predict(X_test)

print("--- Relatório de Classificação - Perceptron Otimizado ---")
print(classification_report(y_test, y_pred_opt))

plt.figure(figsize=(8, 6))
cm_opt = confusion_matrix(y_test, y_pred_opt)
disp_opt = ConfusionMatrixDisplay(confusion_matrix=cm_opt, display_labels=['No Disease', 'Heart Disease'])
disp_opt.plot(cmap='Blues', values_format='d')
plt.title('Matriz de Confusão - Perceptron Otimizado')
plt.show()

# Extração de Pesos do Perceptron Otimizado
pesos = optimized_perceptron.named_steps['classifier'].coef_[0]
bias = optimized_perceptron.named_steps['classifier'].intercept_
feature_names = num_features + list(optimized_perceptron.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(cat_features))

df_pesos = pd.DataFrame({'Feature': feature_names, 'Peso': pesos})
print(f"\nBias do Perceptron: {bias}")
print(df_pesos.sort_values(by='Peso', ascending=False))

# ==========================================
# 5. ÁRVORE DE DECISÃO
# ==========================================
tree_model = Pipeline(steps=[
    ('preprocessor', preprocessor_rob),
    ('classifier', DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        max_depth=4
    ))
])

tree_model.fit(X_train, y_train)
y_pred_tree = tree_model.predict(X_test)

print("--- Relatório de Classificação - Árvore de Decisão ---")
print(classification_report(y_test, y_pred_tree))

plt.figure(figsize=(8, 6))
cm_tree = confusion_matrix(y_test, y_pred_tree)
disp_tree = ConfusionMatrixDisplay(confusion_matrix=cm_tree, display_labels=['No Disease', 'Heart Disease'])
disp_tree.plot(cmap='Greens', values_format='d')
plt.title('Matriz de Confusão - Árvore de Decisão')
plt.show()

# Visualização da estrutura da Árvore
plt.figure(figsize=(20, 10))
plot_tree(tree_model.named_steps['classifier'],
          feature_names=feature_names,
          class_names=['No', 'Yes'],
          filled=True, rounded=True, fontsize=10)
plt.title("Regras de Decisão da Árvore")
plt.show()

# ==========================================
# 6. OTIMIZAÇÃO DA ÁRVORE DE DECISÃO (GRID SEARCH)
# ==========================================
# Busca V1
param_grid = {
    'classifier__max_depth': [3, 4, 5, 6, 8, 10],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(
    tree_model,
    param_grid,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
print(f"Melhores Hiperparâmetros (V1): {grid_search.best_params_}")

y_pred_best = grid_search.best_estimator_.predict(X_test)
print("\n--- Relatório de Classificação - Árvore Otimizada (V1) ---")
print(classification_report(y_test, y_pred_best))

# Busca V2 (Expandida)
param_grid_expanded = {
    'classifier__criterion': ['gini', 'entropy'],
    'classifier__max_depth': [5, 6, 7, 8, 12],
    'classifier__min_samples_split': [10, 20, 50],
    'classifier__min_samples_leaf': [5, 10, 20],
    'classifier__max_features': [None, 'sqrt', 'log2'],
    'classifier__ccp_alpha': [0.0, 0.001, 0.01]
}

grid_search_v2 = GridSearchCV(
    tree_model,
    param_grid_expanded,
    cv=5,
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search_v2.fit(X_train, y_train)
print(f"Novos Melhores Hiperparâmetros (V2): {grid_search_v2.best_params_}")

y_pred_v2 = grid_search_v2.best_estimator_.predict(X_test)
print("\n--- Relatório de Classificação - Árvore Otimizada (V2) ---")
print(classification_report(y_test, y_pred_v2))

# ==========================================
# 7. RANDOM FOREST (ENSEMBLE)
# ==========================================
rf_model = Pipeline(steps=[
    ('preprocessor', preprocessor_rob),
    ('classifier', RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight='balanced',
        n_jobs=-1
    ))
])

param_grid_rf = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [8, 12, 15],
    'classifier__min_samples_leaf': [5, 10]
}

grid_rf = GridSearchCV(rf_model, param_grid_rf, cv=5, scoring='f1', verbose=1)
grid_rf.fit(X_train, y_train)

print(f"Melhores parâmetros RF: {grid_rf.best_params_}")
y_pred_rf = grid_rf.best_estimator_.predict(X_test)

print("\n--- Relatório de Classificação - RANDOM FOREST ---")
print(classification_report(y_test, y_pred_rf))

# ==========================================
# 8. FEATURE IMPORTANCE (RANDOM FOREST)
# ==========================================
importances = grid_rf.best_estimator_.named_steps['classifier'].feature_importances_

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(15), palette='magma')
plt.title('Top 15 Variáveis Mais Importantes - Random Forest')
plt.show()
