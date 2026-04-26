import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CARGA Y PREPROCESAMIENTO
# ==========================================

def load_and_prepare_data(filepath):
    """Carga el CSV y realiza un preprocesamiento básico."""
    df = pd.read_csv(filepath)
    
    # Asumimos que la última columna es 'target' (fraude: 1, no fraude: 0)
    # y las demás son características (features)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    
    # Es una buena práctica escalar los datos para modelos basados en gradiente/perceptrones
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

# ==========================================
# 2. ESTUDIO DE APRENDIZAJE (Todo el dataset)
# ==========================================

def study_learning(X, y):
    print("--- Iniciando Estudio de Aprendizaje ---")
    
    # A) Perceptrón Lineal (Regresión Lineal)
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    linear_preds = linear_model.predict(X) 
    # El lineal puede dar valores < 0 o > 1, los truncamos para comparar "probabilidades"
    linear_preds_clipped = np.clip(linear_preds, 0, 1)
    
    # B) Perceptrón No Lineal (Regresión Logística / Sigmoide)
    # Usamos LogisticRegression que es un perceptrón con activación sigmoide
    non_linear_model = LogisticRegression(solver='liblinear')
    non_linear_model.fit(X, y)
    non_linear_probs = non_linear_model.predict_proba(X)[:, 1] # Probabilidades de clase 1
    
    # Comparación visual simple (Error Cuadrático Medio como proxy de aprendizaje)
    mse_linear = np.mean((y - linear_preds_clipped)**2)
    mse_non_linear = np.mean((y - non_linear_probs)**2)
    
    print(f"MSE Perceptrón Lineal: {mse_linear:.4f}")
    print(f"MSE Perceptrón No Lineal: {mse_non_linear:.4f}")
    
    return non_linear_model # Retornamos el seleccionado para el siguiente paso

# ==========================================
# 3. ESTUDIO DE GENERALIZACIÓN
# ==========================================

def study_generalization(X, y):
    print("\n--- Iniciando Estudio de Generalización ---")
    
    # b) Estrategia: División Entrenamiento/Prueba con Estratificación
    # Esto asegura que la proporción de fraudes sea igual en ambos sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = LogisticRegression(solver='liblinear')
    model.fit(X_train, y_train)
    
    # Obtener probabilidades para el análisis de umbral
    probs_test = model.predict_proba(X_test)[:, 1]
    
    # a) Métricas de evaluación
    # Usamos un umbral estándar de 0.5 para el reporte inicial
    y_pred_default = (probs_test >= 0.5).astype(int)
    
    print(f"Métricas con umbral 0.5:")
    print(f"- Precision: {precision_score(y_test, y_pred_default):.4f}")
    print(f"- Recall (Exhaustividad): {recall_score(y_test, y_pred_default):.4f}")
    print(f"- F1-Score: {f1_score(y_test, y_pred_default):.4f}")
    
    return X_test, y_test, probs_test

# ==========================================
# 4. RECOMENDACIÓN DE UMBRAL (Threshold)
# ==========================================

def recommend_threshold(y_test, probs_test):
    """Analiza la curva Precision-Recall para recomendar un umbral."""
    precision, recall, thresholds = precision_recall_curve(y_test, probs_test)
    
    # Graficar para análisis visual
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, precision[:-1], 'b--', label='Precisión')
    plt.plot(thresholds, recall[:-1], 'g-', label='Recall (Sensibilidad)')
    plt.xlabel('Umbral (Threshold)')
    plt.title('Análisis de Umbral para Detección de Fraude')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Recomendación lógica: En fraude, solemos priorizar Recall.
    # Buscamos un umbral donde el Recall sea alto (ej. > 0.8) sin destruir la precisión.
    idx = np.where(recall >= 0.8)[0][-1] 
    best_t = thresholds[idx]
    
    print(f"\nRecomendación:")
    print(f"Se recomienda un umbral de: {best_t:.2f}")
    print(f"Esto permite capturar al menos el 80% de los fraudes.")

# ==========================================
# EJECUCIÓN PRINCIPAL
# ==========================================

if __name__ == "__main__":
    try:
        # 1. Preparación
        X, y = load_and_prepare_data('fraud_dataset.csv')
        
        # 2. Aprendizaje
        # Aquí responderías las preguntas a, b, c de la primera parte
        best_model_arch = study_learning(X, y)
        
        # 3. Generalización
        X_test, y_test, probs = study_generalization(X, y)
        
        # 4. Umbral
        recommend_threshold(y_test, probs)
        
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'farud_dataset.csv'. Por favor asegúrate de que esté en la misma carpeta.")