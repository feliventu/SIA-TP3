import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, learning_curve
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
    
    # A Perceptrón Lineal (Regresión Lineal)
    linear_model = LinearRegression()
    linear_model.fit(X, y)
    linear_preds = linear_model.predict(X) 
    linear_preds_clipped = np.clip(linear_preds, 0, 1)
    
    # B Perceptrón No Lineal (Regresión Logística / Sigmoide)
    non_linear_model = LogisticRegression(solver='liblinear')
    non_linear_model.fit(X, y)
    non_linear_probs = non_linear_model.predict_proba(X)[:, 1] 
    
    # Cálculos de Error
    mse_linear = np.mean((y - linear_preds_clipped)**2)
    mse_non_linear = np.mean((y - non_linear_probs)**2)
    
    print(f"MSE Perceptrón Lineal: {mse_linear:.4f}")
    print(f"MSE Perceptrón No Lineal: {mse_non_linear:.4f}")
    
    # === NUEVO: GRÁFICO DE COMPARACIÓN ===
    plt.figure(figsize=(10, 6))
    
    # Obtenemos la combinación lineal (z) para usarla como eje X continuo
    z_values = non_linear_model.decision_function(X)
    
    # Ordenamos los valores para que las líneas se dibujen de izquierda a derecha correctamente
    sorted_indices = np.argsort(z_values)
    z_sorted = z_values[sorted_indices]
    linear_preds_sorted = linear_preds[sorted_indices]
    non_linear_probs_sorted = non_linear_probs[sorted_indices]
    
    # Manejo de 'y' dependiendo de si es un DataFrame/Series de Pandas o un array de Numpy
    if isinstance(y, pd.Series):
        y_sorted = y.values[sorted_indices]
    else:
        y_sorted = y[sorted_indices]

    # Graficamos los datos reales (Cruces grises en Y=0 o Y=1)
    plt.scatter(z_sorted, y_sorted, color='gray', alpha=0.3, label='Datos Reales (0 o 1)', marker='x')
    
    # Graficamos la predicción del Lineal (Recta roja)
    plt.plot(z_sorted, linear_preds_sorted, color='red', linestyle='--', linewidth=2, label='Predicción Lineal')
    
    # Graficamos la predicción del No Lineal (Curva azul Sigmoide)
    plt.plot(z_sorted, non_linear_probs_sorted, color='blue', linewidth=3, label='Predicción No Lineal (Sigmoide)')

    # Líneas guía visuales en 0 y 1
    plt.axhline(1, color='black', linestyle=':', alpha=0.5)
    plt.axhline(0, color='black', linestyle=':', alpha=0.5)
    
    plt.title("Comparación de Aprendizaje: Perceptrón Lineal vs No Lineal")
    plt.xlabel("Combinación de Variables (Valor $z$)")
    plt.ylabel("Predicción del Modelo")
    plt.legend()
    plt.grid(True)
    plt.savefig("comparacion_aprendizaje.png")
    
    return non_linear_model

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
    plt.savefig("curva_precision_recall.png")
    
    # Recomendación lógica: En fraude, solemos priorizar Recall.
    # Buscamos un umbral donde el Recall sea alto (ej. > 0.8) sin destruir la precisión.
    idx = np.where(recall >= 0.8)[0][-1] 
    best_t = thresholds[idx]
    
    print(f"\nRecomendación:")
    print(f"Se recomienda un umbral de: {best_t:.2f}")
    print(f"Esto permite capturar al menos el 80% de los fraudes.")
    
def plot_saturation_curve(X, y):
    print("--- Generando Gráfico de Saturación (Curva de Aprendizaje) ---")
    
    # Usamos el Perceptrón No Lineal (Regresión Logística)
    model = LogisticRegression(solver='liblinear')
    
    # Calculamos la curva de aprendizaje
    # cv=5 hace validación cruzada, train_sizes divide los datos en incrementos
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy' # Puedes usar 'f1' o 'neg_mean_squared_error' también
    )
    
    # Calculamos los promedios y desviaciones estándar
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    # Graficamos
    plt.figure(figsize=(8, 5))
    plt.title("Curva de Aprendizaje: Visualización de Saturación")
    plt.xlabel("Cantidad de datos de entrenamiento")
    plt.ylabel("Rendimiento (Exactitud)")
    
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Rendimiento en Entrenamiento")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Rendimiento en Prueba (Validación)")
    
    plt.legend(loc="best")
    plt.grid(True)
    plt.ylim(0.5, 1.05) # Ajusta estos límites según tus resultados reales
    plt.savefig("curva_aprendizaje.png")

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
        
        plot_saturation_curve(X, y)
        
    except FileNotFoundError:
        print("Error: No se encontró el archivo 'farud_dataset.csv'. Por favor asegúrate de que esté en la misma carpeta.")