# Predicción de Abandono de Clientes (Customer Churn) — Proyecto de Machine Learning

## 📌 Descripción general del proyecto

Este proyecto aborda el problema de **predicción de abandono de clientes (Customer Churn)** utilizando técnicas de **Machine Learning supervisado**.  
El objetivo es construir un modelo **realista, interpretable y correctamente evaluado**, siguiendo buenas prácticas de Ciencia de Datos aplicadas a datos reales.

El proyecto pone especial énfasis en:
- separación correcta de los datos
- métricas adecuadas para datasets desbalanceados
- ajuste de threshold
- comparación honesta de modelos
- reproducibilidad y claridad del proceso

---

## 🎯 Objetivo de negocio

El churn busca identificar clientes con alta probabilidad de abandonar un servicio.  
En este tipo de problemas, **no detectar un cliente que abandona (falso negativo)** suele ser más costoso que alertar a uno que no lo hará.

Por este motivo, se priorizan métricas como **recall** y **F1-score**, en lugar de accuracy.

---

## 📊 Dataset

**Telco Customer Churn Dataset**

- Variable objetivo binaria: `Churn`
- Desbalance moderado de clases
- Variables numéricas y categóricas

Codificación del target:
```python
y = data['Churn'].map({'No': 0, 'Yes': 1})

#----------------------------------------------------

🔀 División de los datos

Se aplicó una división estricta y correcta:

Entrenamiento (Train): 60%

Validación (Validation): 20%

Test: 20%

Utilizando stratify para preservar la proporción de clases.

👉 El conjunto test se utilizó una sola vez, al final del proyecto.

#----------------------------------------------------

🔧 Preprocesamiento

Se implementó un pipeline estructurado con ColumnTransformer.

Variables numéricas

Imputación por mediana

RobustScaler

Variables categóricas

Imputación por valor más frecuente

One-Hot Encoding (handle_unknown='ignore')

#----------------------------------------------------

🤖 Modelos evaluados
Regresión Logística (Modelo final)

Entrenamiento baseline

Optimización de hiperparámetros con GridSearchCV

Métrica de optimización: Average Precision (PR-AUC)

Ajuste manual del threshold usando validation

Random Forest (Modelo de comparación)

Evaluado como alternativa no lineal

Comportamiento excesivamente conservador

F1-score inestable tras el ajuste de threshold

Finalmente descartado

#----------------------------------------------------

⚙️ Ajuste de threshold

Dado que el modelo produce probabilidades, el threshold por defecto (0.5) no resultó óptimo.

Se evaluaron distintos valores sobre el conjunto de validación y se seleccionó:

Threshold final: 0.4

Este valor ofreció un mejor equilibrio entre recall y F1-score.

#----------------------------------------------------

🧪 Resultados finales en Test

El modelo final de Regresión Logística fue reentrenado utilizando train + validation, y evaluado una sola vez sobre test.

Resultados en test:

F1-score: 0.60

Recall: 0.64

La leve caída respecto a validation es esperable y confirma una buena capacidad de generalización, sin data leakage.

#----------------------------------------------------

📈 Visualización con PCA (Interpretabilidad)

Se aplicó PCA únicamente con fines exploratorios y de visualización, no para entrenar el modelo.

La proyección a 2 componentes muestra un alto solapamiento entre clientes que abandonan y los que no, lo que evidencia:

la complejidad del problema

la ausencia de una separación clara en baja dimensión

Este gráfico ayuda a interpretar por qué métricas moderadas son esperables en un problema real de churn.

#----------------------------------------------------
🧠 Conclusiones principales

La Regresión Logística demostró ser un modelo sólido e interpretable

El ajuste de threshold tuvo un impacto significativo en las métricas relevantes

Modelos más complejos no garantizaron mejores resultados

El solapamiento entre clases limita el desempeño máximo alcanzable

El proceso y las decisiones técnicas son tan importantes como la métrica final

#----------------------------------------------------

🛠️ Tecnologías utilizadas

Python

Pandas / NumPy

Scikit-learn

Matplotlib

#-----------------------------------------------------
👤 Autor

Joaquín Castro
Estudiante de Ciencia de Datos / Machine Learning


## 🧩 Comentario final

Este README:
- no exagera resultados  
- explica decisiones  
- muestra criterio real  
- **queda muy bien para GitHub y LinkedIn**

Cuando quieras, el próximo paso puede ser:
- texto corto para LinkedIn
- descripción para CV
- o planear el **siguiente proyecto** con otro enfoque

Este proyecto ya está **cerrado como corresponde** 👏
