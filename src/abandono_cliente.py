import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV



data = pd.read_csv(r'C:\Users\Joaco\Desktop\Proyectos_ML\Segundo proyecto_abandono_cliente\data\WA_Fn-UseC_-Telco-Customer-Churn.csv')
X = data.drop('Churn', axis=1)
y = data['Churn'].map({'No': 0, 'Yes': 1})

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())
])

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

proceso = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

modelo_lr = Pipeline([
    ('proceso', proceso),
    ('modelo_lr', LogisticRegression(max_iter=1000, C=0.1, random_state=42))
])

#----------Entrenamos y predecimos con LR para tener nocion de como se comporta el algoritmo---------
modelo_lr.fit(X_train, y_train)
y_val_pred = modelo_lr.predict(X_val)

f1 = f1_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)

print(f"F1-score (validation): {f1:.4f}")
print(f"Recall (validation): {recall:.4f}")

#---------Buscamos parámetros con GridSearchCV--------------
def evaluar_modelo(modelo, X, y, nombre="Modelo"):
    y_pred = modelo.predict(X)
    
    f1 = f1_score(y, y_pred)
    recall = recall_score(y, y_pred)
    
    print(f"\nResultados - {nombre}")
    print(f"F1-score:  {f1:.4f}")
    print(f"Recall:    {recall:.4f}")


param_grid = {
    "modelo_lr__C": [0.001, 0.01, 0.1, 1, 10, 100],
    "modelo_lr__penalty": ["l1", "l2"],
    "modelo_lr__class_weight": [None, "balanced"]
}

grid_search = GridSearchCV(modelo_lr, 
                           param_grid, 
                           cv=5, 
                           scoring="average_precision", 
                           n_jobs=-1)

grid_search.fit(X_train, y_train)

# LR base
evaluar_modelo(modelo_lr, X_val, y_val, "Logistic Regresion - Base")

# LR optimizado
best_lr = grid_search.best_estimator_
evaluar_modelo(best_lr, X_val, y_val, "Logistic Regresion - Optimizado")

y_proba_val = best_lr.predict_proba(X_val)[:, 1]
thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]

for t in thresholds:
    y_pred = (y_proba_val >= t).astype(int)
    f1 = f1_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    print(f"threshold={t} | F1={f1}")
    print(f"threshold={t} | Recall={recall}\n")


X_train_final = pd.concat([X_train, X_val])
y_train_final = pd.concat([y_train, y_val])

best_lr.fit(X_train_final, y_train_final)

threshold_final = 0.4

y_proba_test = best_lr.predict_proba(X_test)[:, 1]
y_test_pred = (y_proba_test >= threshold_final).astype(int)

f1_test = f1_score(y_test, y_test_pred)
recall_test = recall_score(y_test, y_test_pred)

print("\nResultados finales en TEST")
print(f"F1-score (test): {f1_test:.4f}")
print(f"Recall (test):   {recall_test:.4f}")
