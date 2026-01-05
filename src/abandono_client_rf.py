import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


data = pd.read_csv(r'C:\Users\Joaco\Desktop\Proyectos_ML\Segundo proyecto_abandono_cliente\data\WA_Fn-UseC_-Telco-Customer-Churn.csv')
X = data.drop('Churn', axis=1)
y = data['Churn'].map({'No': 0, 'Yes': 1})

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_train.select_dtypes(include=['object']).columns

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

proceso_rf = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median'))]), num_cols),
    ('cat', cat_pipeline, cat_cols)
])

modelo_rf = Pipeline([
    ('proceso', proceso_rf),
    ('modelo_rf', RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ))
])

modelo_rf.fit(X_train, y_train)
y_val_pred = modelo_rf.predict(X_val)

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

param_grid_rf = {
    'modelo_rf__n_estimators': [100, 300, 500],
    'modelo_rf__max_depth': [None],
    'modelo_rf__min_samples_leaf': [5, 10, 20],
    'modelo_rf__class_weight': [None, 'balanced']
}

grid_search_rf = GridSearchCV(modelo_rf, 
                              param_grid_rf, 
                              cv=5, 
                              scoring="average_precision", 
                              n_jobs=-1
)

grid_search_rf.fit(X_train, y_train)

# Random base
evaluar_modelo(modelo_rf, X_val, y_val, "Random forest - Base")

# Random optimizado
best_lr = grid_search_rf.best_estimator_
evaluar_modelo(best_lr, X_val, y_val, "Random forest - Optimizado")


y_proba_val = best_lr.predict_proba(X_val)[:, 1]
thresholds = [0.5, 0.4, 0.3, 0.2, 0.1]

for t in thresholds:
    y_pred_t = (y_proba_val >= t).astype(int)
    print(
        f"threshold={t} | "
        f"F1={f1_score(y_val, y_pred_t):.3f} | "
        f"Recall={recall_score(y_val, y_pred_t):.3f}"
    )
