from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

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


pca = Pipeline([
    ('proceso', proceso),
    ('pca', PCA(n_components=2))
])


data_reducido = pca.fit_transform(X_train)

pca_df = pd.DataFrame(
    data_reducido,
    columns=['PC1', 'PC2']
)

pca_df['Churn'] = y_train.values


#------Gráfico---------
plt.figure(figsize=(8, 6))

plt.scatter(
    pca_df[pca_df['Churn'] == 0]['PC1'],
    pca_df[pca_df['Churn'] == 0]['PC2'],
    alpha=0.5,
    label='No Churn'
)

plt.scatter(
    pca_df[pca_df['Churn'] == 1]['PC1'],
    pca_df[pca_df['Churn'] == 1]['PC2'],
    alpha=0.5,
    label='Churn'
)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA (2 componentes) - Customer Churn')
plt.legend()
plt.show()

