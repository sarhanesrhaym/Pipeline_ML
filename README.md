# 🚀 Pipeline Python pour le Machine Learning

## Présentation

Un **Pipeline** en Machine Learning est une séquence d'étapes de traitement de données et d'apprentissage automatique enchaînées de manière ordonnée. Il permet d'automatiser le flux de travail complet allant du prétraitement des données jusqu'à la prédiction du modèle.

---

## 📚 Table des Matières

1. [Qu'est-ce qu'un Pipeline ?](#quest-ce-quun-pipeline-)
2. [Pourquoi utiliser un Pipeline ?](#pourquoi-utiliser-un-pipeline-)
3. [Structure d'un Pipeline](#structure-dun-pipeline)
4. [Exemples de Pipelines](#exemples-de-pipelines)
5. [Installation & Utilisation](#installation--utilisation)
6. [Contenu du Projet](#contenu-du-projet)

---

## Qu'est-ce qu'un Pipeline ?

Un Pipeline est une abstraction qui regroupe plusieurs transformations de données (prétraitement) et un estimateur (modèle) en un seul objet cohérent.

```
Données Brutes → [Étape 1: Nettoyage] → [Étape 2: Transformation] → [Étape 3: Modèle] → Prédiction
```

En Python, la bibliothèque **scikit-learn** fournit la classe `Pipeline` qui simplifie considérablement ce processus.

---

## Pourquoi utiliser un Pipeline ?

| Avantage | Description |
|---|---|
| ✅ **Reproductibilité** | Les mêmes transformations sont appliquées systématiquement en entraînement et en production |
| ✅ **Prévention du Data Leakage** | Les transformations sont ajustées uniquement sur les données d'entraînement |
| ✅ **Code plus propre** | Moins de code répétitif, meilleure lisibilité |
| ✅ **Validation croisée simplifiée** | `cross_val_score` et `GridSearchCV` fonctionnent directement avec les pipelines |
| ✅ **Déploiement facilité** | Sauvegarder et charger un seul objet pipeline avec `joblib` |

---

## Structure d'un Pipeline

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),       # Étape 1 : Normalisation
    ('classifier', LogisticRegression()) # Étape 2 : Modèle
])

pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)
```

---

## Exemples de Pipelines

### 1. Pipeline Classique (Classification)

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

pipe = Pipeline([
    ('normalisation', StandardScaler()),
    ('svm', SVC(kernel='rbf'))
])
pipe.fit(X_train, y_train)
score = pipe.score(X_test, y_test)
```

### 2. Pipeline avec Imputation et Encodage

```python
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier

# Transformations pour colonnes numériques
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Transformations pour colonnes catégorielles
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combinaison avec ColumnTransformer
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

# Pipeline final
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
```

### 3. Pipeline avec Optimisation des Hyperparamètres

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svc', SVC())
])

param_grid = {
    'svc__C': [0.1, 1, 10],
    'svc__kernel': ['linear', 'rbf'],
    'svc__gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
print(f"Meilleurs paramètres : {grid_search.best_params_}")
```

---

## Installation & Utilisation

### Prérequis

```bash
pip install -r requirements.txt
```

### Exécuter les exemples Python

```bash
python pipeline_ml.py
```

### Ouvrir le notebook Jupyter

```bash
jupyter notebook Pipeline_ML_Presentation.ipynb
```

---

## Contenu du Projet

```
Pipeline_ML/
│
├── README.md                          # Cette présentation
├── requirements.txt                   # Dépendances Python
├── pipeline_ml.py                     # Script Python avec exemples complets
└── Pipeline_ML_Presentation.ipynb     # Notebook Jupyter interactif
```

---

## Concepts Clés

### Transformers vs Estimators

- **Transformer** : objet avec méthodes `fit()` et `transform()` (ex: `StandardScaler`, `PCA`)
- **Estimator** : objet avec méthodes `fit()` et `predict()` (ex: `LogisticRegression`, `RandomForest`)
- **Pipeline** : chaîne de transformers suivie d'un estimator final

### Méthodes importantes du Pipeline

| Méthode | Description |
|---|---|
| `fit(X, y)` | Entraîne toutes les étapes du pipeline |
| `predict(X)` | Transforme les données et fait une prédiction |
| `transform(X)` | Applique les transformations (sans l'estimateur final) |
| `fit_transform(X, y)` | Combine `fit` et `transform` |
| `score(X, y)` | Évalue le pipeline sur les données fournies |
| `set_params(**params)` | Modifie les paramètres des étapes |
| `get_params()` | Retourne tous les paramètres du pipeline |

---

## Ressources

- [Documentation scikit-learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
- [Guide utilisateur scikit-learn - Pipelines](https://scikit-learn.org/stable/modules/compose.html)
- [ColumnTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.compose.ColumnTransformer.html)