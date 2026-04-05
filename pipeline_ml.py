"""
Pipeline Python pour le Machine Learning
=========================================
Ce script présente les concepts et exemples pratiques
des Pipelines scikit-learn utilisés en Machine Learning.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import load_iris, load_breast_cancer, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=" * 60)
print("  PRÉSENTATION DES PIPELINES PYTHON EN MACHINE LEARNING")
print("=" * 60)


# ──────────────────────────────────────────────────────────────
# 1. PIPELINE SIMPLE : Normalisation + Régression Logistique
# ──────────────────────────────────────────────────────────────
def exemple_pipeline_simple():
    print("\n" + "─" * 60)
    print("1. PIPELINE SIMPLE : StandardScaler + LogisticRegression")
    print("─" * 60)

    # Chargement des données
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Création du pipeline
    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=200, random_state=42)),
    ])

    # Entraînement
    pipeline.fit(X_train, y_train)

    # Évaluation
    score = pipeline.score(X_test, y_test)
    print(f"  Jeu de données : Iris ({X.shape[0]} échantillons, {X.shape[1]} features)")
    print(f"  Étapes du pipeline : {[step[0] for step in pipeline.steps]}")
    print(f"  Précision sur le jeu de test : {score:.4f} ({score*100:.2f}%)")

    # Validation croisée
    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring="accuracy")
    print(f"  Validation croisée (5-fold) : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return pipeline


# ──────────────────────────────────────────────────────────────
# 2. PIPELINE AVEC PCA ET SVM
# ──────────────────────────────────────────────────────────────
def exemple_pipeline_pca_svm():
    print("\n" + "─" * 60)
    print("2. PIPELINE AVEC PCA + SVM")
    print("─" * 60)

    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=10)),
        ("svm", SVC(kernel="rbf", random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    score = accuracy_score(y_test, y_pred)

    print(f"  Jeu de données : Breast Cancer ({X.shape[0]} échantillons, {X.shape[1]} features)")
    print(f"  Étapes : StandardScaler → PCA(10 composantes) → SVM(rbf)")
    print(f"  Précision : {score:.4f} ({score*100:.2f}%)")
    print(f"  Variance expliquée par PCA : {pipeline['pca'].explained_variance_ratio_.sum():.4f}")

    return pipeline


# ──────────────────────────────────────────────────────────────
# 3. PIPELINE AVEC DONNÉES MIXTES (ColumnTransformer)
# ──────────────────────────────────────────────────────────────
def exemple_pipeline_colonnes_mixtes():
    print("\n" + "─" * 60)
    print("3. PIPELINE AVEC DONNÉES MIXTES (ColumnTransformer)")
    print("─" * 60)

    # Création d'un dataset synthétique avec colonnes numériques et catégorielles
    np.random.seed(42)
    n = 300
    df = pd.DataFrame({
        "age":       np.random.randint(18, 70, n).astype(float),
        "salaire":   np.random.normal(45000, 15000, n),
        "experience": np.random.randint(0, 40, n).astype(float),
        "niveau":    np.random.choice(["Débutant", "Intermédiaire", "Expert"], n),
        "département": np.random.choice(["Ventes", "Tech", "RH", "Finance"], n),
    })
    # Introduire quelques valeurs manquantes
    df.loc[df.sample(frac=0.05, random_state=1).index, "age"] = np.nan
    df.loc[df.sample(frac=0.05, random_state=2).index, "salaire"] = np.nan

    y = (df["salaire"].fillna(df["salaire"].median()) > 45000).astype(int)
    X = df.drop(columns=[])  # toutes les colonnes sont des features

    numeric_features = ["age", "salaire", "experience"]
    categorical_features = ["niveau", "département"]

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ])

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    print(f"  Dataset synthétique : {n} lignes, colonnes numériques + catégorielles")
    print(f"  Traitement numérique : Imputation(médiane) → StandardScaler")
    print(f"  Traitement catégoriel : Imputation(fréquent) → OneHotEncoder")
    print(f"  Modèle final : RandomForestClassifier")
    print(f"  Précision : {score:.4f} ({score*100:.2f}%)")

    return pipeline


# ──────────────────────────────────────────────────────────────
# 4. OPTIMISATION DES HYPERPARAMÈTRES AVEC GRIDSEARCHCV
# ──────────────────────────────────────────────────────────────
def exemple_gridsearch_pipeline():
    print("\n" + "─" * 60)
    print("4. OPTIMISATION DES HYPERPARAMÈTRES (GridSearchCV + Pipeline)")
    print("─" * 60)

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("svc", SVC(random_state=42)),
    ])

    param_grid = {
        "svc__C":      [0.1, 1, 10, 100],
        "svc__kernel": ["linear", "rbf"],
        "svc__gamma":  ["scale", "auto"],
    }

    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_score = grid_search.score(X_test, y_test)
    print(f"  Meilleurs paramètres : {grid_search.best_params_}")
    print(f"  Meilleur score CV    : {grid_search.best_score_:.4f}")
    print(f"  Score sur test set   : {best_score:.4f} ({best_score*100:.2f}%)")

    return grid_search.best_estimator_


# ──────────────────────────────────────────────────────────────
# 5. SÉLECTION DE FEATURES DANS UN PIPELINE
# ──────────────────────────────────────────────────────────────
def exemple_pipeline_feature_selection():
    print("\n" + "─" * 60)
    print("5. PIPELINE AVEC SÉLECTION DE FEATURES (SelectKBest)")
    print("─" * 60)

    X, y = make_classification(
        n_samples=500, n_features=20, n_informative=5,
        n_redundant=5, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipeline = Pipeline(steps=[
        ("scaler", StandardScaler()),
        ("selector", SelectKBest(score_func=f_classif, k=10)),
        ("classifier", GradientBoostingClassifier(n_estimators=100, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)

    selected_mask = pipeline["selector"].get_support()
    print(f"  Dataset : 500 échantillons, 20 features (5 informatives)")
    print(f"  Features sélectionnées : {selected_mask.sum()} / 20")
    print(f"  Précision : {score:.4f} ({score*100:.2f}%)")

    return pipeline


# ──────────────────────────────────────────────────────────────
# 6. SAUVEGARDE ET CHARGEMENT DU PIPELINE
# ──────────────────────────────────────────────────────────────
def exemple_sauvegarde_pipeline(pipeline):
    print("\n" + "─" * 60)
    print("6. SAUVEGARDE ET CHARGEMENT DU PIPELINE (joblib)")
    print("─" * 60)

    chemin = "/tmp/pipeline_ml_demo.joblib"
    joblib.dump(pipeline, chemin)
    print(f"  Pipeline sauvegardé dans : {chemin}")

    pipeline_charge = joblib.load(chemin)
    print(f"  Pipeline rechargé avec succès.")
    print(f"  Étapes : {[step[0] for step in pipeline_charge.steps]}")


# ──────────────────────────────────────────────────────────────
# 7. COMPARAISON DE PLUSIEURS PIPELINES
# ──────────────────────────────────────────────────────────────
def exemple_comparaison_pipelines():
    print("\n" + "─" * 60)
    print("7. COMPARAISON DE PLUSIEURS PIPELINES")
    print("─" * 60)

    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipelines = {
        "LR  (Régression Logistique)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=200, random_state=42)),
        ]),
        "KNN (K-plus proches voisins)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "SVM (Machines à Vecteurs de Support)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", random_state=42)),
        ]),
        "RF  (Forêt Aléatoire)": Pipeline([
            ("scaler", MinMaxScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]),
        "GB  (Gradient Boosting)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ]),
    }

    results = {}
    print(f"\n  {'Modèle':<45} {'Précision':<12} {'CV (5-fold)'}")
    print(f"  {'─'*45} {'─'*12} {'─'*20}")

    for name, pipe in pipelines.items():
        pipe.fit(X_train, y_train)
        test_score = pipe.score(X_test, y_test)
        cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="accuracy")
        results[name] = {"test": test_score, "cv_mean": cv_scores.mean(), "cv_std": cv_scores.std()}
        print(f"  {name:<45} {test_score:.4f}       {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    best = max(results, key=lambda k: results[k]["cv_mean"])
    print(f"\n  🏆 Meilleur pipeline (CV) : {best}")

    return results


# ──────────────────────────────────────────────────────────────
# POINT D'ENTRÉE
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    p1 = exemple_pipeline_simple()
    p2 = exemple_pipeline_pca_svm()
    p3 = exemple_pipeline_colonnes_mixtes()
    p4 = exemple_gridsearch_pipeline()
    p5 = exemple_pipeline_feature_selection()
    exemple_sauvegarde_pipeline(p1)
    exemple_comparaison_pipelines()

    print("\n" + "=" * 60)
    print("  FIN DE LA PRÉSENTATION")
    print("=" * 60)
