import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def carregar_dados(caminho_csv="breast-cancer.csv"):
    df = pd.read_csv(caminho_csv, na_values="?")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def construir_preprocessador(colunas):
    pipeline_categorico = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer(
        transformers=[
            ("cat", pipeline_categorico, colunas)
        ]
    )


def main():
    X, y = carregar_dados("breast-cancer.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    colunas_features = X.columns.tolist()
    preprocessador = construir_preprocessador(colunas_features)

    modelos = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=7)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    melhor_nome = None
    melhor_pipeline = None
    melhor_score_cv = -1.0

    print("Avaliacao - Validacao Cruzada (5-Fold):")
    for nome, estimador in modelos.items():
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessador),
            ("classifier", estimador)
        ])

        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        media_acc = scores.mean()
        print(f"{nome:20s} | Acuracia media: {media_acc:.4f}")

        if media_acc > melhor_score_cv:
            melhor_score_cv = media_acc
            melhor_nome = nome
            melhor_pipeline = pipeline

    melhor_pipeline.fit(X_train, y_train)
    y_pred = melhor_pipeline.predict(X_test)
    acc_teste = accuracy_score(y_test, y_pred)

    print(f"\nMelhor modelo: {melhor_nome}")
    print(f"Acuracia no teste: {acc_teste:.4f}\n")
    print("Relatorio de Classificacao:")
    print(classification_report(y_test, y_pred))
    print("Matriz de Confusao:")
    print(confusion_matrix(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    caminho_modelo = os.path.join("models", "best_model.joblib")
    joblib.dump(melhor_pipeline, caminho_modelo)
    print(f"\nModelo salvo em: {caminho_modelo}")


if __name__ == "__main__":
    main()