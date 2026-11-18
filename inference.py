import os
import joblib
import pandas as pd

CAMINHO_MODELO = os.path.join("models", "best_model.joblib")


def prever_diagnostico(dados_paciente):
    if not os.path.exists(CAMINHO_MODELO):
        raise FileNotFoundError("Modelo nao encontrado. Execute train.py primeiro.")

    modelo = joblib.load(CAMINHO_MODELO)
    df_paciente = pd.DataFrame([dados_paciente])

    classe_predita = modelo.predict(df_paciente)[0]
    probabilidades = modelo.predict_proba(df_paciente)[0]
    classes = modelo.classes_

    probs_formatadas = {
        cls: round(float(prob) * 100, 2)
        for cls, prob in zip(classes, probabilidades)
    }

    return {
        "Class": classe_predita,
        "probabilidades": probs_formatadas
    }


if __name__ == "__main__":
    nova_paciente = {
        "age": "40-49",
        "menopause": "premeno",
        "tumor-size": "25-29",
        "inv-nodes": "0-2",
        "node-caps": "no",
        "deg-malig": 2,
        "breast": "left",
        "breast-quad": "left_low",
        "irradiat": "no"
    }

    resultado = prever_diagnostico(nova_paciente)
    print("Resultado da Inferencia:")
    print(f"Class: {resultado['Class']}")
    print(f"Probabilidades (%): {resultado['probabilidades']}")