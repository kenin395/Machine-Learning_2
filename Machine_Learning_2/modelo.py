
import joblib
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)


RUTA_MODELO = Path(__file__).resolve().parent / "modelo_neurona.pkl"


def entrenar_y_guardar():
    datos = load_breast_cancer()

    X = datos.data
    y = datos.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    # Escalar los datos y entrenar el clasificador.
    modelo = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        )
    )

    modelo.fit(X_train, y_train)

    predicciones = modelo.predict(X_test)
    exactitud = accuracy_score(y_test, predicciones)

    print("\nMatriz de confusión:")
    print(confusion_matrix(y_test, predicciones, labels=[0, 1]))

    print("\nReporte de clasificación:")
    print(classification_report(
        y_test,
        predicciones,
        labels=[0, 1],
        target_names=["Maligno", "Benigno"],
        zero_division=0
    ))

    joblib.dump({
        "modelo": modelo,
        "caracteristicas": datos.feature_names.tolist(),
        "clases": datos.target_names.tolist(),
        "exactitud": float(exactitud)
    }, RUTA_MODELO)

    print("\nModelo entrenado y guardado.")
    print(f"Exactitud de prueba: {exactitud * 100:.2f}%")
    print(f"Archivo guardado en: {RUTA_MODELO}")


if __name__ == "__main__":
    entrenar_y_guardar()
