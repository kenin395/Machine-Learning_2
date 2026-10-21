
from pathlib import Path

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

RUTA_MODELO = Path(__file__).resolve().parent / "modelo_neurona.pkl"

# Cargar el modelo nuevo de Regresión Logística
datos = joblib.load(RUTA_MODELO)

modelo = datos["modelo"]
caracteristicas = datos["caracteristicas"]
clases = datos["clases"]

print("Modelo cargado correctamente.")
print("Número de características:", len(caracteristicas))


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/api/caracteristicas", methods=["GET"])
def obtener_caracteristicas():
    return jsonify({
        "caracteristicas": caracteristicas
    })


@app.route("/api/predecir", methods=["POST"])
def predecir():
    datos_json = request.get_json(silent=True)

    if not isinstance(datos_json, dict):
        return jsonify({"error": "Envía un objeto JSON válido."}), 400

    valores = datos_json.get("valores")

    if not isinstance(valores, list) or len(valores) != len(caracteristicas):
        return jsonify({
            "error": f"Debes enviar exactamente {len(caracteristicas)} valores."
        }), 400

    try:
        valores = np.asarray(valores, dtype=float)
    except (TypeError, ValueError):
        return jsonify({"error": "Todos los valores deben ser numéricos."}), 400

    if not np.all(np.isfinite(valores)):
        return jsonify({"error": "Los valores deben ser números finitos."}), 400

    entrada = valores.reshape(1, -1)

    prediccion = int(modelo.predict(entrada)[0])
    probabilidades = modelo.predict_proba(entrada)[0]

    if prediccion == 0:
        resultado_texto = "Maligno"
    else:
        resultado_texto = "Benigno"

    if isinstance(clases, (list, tuple, np.ndarray)) and 0 <= prediccion < len(clases):
        clase_predicha = str(clases[prediccion]).capitalize()
    else:
        clase_predicha = str(prediccion)

    return jsonify({
        "prediccion": prediccion,
        "resultado": resultado_texto,
        "clase": clase_predicha,
        "probabilidad_maligno": round(float(probabilidades[0]) * 100, 2),
        "probabilidad_benigno": round(float(probabilidades[1]) * 100, 2),
        "exactitud_prueba": datos["exactitud"],
        "mensaje": (
            "Estimación de un modelo educativo. "
            "No confirma ni descarta cáncer de forma diagnóstica."
        )
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
