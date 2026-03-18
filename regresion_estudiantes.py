"""
regresion_estudiantes.py — Regresión Lineal Multivariable: Dataset Estudiantes
===============================================================================
Actividad de clase  ·  Sigue la estructura de GAres.py del profesor.

Dataset: UCI Student Performance (student-mat.csv)
  · 30 variables de entrada  (codificadas + MinMaxScaler)
  · 3 variables de salida     G1 (parcial 1), G2 (parcial 2), G3 (final)  →  escala 0–20

Cromosoma:  CromosomaBinRes  con  16 bits por gen  en  [-1, 1]
  genes = [ W aplanado (30×3 = 90)  |  b (3) ]   →  93 genes en total

Aptitud:    AptitudRegresion  (definida aquí)
  · forward pass  ŷ = X · W + B  (lineal, sin activación)
  · usa  MSE()  de  interfaz/regresion_classi.py
  · fitness = 1 / (MSE + 1e-6)      →  maximizar

Salida:     gráfica 'Predicción vs Real' para G1, G2, G3
"""

import io
import urllib.request
import zipfile

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from GAbin import GA
from CromosomaBin import CromosomaBinRes
from interfaz.aptitud import Aptitud          # clase base abstracta
from interfaz.regresion_classi import MSE     # reutilizar MSE existente
from interfaz.seleccion import SeleccionTorneo
from interfaz.reproducion import ReproduccionUnCruceRes
from interfaz.mutacion import MutacionRes


# ── Función de predicción lineal ──────────────────────────────────────────────
# No existe en regresion_classi.py (la que hay usa softmax → clasificación).
# Para regresión necesitamos salida lineal sin activación.

def predecir_regresion(
    c: CromosomaBinRes,
    X: np.ndarray,
    ATRIBUTOS: int,
    SALIDAS: int,
) -> np.ndarray:
    """
    Predicción de regresión lineal.

        ŷ = X · W + B

    Distribución de genes en el cromosoma:
      · genes [0 … ATRIBUTOS×SALIDAS − 1]  →  W  de forma (ATRIBUTOS, SALIDAS)
      · genes [ATRIBUTOS×SALIDAS … final]   →  B  de forma (1, SALIDAS)
    """
    offset = ATRIBUTOS * SALIDAS
    W = np.array([c.get_gen_valor(i) for i in range(offset)]).reshape(ATRIBUTOS, SALIDAS)
    B = np.array([c.get_gen_valor(b) for b in range(offset, offset + SALIDAS)]).reshape(1, SALIDAS)
    return X @ W + B


# ── Clase de aptitud para regresión ──────────────────────────────────────────

class AptitudRegresion(Aptitud):
    """
    Aptitud para regresión lineal multivariable.

    Utiliza MSE() de interfaz/regresion_classi.py como función de pérdida.
    Fitness = 1 / (MSE + 1e-6)  →  el motor GA maximiza este valor.

    Params
    ------
    atributos : número de features de entrada (30)
    salidas   : número de variables objetivo   (3)
    """

    def __init__(self, atributos: int, salidas: int):
        self.atributos = atributos
        self.salidas   = salidas

    def evaluar_poblacion(self, poblacion, var):
        X_train = var["X_TRAIN"]
        Y_train = var["Y_TRAIN"]
        for cromosoma in poblacion:
            yh = predecir_regresion(cromosoma, X_train, self.atributos, self.salidas)
            mse_val = MSE(yh, Y_train)          # función del profesor
            cromosoma.score = 1.0 / (mse_val + 1e-6)


# ── Punto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Descargar dataset (UCI Student Performance) ───────────────────────
    print("Descargando Student Performance dataset (UCI) …")
    URL = ("https://archive.ics.uci.edu/ml/"
           "machine-learning-databases/00320/student.zip")
    try:
        with urllib.request.urlopen(URL, timeout=30) as resp:
            with zipfile.ZipFile(io.BytesIO(resp.read())) as zf:
                with zf.open("student-mat.csv") as f:
                    df = pd.read_csv(f, sep=";")
        print(f"  Dataset cargado: {df.shape[0]} muestras  ×  {df.shape[1]} columnas")
    except Exception as exc:
        print(f"\n  ERROR al descargar el dataset: {exc}")
        print("  Descarga student-mat.csv manualmente desde:")
        print("  https://archive.ics.uci.edu/ml/machine-learning-databases/00320/")
        print("  y colócalo en el mismo directorio que este script.")
        import sys
        sys.exit(1)

    # ── 2. Preprocesamiento ──────────────────────────────────────────────────
    print("\nPreprocesando datos …")

    # Variables binarias 'yes'/'no' → 0 / 1
    binary_cols = [
        "schoolsup", "famsup", "paid", "activities",
        "nursery", "higher", "internet", "romantic",
    ]
    for col in binary_cols:
        df[col] = (df[col] == "yes").astype(int)

    # Variables categóricas nominales → entero por categoría (LabelEncoder)
    categorical_cols = [
        "school", "sex", "address", "famsize", "Pstatus",
        "Mjob", "Fjob", "reason", "guardian",
    ]
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    # Variables de entrada (30) y salida (3: G1, G2, G3)
    target_cols  = ["G1", "G2", "G3"]
    feature_cols = [c for c in df.columns if c not in target_cols]
    X_raw = df[feature_cols].values.astype(float)
    y_raw = df[target_cols].values.astype(float)

    print(f"  Features : {X_raw.shape[1]}  |  Targets : {y_raw.shape[1]}  "
          f"(G1, G2, G3 en escala 0–20)")

    # MinMaxScaler en entradas (tal como indica la práctica)
    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_raw)

    # Escalar salidas a [0, 1] para que sean compatibles con W ∈ [-1, 1]
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y_raw)

    # División 80 % entrenamiento / 20 % prueba
    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        X_scaled, y_scaled, test_size=0.20, random_state=42
    )
    print(f"  Entrenamiento: {X_TRAIN.shape[0]}  |  Prueba: {X_TEST.shape[0]}")

    # ── 3. Configuración del Algoritmo Genético ──────────────────────────────
    ATRIBUTOS       = X_TRAIN.shape[1]          # 30
    SALIDAS         = Y_TRAIN.shape[1]          # 3
    CARACTERISTICAS = ATRIBUTOS * SALIDAS + SALIDAS   # 93 genes

    cant_ind = 200
    objetivo = "maximizar"

    gaparams = {
        "Cromosoma":        CromosomaBinRes,
        "Cromosoma_params": [CARACTERISTICAS, 16, -1, 1],
        "Aptitud":          AptitudRegresion(ATRIBUTOS, SALIDAS),
        "Seleccion":        SeleccionTorneo(10, objetivo),
        "Reproduccion":     ReproduccionUnCruceRes(objetivo),
        "Mutacion":         MutacionRes(0.02),
        "por_remplazo":     0.5,
        "generaciones":     5_000,
        "objetivo":         objetivo,
        "stop":             "loss",
    }

    print(f"\n  Configuración:")
    print(f"    Genes por cromosoma : {CARACTERISTICAS}  "
          f"(W={ATRIBUTOS}×{SALIDAS} + b={SALIDAS})")
    print(f"    Individuos          : {cant_ind}")
    print(f"    Generaciones máx.   : {gaparams['generaciones']:,}")
    print(f"    Mutación (prob bit) : {gaparams['Mutacion'].prob}")
    print(f"    Bits por gen        : 16  →  resolución {2**16} niveles en [-1, 1]")
    print(f"    Fitness             : 1 / (MSE + 1e-6)")

    # ── 4. Ejecutar AG ───────────────────────────────────────────────────────
    print("\nIniciando evolución …\n")
    variables = {"X_TRAIN": X_TRAIN, "Y_TRAIN": Y_TRAIN}

    algoritmoGA = GA(cant_ind, None, gaparams)
    algoritmoGA.buscar_solucion(1e-8, variables)

    # ── 5. Evaluar mejor individuo ───────────────────────────────────────────
    top10 = sorted(
        algoritmoGA.mejor_generacion[:10],
        key=lambda c: c.score,
        reverse=True,
    )
    topCrom = top10[0]

    # Predicción en test (escala normalizada [0,1])
    yh_norm = predecir_regresion(topCrom, X_TEST, ATRIBUTOS, SALIDAS)
    mse_norm = MSE(yh_norm, Y_TEST)

    # Desnormalizar a escala original (0–20)
    yh_orig  = scaler_y.inverse_transform(np.clip(yh_norm, 0.0, 1.0))
    y_orig   = scaler_y.inverse_transform(Y_TEST)

    mse_total = float(np.mean((yh_orig - y_orig) ** 2))
    rmse_total = float(np.sqrt(mse_total))

    print(f"\n{'─'*55}")
    print(f"  Mejor fitness (1/MSE)       : {topCrom.score:.4f}")
    print(f"  MSE en test (norm.)         : {mse_norm:.6f}")
    print(f"  MSE en test (escala orig.)  : {mse_total:.4f}")
    print(f"  RMSE en test (escala orig.) : {rmse_total:.4f}  puntos")
    print(f"{'─'*55}")

    # Métricas por variable objetivo
    print(f"\n  {'Variable':<8}  {'MSE':>8}  {'RMSE':>8}")
    print(f"  {'─'*8}  {'─'*8}  {'─'*8}")
    for i, nombre in enumerate(target_cols):
        mse_i  = float(np.mean((y_orig[:, i] - yh_orig[:, i]) ** 2))
        rmse_i = float(np.sqrt(mse_i))
        print(f"  {nombre:<8}  {mse_i:>8.4f}  {rmse_i:>8.4f}")

    # ── 6. Gráfica Predicción vs Real ────────────────────────────────────────
    print("\nGenerando gráfica Predicción vs Real …")

    colores = ["#4A90D9", "#E67E22", "#27AE60"]
    labels  = ["G1 — Parcial 1", "G2 — Parcial 2", "G3 — Final"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for i, (ax, label, color) in enumerate(zip(axes, labels, colores)):
        real = y_orig[:, i]
        pred = yh_orig[:, i]

        # Scatter
        ax.scatter(real, pred, color=color, alpha=0.65,
                   edgecolors="white", linewidths=0.5, s=65, label="Muestras")

        # Línea de predicción perfecta
        lo = min(real.min(), pred.min()) - 0.5
        hi = max(real.max(), pred.max()) + 0.5
        ax.plot([lo, hi], [lo, hi], "r--", lw=1.6, label="Predicción perfecta")

        mse_i  = float(np.mean((real - pred) ** 2))
        rmse_i = float(np.sqrt(mse_i))

        ax.set_xlabel(f"Calificación Real — {label}", fontsize=10)
        ax.set_ylabel("Calificación Predicha",         fontsize=10)
        ax.set_title(f"{label}\nRMSE = {rmse_i:.3f} pts",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.30)

    fig.suptitle(
        f"Regresión Estudiantes — Predicción vs Real\n"
        f"(MSE total = {mse_total:.4f}  ·  RMSE = {rmse_total:.4f} pts)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("estudiantes_prediccion_vs_real.png", dpi=150, bbox_inches="tight")
    print("  Gráfica guardada: estudiantes_prediccion_vs_real.png")
    plt.show()
