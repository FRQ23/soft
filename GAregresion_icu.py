"""
GAregresion_icu.py — Regresión: Predicción de SpO2 en UCI (MIMIC-IV Sintético)
===============================================================================
Actividad de clase  ·  Sigue la estructura de GAres.py del profesor.

Dataset : MIMIC-IV Style ICU Synthetic Dataset (Kaggle – sinanshereef, 2026)
Archivo : data/sepsis_icu_synthetic.csv
Filas   : 5,000 estancias UCI
Target  : spo2_mean — saturación de oxígeno en sangre (%, valor continuo)

Problemática:
  Predecir el nivel de oxigenación (SpO2) de un paciente en UCI a partir de
  sus signos vitales, valores de laboratorio, comorbilidades y scores clínicos.
  Valores bajos de SpO2 (<90%) indican hipoxemia y riesgo vital.

Cromosoma : CromosomaBinRes  16 bits  rango [-1, 1]
  genes = [ W aplanado (n_feat × 1)  |  b (1) ]

Fitness   : 1 / (MSE + 1e-6)   →   maximizar
            usa MSE() de interfaz/regresion_classi.py

Preprocesamiento:
  · Valores faltantes → mediana por columna
  · Variables categóricas → LabelEncoder
  · Entradas → MinMaxScaler  (tal como indica la práctica)
  · Salida (spo2_mean) → MinMaxScaler (para compatibilidad con W ∈ [-1,1])
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from GAbin import GA
from CromosomaBin import CromosomaBinRes
from interfaz.aptitud import Aptitud
from interfaz.regresion_classi import MSE
from interfaz.seleccion import SeleccionTorneo
from interfaz.reproducion import ReproduccionUnCruceRes
from interfaz.mutacion import MutacionRes


# ── Función de predicción lineal (salida continua, sin activación) ────────────

def predecir_regresion(c: CromosomaBinRes, X: np.ndarray,
                       ATRIBUTOS: int, SALIDAS: int) -> np.ndarray:
    """ŷ = X · W + B  (regresión lineal, sin activación)."""
    genes  = c.get_all_valores()
    offset = ATRIBUTOS * SALIDAS
    W = genes[:offset].reshape(ATRIBUTOS, SALIDAS)
    B = genes[offset:offset + SALIDAS].reshape(1, SALIDAS)
    return X @ W + B


# ── Clase de aptitud para regresión ──────────────────────────────────────────

class AptitudRegresion(Aptitud):
    """
    Fitness = 1 / (MSE + 1e-6)  →  maximizar.
    Usa MSE() de interfaz/regresion_classi.py.
    """
    def __init__(self, atributos: int, salidas: int):
        self.atributos = atributos
        self.salidas   = salidas

    def evaluar_poblacion(self, poblacion, var):
        X_tr, Y_tr = var["X_TRAIN"], var["Y_TRAIN"]
        for cromosoma in poblacion:
            yh          = predecir_regresion(cromosoma, X_tr, self.atributos, self.salidas)
            mse_val     = MSE(yh, Y_tr)
            cromosoma.score = 1.0 / (mse_val + 1e-6)


# ── Punto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "sepsis_icu_synthetic.csv")

    # ── 1. Cargar dataset ────────────────────────────────────────────────────
    print("Cargando MIMIC-IV ICU dataset …")
    df = pd.read_csv(DATA_PATH)
    print(f"  Shape original: {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"  Valores faltantes: {df.isnull().sum().sum()}")

    # ── 2. Preprocesamiento ──────────────────────────────────────────────────
    print("\nPreprocesando …")

    # Target: SpO2 promedio (saturación de oxígeno)
    TARGET = "spo2_mean"

    # Excluir columnas que causarían fuga de datos o son identificadores
    DROP_COLS = [
        "subject_id", "hadm_id",          # identificadores
        TARGET,                            # target (se separa abajo)
        "spo2_min", "spo2_max",            # fuga directa de SpO2
        "sepsis_label",                    # otro target del dataset
    ]
    drop_existentes = [c for c in DROP_COLS if c in df.columns]
    y_raw = df[TARGET].values.reshape(-1, 1).astype(float)
    X_df  = df.drop(columns=drop_existentes)

    # Rellenar valores faltantes con la mediana de cada columna
    for col in X_df.columns:
        if X_df[col].isnull().any():
            if X_df[col].dtype == object:
                X_df[col].fillna(X_df[col].mode()[0], inplace=True)
            else:
                X_df[col].fillna(X_df[col].median(), inplace=True)

    # Codificar variables categóricas
    le = LabelEncoder()
    for col in X_df.select_dtypes(include="object").columns:
        X_df[col] = le.fit_transform(X_df[col].astype(str))

    X_raw = X_df.values.astype(float)
    print(f"  Features finales : {X_raw.shape[1]}")
    print(f"  Target           : {TARGET}  (SpO2 %, continuo)")
    print(f"  Rango SpO2       : [{y_raw.min():.1f}%, {y_raw.max():.1f}%]")

    # MinMaxScaler en entradas y salida
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X_raw)
    y_scaled = scaler_y.fit_transform(y_raw)          # shape (N, 1)

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        X_scaled, y_scaled, test_size=0.20, random_state=42
    )
    print(f"  Entrenamiento: {X_TRAIN.shape[0]}  |  Prueba: {X_TEST.shape[0]}")

    # ── 3. Pairplot: SpO2 vs signos vitales clave ────────────────────────────
    print("\nGenerando pairplot …")
    vital_cols = ["hr_mean", "resp_rate_mean", "temp_mean", "map_mean", TARGET]
    vital_cols = [c for c in vital_cols if c in df.columns]

    df_plot = df[vital_cols].dropna().sample(min(800, len(df)), random_state=42)

    # Discretizar SpO2 en categorías para colorear el pairplot
    df_plot = df_plot.copy()
    df_plot["SpO2_nivel"] = pd.cut(
        df_plot[TARGET],
        bins=[0, 90, 95, 101],
        labels=["Bajo (<90%)", "Normal (90-95%)", "Alto (>95%)"],
    )

    sns.set_theme(style="ticks")
    grid = sns.pairplot(
        df_plot.drop(columns=[TARGET]),
        hue="SpO2_nivel",
        palette={"Bajo (<90%)": "#e74c3c",
                 "Normal (90-95%)": "#f39c12",
                 "Alto (>95%)": "#27ae60"},
        diag_kind="kde",
        plot_kws={"alpha": 0.5, "s": 20},
        height=2.4,
    )
    grid.figure.suptitle(
        "UCI ICU — Signos Vitales vs Nivel de SpO₂ (saturación de oxígeno)",
        y=1.02, fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    grid.figure.savefig("icu_pairplot.png", dpi=120, bbox_inches="tight")
    print("  Pairplot guardado: icu_pairplot.png")
    plt.show()
    sns.reset_defaults()

    # ── 4. Configuración del AG ──────────────────────────────────────────────
    ATRIBUTOS       = X_TRAIN.shape[1]
    SALIDAS         = 1
    CARACTERISTICAS = ATRIBUTOS * SALIDAS + SALIDAS

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

    print(f"\n  Configuración AG:")
    print(f"    Features  : {ATRIBUTOS}")
    print(f"    Genes     : {CARACTERISTICAS}  (W={ATRIBUTOS}×1 + b=1)")
    print(f"    Individuos: {cant_ind}")
    print(f"    Generaciones: {gaparams['generaciones']:,}")
    print(f"    Fitness   : 1 / (MSE + 1e-6)")

    # ── 5. Ejecutar AG ───────────────────────────────────────────────────────
    print("\nIniciando evolución …\n")
    variables = {"X_TRAIN": X_TRAIN, "Y_TRAIN": Y_TRAIN}
    algoritmoGA = GA(cant_ind, None, gaparams)
    algoritmoGA.buscar_solucion(1e-8, variables)

    # ── 6. Evaluación ────────────────────────────────────────────────────────
    top10   = sorted(algoritmoGA.mejor_generacion[:10],
                     key=lambda c: c.score, reverse=True)
    topCrom = top10[0]

    yh_norm  = predecir_regresion(topCrom, X_TEST, ATRIBUTOS, SALIDAS)
    mse_norm = MSE(yh_norm, Y_TEST)

    # Desnormalizar a escala original (% SpO2)
    yh_orig = scaler_y.inverse_transform(np.clip(yh_norm, 0.0, 1.0))
    y_orig  = scaler_y.inverse_transform(Y_TEST)

    mse_orig  = float(np.mean((yh_orig - y_orig) ** 2))
    rmse_orig = float(np.sqrt(mse_orig))

    print(f"\n{'─'*55}")
    print(f"  Fitness final (1/MSE)       : {topCrom.score:.4f}")
    print(f"  MSE  en test (norm.)        : {mse_norm:.6f}")
    print(f"  MSE  en test (% SpO2)       : {mse_orig:.4f}")
    print(f"  RMSE en test (% SpO2)       : {rmse_orig:.4f}  puntos porcentuales")
    print(f"{'─'*55}")

    # ── 7. Gráfica Predicción vs Real ────────────────────────────────────────
    print("\nGenerando gráfica Predicción vs Real …")

    real = y_orig.flatten()
    pred = yh_orig.flatten()
    residuos = real - pred

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Scatter: Real vs Predicho
    ax = axes[0]
    ax.scatter(real, pred, alpha=0.50, color="#3498db",
               edgecolors="white", linewidths=0.4, s=50, label="Muestras")
    lo = min(real.min(), pred.min()) - 1
    hi = max(real.max(), pred.max()) + 1
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.8, label="Predicción perfecta")
    ax.set_xlabel("SpO₂ Real (%)",    fontsize=11)
    ax.set_ylabel("SpO₂ Predicho (%)", fontsize=11)
    ax.set_title(f"SpO₂ Real vs Predicho\nRMSE = {rmse_orig:.3f} pp",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.30)

    # Histograma de residuos
    ax2 = axes[1]
    ax2.hist(residuos, bins=30, color="#e67e22", edgecolor="white",
             alpha=0.8)
    ax2.axvline(0, color="red", lw=1.8, ls="--", label="Error = 0")
    ax2.axvline(residuos.mean(), color="#2ecc71", lw=1.5,
                ls="-", label=f"Media = {residuos.mean():.2f}")
    ax2.set_xlabel("Residuo  (Real − Predicho)", fontsize=11)
    ax2.set_ylabel("Frecuencia",                 fontsize=11)
    ax2.set_title("Distribución de Residuos",    fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.30)

    fig.suptitle(
        f"Regresión UCI — Predicción de SpO₂ con Algoritmo Genético\n"
        f"MSE = {mse_orig:.4f}  ·  RMSE = {rmse_orig:.4f} pp",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("icu_prediccion_vs_real.png", dpi=150, bbox_inches="tight")
    print("  Gráfica guardada: icu_prediccion_vs_real.png")
    plt.show()
