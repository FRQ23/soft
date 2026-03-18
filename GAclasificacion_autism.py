"""
GAclasificacion_autism.py — Clasificación Binaria: Autism Spectrum Disorder
============================================================================
Actividad de clase  ·  Sigue la estructura de GAres.py del profesor.

Dataset : Autism Screening Adults (Kaggle – faizunnabi / UCI, 2018)
Archivo : data/Autism_Data.arff
Filas   : 704 pacientes
Target  : Class/ASD  →  YES=1 / NO=0  (clasificación binaria)

Problemática:
  Predecir si un paciente tiene Trastorno del Espectro Autista (ASD) a partir
  de sus respuestas al cuestionario AQ-10 y datos sociodemográficos.
  Detección temprana de ASD usando aprendizaje automático evolutivo.

Cromosoma : CromosomaBinRes  16 bits  rango [-1, 1]
  CLASES = 2  →  genes = [ W(n_feat×2)  |  b(2) ]
  Modelo: argmax(softmax(X·W + b))  →  clase 0 ó 1

Fitness   : Accuracy en entrenamiento   →   maximizar
            usa AptitudMNIST(objetivo="maximizar") del profesor
            y regresion_cromosoma() con softmax

Preprocesamiento:
  · Archivo .arff → DataFrame con scipy.io.arff
  · Variables categóricas → LabelEncoder
  · Entradas → StandardScaler
"""

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from GAbin import GA
from CromosomaBin import CromosomaBinRes
from interfaz.aptitud import AptitudMNIST
from interfaz.regresion_classi import (
    regresion_cromosoma,
    accuracy,
    onehot_encode,
    matriz_confusion,
)
from interfaz.seleccion import SeleccionTorneo
from interfaz.reproducion import ReproduccionUnCruceRes
from interfaz.mutacion import MutacionRes


if __name__ == "__main__":

    DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "Autism_Data.arff")

    # ── 1. Cargar archivo (extensión .arff pero contenido CSV) ──────────────
    print("Cargando Autism Screening dataset …")
    df = pd.read_csv(DATA_PATH, na_values=["?", " ?", "? "])

    print(f"  Shape : {df.shape[0]} filas × {df.shape[1]} columnas")
    print(f"  Target: Class/ASD  →  {df['Class/ASD'].value_counts().to_dict()}")

    # ── 2. Preprocesamiento ──────────────────────────────────────────────────
    print("\nPreprocesando …")

    # Separar target
    y_str = df["Class/ASD"].values
    y_all = (y_str == "YES").astype(int)   # YES=1, NO=0

    # Features: eliminar target y columnas redundantes
    DROP = ["Class/ASD", "age_desc"]       # age_desc duplica 'age'
    X_df = df.drop(columns=[c for c in DROP if c in df.columns]).copy()

    # Codificar categorías con LabelEncoder
    le = LabelEncoder()
    for col in X_df.select_dtypes(include="object").columns:
        X_df[col] = le.fit_transform(X_df[col].astype(str))

    X_raw = X_df.values.astype(float)
    print(f"  Features: {X_raw.shape[1]}")
    print(f"  Clases  : NO={int(sum(y_all==0))}  YES={int(sum(y_all==1))}")

    # StandardScaler (los scores A1-A10 ya son binarios, pero normalizar
    # beneficia las columnas continuas como age, result)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    CLASES    = 2
    X_TRAIN, X_TEST, Y_TRAIN_INT, Y_TEST_INT = train_test_split(
        X_scaled, y_all, test_size=0.20, random_state=42, stratify=y_all
    )
    y_train = onehot_encode(Y_TRAIN_INT, CLASES)
    y_test  = onehot_encode(Y_TEST_INT,  CLASES)

    print(f"  Entrenamiento: {X_TRAIN.shape[0]}  |  Prueba: {X_TEST.shape[0]}")

    # ── 3. Pairplot exploratorio ─────────────────────────────────────────────
    print("\nGenerando pairplot …")

    # Usar scores A1-A10 + result para visualizar separación de clases
    plot_cols = [f"A{i}_Score" for i in range(1, 6)] + ["result", "age"]
    plot_cols  = [c for c in plot_cols if c in df.columns]

    df_plot = df[plot_cols + ["Class/ASD"]].copy()
    for col in df_plot.select_dtypes(include="object").columns:
        if col != "Class/ASD":
            df_plot[col] = le.fit_transform(df_plot[col].astype(str))
    df_plot[plot_cols] = df_plot[plot_cols].apply(pd.to_numeric, errors="coerce")

    sns.set_theme(style="ticks")
    grid = sns.pairplot(
        df_plot,
        hue="Class/ASD",
        palette={"NO": "#3498db", "YES": "#e74c3c"},
        diag_kind="kde",
        plot_kws={"alpha": 0.55, "s": 25},
        height=2.2,
    )
    grid.figure.suptitle(
        "Autism Screening — Distribución de Features por Clase (ASD: YES / NO)",
        y=1.02, fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    grid.figure.savefig("autism_pairplot.png", dpi=120, bbox_inches="tight")
    print("  Pairplot guardado: autism_pairplot.png")
    plt.show()
    sns.reset_defaults()

    # ── 4. Configuración del AG ──────────────────────────────────────────────
    ATRIBUTOS       = X_TRAIN.shape[1]
    CARACTERISTICAS = ATRIBUTOS * CLASES + CLASES   # W(n×2) + b(2)

    cant_ind = 200
    objetivo = "maximizar"

    variables = {"X_TRAIN": X_TRAIN, "Y_TRAIN": y_train}

    gaparams = {
        "Cromosoma":        CromosomaBinRes,
        "Cromosoma_params": [CARACTERISTICAS, 16, -1, 1],
        # AptitudMNIST con objetivo="maximizar" usa accuracy como score
        "Aptitud":          AptitudMNIST(ATRIBUTOS, CLASES, objetivo),
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
    print(f"    Clases    : {CLASES}  (NO=0, YES=1)")
    print(f"    Genes     : {CARACTERISTICAS}  (W={ATRIBUTOS}×{CLASES} + b={CLASES})")
    print(f"    Individuos: {cant_ind}")
    print(f"    Generaciones: {gaparams['generaciones']:,}")
    print(f"    Fitness   : Accuracy  (maximizar)")

    # ── 5. Ejecutar AG ───────────────────────────────────────────────────────
    print("\nIniciando evolución …\n")
    algoritmoGA = GA(cant_ind, None, gaparams)
    algoritmoGA.buscar_solucion(1e-8, variables)

    # ── 6. Evaluación ────────────────────────────────────────────────────────
    top10   = sorted(algoritmoGA.mejor_generacion[:10],
                     key=lambda c: c.score, reverse=True)
    topCrom = top10[0]

    yh      = regresion_cromosoma(topCrom, X_TEST, ATRIBUTOS, CLASES)
    acc     = accuracy(yh, y_test)
    acc_tr  = topCrom.score

    print(f"\n{'─'*55}")
    print(f"  Accuracy entrenamiento : {acc_tr:.4f}  ({acc_tr*100:.2f}%)")
    print(f"  Accuracy prueba        : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"{'─'*55}")

    # Tabla de predicciones
    nombres = ["NO (sin ASD)", "YES (con ASD)"]
    print(f"\n  {'#':>3}  {'Real':<14}  {'Predicción':<14}  ✓/✗")
    print(f"  {'─'*46}")
    for i in range(len(y_test)):
        real = int(np.argmax(y_test[i]))
        pred = int(np.argmax(yh[i]))
        ok   = "✓" if real == pred else "✗"
        print(f"  {i+1:>3}  {nombres[real]:<14}  {nombres[pred]:<14}  {ok}")

    # ── 7. Matriz de confusión ───────────────────────────────────────────────
    print("\nGenerando Matriz de Confusión …")
    mc = matriz_confusion(y_test, yh, CLASES)

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(mc, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["NO", "YES"],
                yticklabels=["NO", "YES"])
    ax.set_xlabel("Predicción del Modelo", fontsize=11)
    ax.set_ylabel("Diagnóstico Real",      fontsize=11)
    ax.set_title(
        f"Autism ASD — Clasificación con Algoritmo Genético\n"
        f"Accuracy = {acc:.4f}  ({acc*100:.2f}%)",
        fontsize=12, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    plt.savefig("autism_confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("  Matriz guardada: autism_confusion_matrix.png")
    plt.show()
