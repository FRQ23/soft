"""
GAres_iris.py — Clasificación Multiclase: Dataset Iris
=======================================================
Actividad de clase  ·  Basado en la plantilla GAres.py del profesor.

Flujo:
  1. Carga Iris vía fetch_openml
  2. Pairplot con Seaborn ANTES de entrenar
  3. AG maximiza Accuracy sobre train (AptitudMNIST con objetivo="maximizar")
  4. Evalúa el mejor cromosoma en test
  5. Muestra Accuracy + Matriz de Confusión

Funciones reutilizadas de interfaz/regresion_classi.py:
  · regresion_cromosoma  — forward pass (softmax)
  · accuracy             — métrica de precisión
  · onehot_encode        — codificación one-hot de etiquetas
  · matriz_confusion     — cálculo de la matriz de confusión
"""

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

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # ── 1. Cargar dataset Iris ───────────────────────────────────────────────
    print("Cargando Iris dataset …")
    iris = fetch_openml("iris", version=1, as_frame=True, parser="auto")

    class_map = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    x_all = iris.data.to_numpy().astype(float)
    y_all = np.array([class_map[cls] for cls in iris.target])

    print(f"  Muestras: {x_all.shape[0]}  |  Features: {x_all.shape[1]}  |  Clases: 3")

    # ── 2. Pairplot exploratorio (ANTES de entrenar) ─────────────────────────
    print("\nGenerando pairplot de Seaborn …")
    sns.set_theme(style="ticks", palette="Set1")
    grid = sns.pairplot(data=iris.frame, hue="class", palette="Set1",
                        diag_kind="kde", plot_kws={"alpha": 0.6})
    grid.figure.suptitle("Iris Dataset — Exploración de Features", y=1.02,
                          fontsize=14, fontweight="bold")
    plt.tight_layout()
    grid.figure.savefig("iris_pairplot.png", dpi=130, bbox_inches="tight")
    print("  Pairplot guardado: iris_pairplot.png")
    plt.show()
    sns.reset_defaults()

    # ── 3. Preprocesamiento ──────────────────────────────────────────────────
    CLASES          = 3
    ATRIBUTOS       = 4
    CARACTERISTICAS = ATRIBUTOS * CLASES + CLASES   # 15 genes  [W(4×3) | b(3)]

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = train_test_split(
        x_all, y_all, test_size=50, random_state=42
    )
    y_train = onehot_encode(Y_TRAIN, CLASES)
    y_test  = onehot_encode(Y_TEST,  CLASES)

    print(f"\n  Entrenamiento: {X_TRAIN.shape[0]} muestras  |  "
          f"Prueba: {X_TEST.shape[0]} muestras")

    variables = {"X_TRAIN": X_TRAIN, "Y_TRAIN": y_train}

    # ── 4. Configuración del Algoritmo Genético ──────────────────────────────
    cant_ind = 200
    objetivo = "maximizar"

    gaparams = {
        "Cromosoma":        CromosomaBinRes,
        "Cromosoma_params": [CARACTERISTICAS, 16, -1, 1],
        # AptitudMNIST con objetivo="maximizar" usa accuracy como score
        "Aptitud":          AptitudMNIST(ATRIBUTOS, CLASES, objetivo),
        "Seleccion":        SeleccionTorneo(10, objetivo),
        "Reproduccion":     ReproduccionUnCruceRes(objetivo),
        "Mutacion":         MutacionRes(0.02),
        "por_remplazo":     0.5,
        "generaciones":     10_000,
        "objetivo":         objetivo,
        "stop":             "loss",
    }

    print(f"\n  Configuración:")
    print(f"    Genes por cromosoma : {CARACTERISTICAS}  "
          f"(W={ATRIBUTOS}×{CLASES} + b={CLASES})")
    print(f"    Individuos          : {cant_ind}")
    print(f"    Generaciones máx.   : {gaparams['generaciones']:,}")
    print(f"    Mutación (prob bit) : {gaparams['Mutacion'].prob}")
    print(f"    Bits por gen        : 16  →  resolución {2**16} niveles en [-1, 1]")

    # ── 5. Ejecutar AG ───────────────────────────────────────────────────────
    print("\nIniciando evolución …\n")
    algoritmoGA = GA(cant_ind, None, gaparams)
    algoritmoGA.buscar_solucion(1e-8, variables)

    # ── 6. Seleccionar mejor cromosoma ───────────────────────────────────────
    top10 = sorted(
        algoritmoGA.mejor_generacion[:10],
        key=lambda c: c.score,
        reverse=True,
    )
    topCrom = top10[0]
    print(f"\n  Mejor score (accuracy entrenamiento): {topCrom.score:.4f}  "
          f"({topCrom.score * 100:.2f}%)")

    # ── 7. Evaluación en conjunto de prueba ──────────────────────────────────
    yh  = regresion_cromosoma(topCrom, X_TEST, ATRIBUTOS, CLASES)
    acc = accuracy(yh, y_test)

    print(f"\n{'─'*50}")
    print(f"  Accuracy en TEST : {acc:.4f}  ({acc * 100:.2f}%)")
    print(f"{'─'*50}")

    # Tabla de predicciones
    nombres = ["Setosa", "Versicolor", "Virginica"]
    correctas = 0
    print(f"\n  {'#':>3}  {'Real':<14}  {'Predicción':<14}  {'✓/✗'}")
    print(f"  {'─'*3}  {'─'*14}  {'─'*14}  {'─'*3}")
    for i in range(len(y_test)):
        real = int(np.argmax(y_test[i]))
        pred = int(np.argmax(yh[i]))
        ok   = "✓" if real == pred else "✗"
        if real == pred:
            correctas += 1
        print(f"  {i+1:>3}  {nombres[real]:<14}  {nombres[pred]:<14}  {ok}")
    print(f"\n  Correctas: {correctas}/{len(y_test)}")

    # ── 8. Matriz de confusión ───────────────────────────────────────────────
    print("\nGenerando Matriz de Confusión …")
    mc = matriz_confusion(y_test, yh, CLASES)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(mc, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=nombres, yticklabels=nombres)
    ax.set_xlabel("Predicción del Modelo", fontsize=11)
    ax.set_ylabel("Etiqueta Real",          fontsize=11)
    ax.set_title(
        f"Iris — Clasificación con Algoritmo Genético\n"
        f"Accuracy = {acc:.4f}  ({acc * 100:.2f}%)",
        fontsize=12, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    plt.savefig("iris_confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("  Matriz guardada: iris_confusion_matrix.png")
    plt.show()
