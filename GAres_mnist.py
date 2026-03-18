"""
GAres_mnist.py — Clasificación MNIST con Algoritmo Genético
============================================================
Actividad de clase  ·  Sigue la estructura de GAres.py del profesor.

Problema original: 784 features × 10 clases → 7,850 genes  (tarda ~6 h sin optimizar)

3 optimizaciones aplicadas para hacerlo viable en clase (~20–30 min):
  1. get_all_valores()   — decodificación NumPy vectorizada       (7× más rápido)
  2. PCA (n=50)          — 784 → 50 features antes del AG        (15× menos genes)
  3. Mini-batch          — evalúa ~1,000 muestras por generación  (60× menos cómputo)

Genes resultantes: 50×10 + 10 = 510  (en lugar de 7,850)

Funciones del profesor reutilizadas (interfaz/regresion_classi.py):
  · regresion_cromosoma  (ahora usa get_all_valores internamente)
  · accuracy
  · onehot_encode
  · matriz_confusion
"""

from GAbin import GA
from CromosomaBin import CromosomaBinRes
from interfaz.aptitud import Aptitud
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ── Aptitud con mini-batch (CAMBIO 2) ────────────────────────────────────────

class AptitudMNISTBatch(Aptitud):
    """
    Variante de AptitudMNIST que evalúa sobre un mini-batch aleatorio
    en cada generación para reducir el coste computacional.

    El forward pass ya usa get_all_valores() vía regresion_cromosoma,
    que aplica el CAMBIO 1 (vectorización) automáticamente.
    """

    def __init__(self, atributos: int, clases: int,
                 objetivo: str = "maximizar", batch_size: int = 1000):
        self.atributos  = atributos
        self.clases     = clases
        self.objetivo   = objetivo
        self.batch_size = batch_size

    def evaluar_poblacion(self, poblacion, var):
        X_all = var["X_TRAIN"]
        Y_all = var["Y_TRAIN"]

        # Mini-batch aleatorio por generación (CAMBIO 2)
        idx     = np.random.choice(len(X_all), self.batch_size, replace=False)
        X_batch = X_all[idx]
        Y_batch = Y_all[idx]

        for cromosoma in poblacion:
            YH = regresion_cromosoma(cromosoma, X_batch, self.atributos, self.clases)
            acc = accuracy(YH, Y_batch)
            cromosoma.score = acc


# ── Utilidad: mostrar ejemplos mal clasificados ───────────────────────────────

def mostrar_errores(X_test_orig, y_test_oh, yh, n=10):
    """Muestra las primeras n imágenes mal clasificadas."""
    errores = [
        i for i in range(len(y_test_oh))
        if np.argmax(y_test_oh[i]) != np.argmax(yh[i])
    ]
    if not errores:
        print("  ¡Sin errores en el conjunto de prueba!")
        return

    n_show = min(n, len(errores))
    fig, axes = plt.subplots(2, n_show // 2, figsize=(n_show * 1.4, 5))
    axes = axes.flatten()
    for k, idx in enumerate(errores[:n_show]):
        axes[k].imshow(X_test_orig[idx].reshape(28, 28), cmap="gray")
        real = np.argmax(y_test_oh[idx])
        pred = np.argmax(yh[idx])
        axes[k].set_title(f"Real:{real}  Pred:{pred}", fontsize=8, color="red")
        axes[k].axis("off")
    fig.suptitle(f"MNIST — Primeros {n_show} errores del modelo AG",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("mnist_errores.png", dpi=130, bbox_inches="tight")
    print("  Errores guardados: mnist_errores.png")
    plt.show()


# ── Punto de entrada ─────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── 1. Descargar MNIST ───────────────────────────────────────────────────
    print("Descargando MNIST (puede tardar la primera vez) …")
    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    X_raw = mnist.data.astype(float)
    y_raw = mnist.target.astype(int)
    print(f"  Dataset: {X_raw.shape[0]} muestras  ×  {X_raw.shape[1]} pixels")

    # ── 2. División train/test ───────────────────────────────────────────────
    X_TRAIN_RAW, X_TEST_RAW, Y_TRAIN_INT, Y_TEST_INT = train_test_split(
        X_raw, y_raw, test_size=10_000, random_state=42, stratify=y_raw
    )
    print(f"  Entrenamiento: {X_TRAIN_RAW.shape[0]}  |  Prueba: {X_TEST_RAW.shape[0]}")

    # ── 3. PCA: 784 → 50 componentes  (CAMBIO 3) ────────────────────────────
    N_COMPONENTES = 50
    print(f"\nAplicando PCA: 784 → {N_COMPONENTES} componentes …")

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_TRAIN_RAW)
    X_test_sc  = scaler.transform(X_TEST_RAW)

    pca = PCA(n_components=N_COMPONENTES, random_state=42)
    X_TRAIN_PCA = pca.fit_transform(X_train_sc)
    X_TEST_PCA  = pca.transform(X_test_sc)

    varianza_acum = pca.explained_variance_ratio_.sum() * 100
    print(f"  Varianza acumulada: {varianza_acum:.1f}%  "
          f"con {N_COMPONENTES} componentes")

    # One-hot encode de etiquetas
    CLASES    = 10
    y_train   = onehot_encode(Y_TRAIN_INT, CLASES)
    y_test    = onehot_encode(Y_TEST_INT,  CLASES)

    variables = {"X_TRAIN": X_TRAIN_PCA, "Y_TRAIN": y_train}

    # ── 4. Pairplot de las 3 primeras componentes principales ───────────────
    print("\nGenerando pairplot (primeras 3 componentes PCA) …")
    import pandas as pd
    df_pca = pd.DataFrame(X_TRAIN_PCA[:2000, :3],
                          columns=["PC1", "PC2", "PC3"])
    df_pca["digito"] = Y_TRAIN_INT[:2000].astype(str)

    sns.set_theme(style="ticks")
    grid = sns.pairplot(df_pca, hue="digito", palette="tab10",
                        diag_kind="kde", plot_kws={"alpha": 0.3, "s": 10},
                        height=2.5)
    grid.figure.suptitle(
        f"MNIST — Primeras 3 componentes PCA  "
        f"(varianza acumulada: {varianza_acum:.1f}%)",
        y=1.02, fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    grid.figure.savefig("mnist_pairplot_pca.png", dpi=120, bbox_inches="tight")
    print("  Pairplot guardado: mnist_pairplot_pca.png")
    plt.show()
    sns.reset_defaults()

    # ── 5. Configuración del AG ──────────────────────────────────────────────
    ATRIBUTOS       = N_COMPONENTES         # 50
    CARACTERISTICAS = ATRIBUTOS * CLASES + CLASES   # 510  (vs 7,850 sin PCA)
    BATCH_SIZE      = 1000
    cant_ind        = 200
    objetivo        = "maximizar"

    gaparams = {
        "Cromosoma":        CromosomaBinRes,
        "Cromosoma_params": [CARACTERISTICAS, 16, -1, 1],
        "Aptitud":          AptitudMNISTBatch(ATRIBUTOS, CLASES,
                                              objetivo, BATCH_SIZE),
        "Seleccion":        SeleccionTorneo(10, objetivo),
        "Reproduccion":     ReproduccionUnCruceRes(objetivo),
        "Mutacion":         MutacionRes(0.02),
        "por_remplazo":     0.5,
        "generaciones":     10_000,
        "objetivo":         objetivo,
        "stop":             "loss",
    }

    print(f"\n  Configuración del AG:")
    print(f"    Features (post-PCA) : {ATRIBUTOS}")
    print(f"    Genes por cromosoma : {CARACTERISTICAS}  "
          f"(vs 7,850 sin PCA → {7850//CARACTERISTICAS}× reducción)")
    print(f"    Mini-batch size     : {BATCH_SIZE}")
    print(f"    Individuos          : {cant_ind}")
    print(f"    Generaciones máx.   : {gaparams['generaciones']:,}")
    print(f"    Bits por gen        : 16  (resolución {2**16} niveles)")

    # ── 6. Ejecutar AG ───────────────────────────────────────────────────────
    print("\nIniciando evolución …\n")
    algoritmoGA = GA(cant_ind, None, gaparams)
    algoritmoGA.buscar_solucion(1e-8, variables)

    # ── 7. Evaluación en test completo ───────────────────────────────────────
    top10 = sorted(
        algoritmoGA.mejor_generacion[:10],
        key=lambda c: c.score,
        reverse=True,
    )
    topCrom = top10[0]

    # Evaluar sobre todo el conjunto de prueba
    yh  = regresion_cromosoma(topCrom, X_TEST_PCA, ATRIBUTOS, CLASES)
    acc = accuracy(yh, y_test)

    print(f"\n{'─'*55}")
    print(f"  Accuracy en TEST (10,000 muestras): {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Accuracy por clase:")
    for c in range(CLASES):
        mask  = Y_TEST_INT == c
        acc_c = float(np.mean(np.argmax(yh[mask], axis=1) == c))
        print(f"    Dígito {c}: {acc_c:.3f}  ({int(acc_c*mask.sum())}/{mask.sum()})")
    print(f"{'─'*55}")

    # ── 8. Matriz de confusión ───────────────────────────────────────────────
    print("\nGenerando Matriz de Confusión …")
    mc = matriz_confusion(y_test, yh, CLASES)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(mc, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=list(range(10)),
                yticklabels=list(range(10)))
    ax.set_xlabel("Predicción del Modelo", fontsize=11)
    ax.set_ylabel("Dígito Real",            fontsize=11)
    ax.set_title(
        f"MNIST — Clasificación con Algoritmo Genético + PCA\n"
        f"Accuracy = {acc:.4f}  ({acc*100:.2f}%)",
        fontsize=12, fontweight="bold", pad=12,
    )
    plt.tight_layout()
    plt.savefig("mnist_confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("  Matriz guardada: mnist_confusion_matrix.png")
    plt.show()

    # ── 9. Ejemplos mal clasificados ─────────────────────────────────────────
    mostrar_errores(X_TEST_RAW, y_test, yh, n=10)
