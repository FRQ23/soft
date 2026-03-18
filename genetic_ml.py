"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  ALGORITMO GENÉTICO — APRENDIZAJE AUTOMÁTICO                               ║
║  Regresión Lineal Multivariable y Clasificación Binaria/Multiclase          ║
║  Codificación Real (float genes) · Operadores Evolutivos ML                 ║
╚══════════════════════════════════════════════════════════════════════════════╝

Arquitectura de clases
──────────────────────
  Cromosoma              — cromosoma de codificación real [w1,…,wn, b]
                           (.bits como propiedad retrocompatible para mochila)

  Aptitud  (ABC)         → AptitudMochila       — 0/1 Knapsack (retrocompat.)
                         → AptitudRegresion     — 1 / (MSE + ε)
                         → AptitudClasificacion — Accuracy (binaria/multiclase)

  Seleccion (ABC)        → SeleccionRango        — selección por rango
                         → SeleccionRuleta       — ruleta proporcional al fitness

  Reproduccion (ABC)     → ReproduccionDosPuntos — cruce de dos puntos (float)

  Mutacion (ABC)         → MutacionBitFlip       — flip para genes binarios
                         → MutacionGaussiana     — ruido gaussiano para genes reales

  DatosGeneracion        — snapshot por generación (persistencia JSON)
  AlgoritmoGenetico      — motor evolutivo genérico con dashboard visual
"""

import abc
import json
import random
import datetime
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
import seaborn as sns

from sklearn.datasets import load_iris, load_diabetes
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    mean_squared_error,
)


# ══════════════════════════════════════════════════════════════════════════════
#  CROMOSOMA  (Codificación Real)
# ══════════════════════════════════════════════════════════════════════════════

class Cromosoma:
    """
    Cromosoma de codificación real: cada gen es un número flotante.

    Para un modelo lineal representa:
      · Regresión / Clasificación binaria  → [w1, w2, …, wn, b]
      · Clasificación multiclase (k clases) → [W aplanado (n×k), b1, …, bk]

    La propiedad .bits (redondeo) mantiene compatibilidad con AptitudMochila
    y MutacionBitFlip del problema original de la mochila.
    """

    def __init__(
        self,
        n_genes: int,
        genes: Optional[List[float]] = None,
        init_range: Tuple[float, float] = (-1.0, 1.0),
        binary_init: bool = False,
    ):
        self.n_genes: int = n_genes
        if genes is not None:
            if len(genes) != n_genes:
                raise ValueError(
                    f"Longitud de genes {len(genes)} ≠ n_genes {n_genes}"
                )
            self.genes: List[float] = genes[:]
        else:
            if binary_init:
                self.genes = [float(random.randint(0, 1)) for _ in range(n_genes)]
            else:
                lo, hi = init_range
                self.genes = [random.uniform(lo, hi) for _ in range(n_genes)]
        self.fitness: float = 0.0

    # ── Retrocompatibilidad con la mochila ─────────────────────────────────

    @property
    def bits(self) -> List[int]:
        """Redondea genes flotantes a enteros 0/1 para operadores binarios."""
        return [int(round(max(0.0, min(1.0, g)))) for g in self.genes]

    @bits.setter
    def bits(self, values: List[int]) -> None:
        self.genes = [float(v) for v in values]

    # ── Utilidades ─────────────────────────────────────────────────────────

    def clone(self) -> "Cromosoma":
        """Copia profunda preservando el fitness."""
        c = Cromosoma(self.n_genes, self.genes)
        c.fitness = self.fitness
        return c

    def to_dict(self) -> Dict[str, Any]:
        return {"genes": self.genes, "fitness": self.fitness}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Cromosoma":
        """Soporta formato nuevo ('genes') y formato legado ('bits')."""
        if "genes" in d:
            genes_data = [float(g) for g in d["genes"]]
        else:
            genes_data = [float(b) for b in d["bits"]]
        c = cls(len(genes_data), genes_data)
        c.fitness = float(d["fitness"])
        return c

    def to_array(self) -> np.ndarray:
        return np.array(self.genes, dtype=float)

    def __repr__(self) -> str:
        preview = ", ".join(f"{g:.4f}" for g in self.genes[:5])
        suffix = f", … [{self.n_genes} genes]" if self.n_genes > 5 else ""
        return f"Cromosoma(fitness={self.fitness:.6f}, genes=[{preview}{suffix}])"


# ══════════════════════════════════════════════════════════════════════════════
#  INTERFACES ABSTRACTAS  (jerarquía de clases modular original)
# ══════════════════════════════════════════════════════════════════════════════

class Aptitud(abc.ABC):
    """Evaluador de aptitud (fitness). Interfaz abstracta."""

    @abc.abstractmethod
    def evaluar(self, cromosoma: Cromosoma) -> float:
        """Retorna el fitness escalar de un cromosoma individual."""

    def evaluar_poblacion(self, poblacion: List[Cromosoma]) -> None:
        """Evalúa y asigna fitness a toda la población."""
        for c in poblacion:
            c.fitness = self.evaluar(c)


class Seleccion(abc.ABC):
    """Operador de selección de padres. Interfaz abstracta."""

    @abc.abstractmethod
    def seleccionar(self, poblacion: List[Cromosoma], n: int) -> List[Cromosoma]:
        """Retorna *n* cromosomas seleccionados (clonados) de la población."""


class Reproduccion(abc.ABC):
    """Operador de cruce / recombinación. Interfaz abstracta."""

    @abc.abstractmethod
    def cruzar(
        self, padre: Cromosoma, madre: Cromosoma
    ) -> Tuple[Cromosoma, Cromosoma]:
        """Produce dos hijos a partir de dos padres."""


class Mutacion(abc.ABC):
    """Operador de mutación. Interfaz abstracta."""

    @abc.abstractmethod
    def mutar(self, cromosoma: Cromosoma) -> Cromosoma:
        """Retorna una copia (posiblemente mutada) del cromosoma."""


# ══════════════════════════════════════════════════════════════════════════════
#  FITNESS — Mochila  (retrocompatible, usa propiedad .bits)
# ══════════════════════════════════════════════════════════════════════════════

class AptitudMochila(Aptitud):
    """
    Función de aptitud para el problema de la mochila 0/1 con penalización suave.

        score = max(0, valor_total − penalización)
        penalización = (peso_total − capacidad) × factor  [solo si sobrepasa]
    """

    def __init__(
        self,
        weights: List[float],
        values: List[float],
        capacity: float,
        penalty_factor: float = 10.0,
    ):
        if len(weights) != len(values):
            raise ValueError("weights y values deben tener la misma longitud.")
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.penalty_factor = penalty_factor

    def evaluar(self, cromosoma: Cromosoma) -> float:
        bits = cromosoma.bits
        total_weight = sum(w * b for w, b in zip(self.weights, bits))
        total_value  = sum(v * b for v, b in zip(self.values,  bits))
        if total_weight > self.capacity:
            penalty = (total_weight - self.capacity) * self.penalty_factor
            return max(0.0, total_value - penalty)
        return float(total_value)

    def peso_total(self, cromosoma: Cromosoma) -> float:
        return sum(w * b for w, b in zip(self.weights, cromosoma.bits))

    def valor_total(self, cromosoma: Cromosoma) -> float:
        return sum(v * b for v, b in zip(self.values, cromosoma.bits))


# ══════════════════════════════════════════════════════════════════════════════
#  FITNESS — Regresión Lineal Multivariable
# ══════════════════════════════════════════════════════════════════════════════

class AptitudRegresion(Aptitud):
    """
    Fitness basado en MSE invertido para regresión lineal multivariable.

    Modelo:   ŷ = X · w + b
    Genes:    [w1, w2, …, w_n, b]   (longitud = n_features + 1)

    Fitness:  1 / (MSE + ε)   →  maximizar (cuanto menor MSE, mayor fitness)

    Parameters
    ----------
    X_train : ndarray, shape (m, n)
    y_train : ndarray, shape (m,)
    epsilon : float — evita división por cero cuando MSE ≈ 0
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epsilon: float = 1e-8,
    ):
        self.X_train    = np.asarray(X_train, dtype=float)
        self.y_train    = np.asarray(y_train, dtype=float)
        self.epsilon    = epsilon
        self.n_features = X_train.shape[1]

    def evaluar(self, cromosoma: Cromosoma) -> float:
        genes = cromosoma.to_array()
        w = genes[: self.n_features]
        b = genes[self.n_features]
        y_pred = self.X_train @ w + b
        mse = float(np.mean((self.y_train - y_pred) ** 2))
        return 1.0 / (mse + self.epsilon)

    def predecir(self, X: np.ndarray, cromosoma: Cromosoma) -> np.ndarray:
        """Predicción sobre un conjunto externo X."""
        genes = cromosoma.to_array()
        w = genes[: self.n_features]
        b = genes[self.n_features]
        return np.asarray(X, dtype=float) @ w + b

    def mse(self, X: np.ndarray, y: np.ndarray, cromosoma: Cromosoma) -> float:
        y_pred = self.predecir(X, cromosoma)
        return float(np.mean((np.asarray(y, dtype=float) - y_pred) ** 2))


# ══════════════════════════════════════════════════════════════════════════════
#  FITNESS — Clasificación Binaria / Multiclase
# ══════════════════════════════════════════════════════════════════════════════

class AptitudClasificacion(Aptitud):
    """
    Fitness basado en Accuracy para clasificación lineal.

    Binaria (n_classes=2)
    ─────────────────────
      Genes:  [w1,…,wn, b]          longitud = n_features + 1
      Modelo: σ(X·w + b) ≥ 0.5  →  clase 1

    Multiclase (n_classes=k)
    ────────────────────────
      Genes:  [W aplanado (n×k), b1,…,bk]   longitud = n_features*k + k
      Modelo: argmax(softmax(X·W + b))

    Fitness = Accuracy en entrenamiento ∈ [0, 1]

    Parameters
    ----------
    X_train    : ndarray, shape (m, n)
    y_train    : ndarray, shape (m,)   — etiquetas enteras 0, 1, …, k-1
    n_classes  : int — número de clases del problema
    """

    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        n_classes: int = 2,
    ):
        self.X_train    = np.asarray(X_train, dtype=float)
        self.y_train    = np.asarray(y_train, dtype=int)
        self.n_features = X_train.shape[1]
        self.n_classes  = n_classes

    # ── Funciones de activación ────────────────────────────────────────────

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z_shift = z - z.max(axis=1, keepdims=True)
        exp_z   = np.exp(z_shift)
        return exp_z / exp_z.sum(axis=1, keepdims=True)

    # ── Predicción ─────────────────────────────────────────────────────────

    def predecir(self, X: np.ndarray, cromosoma: Cromosoma) -> np.ndarray:
        """Retorna etiquetas predichas para X."""
        X    = np.asarray(X, dtype=float)
        n, k = self.n_features, self.n_classes
        genes = cromosoma.to_array()

        if k == 2:
            w    = genes[:n]
            b    = genes[n]
            prob = self._sigmoid(X @ w + b)
            return (prob >= 0.5).astype(int)
        else:
            W      = genes[: n * k].reshape(n, k)
            b      = genes[n * k :]
            logits = X @ W + b
            return np.argmax(self._softmax(logits), axis=1)

    def evaluar(self, cromosoma: Cromosoma) -> float:
        y_pred = self.predecir(self.X_train, cromosoma)
        return float(np.mean(y_pred == self.y_train))

    def accuracy(self, X: np.ndarray, y: np.ndarray, cromosoma: Cromosoma) -> float:
        """Accuracy en un conjunto externo."""
        y_pred = self.predecir(X, cromosoma)
        return float(np.mean(y_pred == np.asarray(y, dtype=int)))

    @property
    def n_genes(self) -> int:
        """Longitud requerida del cromosoma para este problema."""
        if self.n_classes == 2:
            return self.n_features + 1
        return self.n_features * self.n_classes + self.n_classes


# ══════════════════════════════════════════════════════════════════════════════
#  SELECCIÓN — Por Rango
# ══════════════════════════════════════════════════════════════════════════════

class SeleccionRango(Seleccion):
    """
    Selección basada en rango (Rank-Based).

    1. Ordena la población de peor (rango 1) a mejor (rango N).
    2. Probabilidad de selección ∝ rango  →  evita convergencia prematura.
    """

    def seleccionar(self, poblacion: List[Cromosoma], n: int) -> List[Cromosoma]:
        sorted_pop = sorted(poblacion, key=lambda c: c.fitness)
        N          = len(sorted_pop)
        ranks      = list(range(1, N + 1))
        sum_ranks  = sum(ranks)
        top_rank   = ranks[-1]
        exp_cnt    = [top_rank * (r / sum_ranks) for r in ranks]
        selected   = random.choices(sorted_pop, weights=exp_cnt, k=n)
        return [c.clone() for c in selected]


# ══════════════════════════════════════════════════════════════════════════════
#  SELECCIÓN — Ruleta Estocástica
# ══════════════════════════════════════════════════════════════════════════════

class SeleccionRuleta(Seleccion):
    """
    Selección por Ruleta Proporcional al Fitness.

    P(i) = fitness(i) / Σ fitness
    Si todos los fitness son 0 usa distribución uniforme.
    """

    def seleccionar(self, poblacion: List[Cromosoma], n: int) -> List[Cromosoma]:
        total = sum(c.fitness for c in poblacion)
        if total == 0.0:
            return [random.choice(poblacion).clone() for _ in range(n)]
        weights  = [c.fitness / total for c in poblacion]
        selected = random.choices(poblacion, weights=weights, k=n)
        return [c.clone() for c in selected]


# ══════════════════════════════════════════════════════════════════════════════
#  CRUCE — Dos Puntos
# ══════════════════════════════════════════════════════════════════════════════

class ReproduccionDosPuntos(Reproduccion):
    """
    Cruce de Dos Puntos.

    Elige dos puntos de corte p1 < p2 al azar.  Intercambia el segmento
    central entre padres para generar dos hijos.

        h1 = padre[:p1] + madre[p1:p2] + padre[p2:]
        h2 = madre[:p1] + padre[p1:p2] + madre[p2:]

    Funciona sin cambios con genes flotantes ya que sólo opera sobre índices.
    """

    def cruzar(
        self, padre: Cromosoma, madre: Cromosoma
    ) -> Tuple[Cromosoma, Cromosoma]:
        n = padre.n_genes
        if n < 3:
            return padre.clone(), madre.clone()
        p1, p2 = sorted(random.sample(range(1, n), 2))
        g_h1 = padre.genes[:p1] + madre.genes[p1:p2] + padre.genes[p2:]
        g_h2 = madre.genes[:p1] + padre.genes[p1:p2] + madre.genes[p2:]
        return Cromosoma(n, g_h1), Cromosoma(n, g_h2)


# ══════════════════════════════════════════════════════════════════════════════
#  MUTACIÓN — Bit-Flip  (para problemas binarios / mochila)
# ══════════════════════════════════════════════════════════════════════════════

class MutacionBitFlip(Mutacion):
    """
    Mutación por inversión de bit para genes de codificación binaria.

    Cada gen se invierte (0 → 1 ó 1 → 0) con probabilidad *prob_mutacion*.
    Funciona sobre genes flotantes ∈ {0.0, 1.0} usando la propiedad .bits.
    """

    def __init__(self, prob_mutacion: float):
        if not 0.0 <= prob_mutacion <= 1.0:
            raise ValueError("prob_mutacion debe estar en [0, 1].")
        self.prob_mutacion = prob_mutacion

    def mutar(self, cromosoma: Cromosoma) -> Cromosoma:
        nuevo = cromosoma.clone()
        for i in range(nuevo.n_genes):
            if random.random() < self.prob_mutacion:
                bit_actual    = int(round(max(0.0, min(1.0, nuevo.genes[i]))))
                nuevo.genes[i] = float(1 - bit_actual)
        return nuevo


# ══════════════════════════════════════════════════════════════════════════════
#  MUTACIÓN — Gaussiana  (para problemas de codificación real)
# ══════════════════════════════════════════════════════════════════════════════

class MutacionGaussiana(Mutacion):
    """
    Mutación Gaussiana para cromosomas de codificación real.

    Cada gen recibe una perturbación aditiva de una distribución normal:
        gen_i  ←  gen_i + N(0, σ)   con probabilidad *prob_mutacion*

    Ideal para espacios de búsqueda continuos como los pesos de un modelo ML.

    Parameters
    ----------
    prob_mutacion : float — probabilidad de mutar cada gen individual (∈ [0,1])
    sigma         : float — desviación estándar del ruido gaussiano
    clip_range    : tuple[float, float] | None
        Si se especifica, recorta los genes mutados al rango dado.
        Útil para mantener los pesos dentro de un rango acotado.
    """

    def __init__(
        self,
        prob_mutacion: float,
        sigma: float = 0.1,
        clip_range: Optional[Tuple[float, float]] = None,
    ):
        if not 0.0 <= prob_mutacion <= 1.0:
            raise ValueError("prob_mutacion debe estar en [0, 1].")
        if sigma <= 0:
            raise ValueError("sigma debe ser positivo.")
        self.prob_mutacion = prob_mutacion
        self.sigma         = sigma
        self.clip_range    = clip_range

    def mutar(self, cromosoma: Cromosoma) -> Cromosoma:
        nuevo = cromosoma.clone()
        for i in range(nuevo.n_genes):
            if random.random() < self.prob_mutacion:
                nuevo.genes[i] += random.gauss(0.0, self.sigma)
                if self.clip_range is not None:
                    lo, hi = self.clip_range
                    nuevo.genes[i] = max(lo, min(hi, nuevo.genes[i]))
        return nuevo


# ══════════════════════════════════════════════════════════════════════════════
#  DATOS DE GENERACIÓN  (modelo de persistencia)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatosGeneracion:
    """Snapshot de una generación para persistencia y análisis de contraste."""

    numero:      int
    individuos:  List[Dict[str, Any]]
    max_fitness: float
    avg_fitness: float
    min_fitness: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "numero":      self.numero,
            "max_fitness": self.max_fitness,
            "avg_fitness": self.avg_fitness,
            "min_fitness": self.min_fitness,
            "individuos":  self.individuos,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DatosGeneracion":
        return cls(
            numero      = d["numero"],
            individuos  = d["individuos"],
            max_fitness = d["max_fitness"],
            avg_fitness = d["avg_fitness"],
            min_fitness = d["min_fitness"],
        )

    def mejor_individuo(self) -> Cromosoma:
        best = max(self.individuos, key=lambda x: x["fitness"])
        return Cromosoma.from_dict(best)

    def todos_los_individuos(self) -> List[Cromosoma]:
        return [Cromosoma.from_dict(ind) for ind in self.individuos]


# ══════════════════════════════════════════════════════════════════════════════
#  MOTOR DEL ALGORITMO GENÉTICO
# ══════════════════════════════════════════════════════════════════════════════

class AlgoritmoGenetico:
    """
    Motor evolutivo genérico.

    Responsabilidades
    ─────────────────
    • Inicializar población aleatoria (float o binaria según init_range/binary_init).
    • Ejecutar el bucle evolutivo por *n_generaciones* generaciones.
    • Rastrear y persistir la Mejor Generación (mayor avg fitness) en JSON.
    • Generar reporte de contraste en consola.
    • Producir dashboard visual en Matplotlib.

    Parameters
    ----------
    n_genes        : número de genes por cromosoma
    tam_poblacion  : tamaño de la población
    aptitud        : instancia de Aptitud (evaluador de fitness)
    seleccion      : instancia de Seleccion
    reproduccion   : instancia de Reproduccion
    mutacion       : instancia de Mutacion
    tasa_reemplazo : fracción de la población reemplazada por generación
    n_generaciones : número de generaciones
    init_range     : rango de inicialización de genes flotantes
    binary_init    : si True, inicializa genes como 0.0 / 1.0
    nombre_problema: texto para los encabezados y gráficas
    archivo_mejor  : ruta del JSON de persistencia
    semilla        : semilla aleatoria para reproducibilidad
    """

    def __init__(
        self,
        n_genes:          int,
        tam_poblacion:    int,
        aptitud:          Aptitud,
        seleccion:        Seleccion,
        reproduccion:     Reproduccion,
        mutacion:         Mutacion,
        tasa_reemplazo:   float = 0.50,
        n_generaciones:   int   = 200,
        init_range:       Tuple[float, float] = (-1.0, 1.0),
        binary_init:      bool  = False,
        nombre_problema:  str   = "Aprendizaje Automático",
        archivo_mejor:    str   = "mejor_generacion.json",
        semilla:          Optional[int] = None,
    ):
        if semilla is not None:
            random.seed(semilla)
            np.random.seed(semilla)

        self.n_genes          = n_genes
        self.tam_poblacion    = tam_poblacion
        self.aptitud          = aptitud
        self.seleccion        = seleccion
        self.reproduccion     = reproduccion
        self.mutacion         = mutacion
        self.tasa_reemplazo   = tasa_reemplazo
        self.n_generaciones   = n_generaciones
        self.init_range       = init_range
        self.binary_init      = binary_init
        self.nombre_problema  = nombre_problema
        self.archivo_mejor    = archivo_mejor

        self.poblacion: List[Cromosoma] = []

        self.historial_avg: List[float] = []
        self.historial_max: List[float] = []
        self.historial_min: List[float] = []

        self.mejor_generacion: Optional[DatosGeneracion] = None
        self.ultima_generacion: Optional[DatosGeneracion] = None

    # ── Helpers privados ───────────────────────────────────────────────────

    def _inicializar_poblacion(self) -> None:
        self.poblacion = [
            Cromosoma(
                self.n_genes,
                init_range   = self.init_range,
                binary_init  = self.binary_init,
            )
            for _ in range(self.tam_poblacion)
        ]

    def _estadisticas(
        self, poblacion: List[Cromosoma]
    ) -> Tuple[float, float, float]:
        fits = [c.fitness for c in poblacion]
        return max(fits), sum(fits) / len(fits), min(fits)

    def _capturar_generacion(self, numero: int) -> DatosGeneracion:
        mx, avg, mn = self._estadisticas(self.poblacion)
        return DatosGeneracion(
            numero      = numero,
            individuos  = [c.to_dict() for c in self.poblacion],
            max_fitness = mx,
            avg_fitness = avg,
            min_fitness = mn,
        )

    def _guardar_mejor_generacion(self, gen: DatosGeneracion) -> None:
        with open(self.archivo_mejor, "w", encoding="utf-8") as f:
            json.dump(gen.to_dict(), f, indent=2, ensure_ascii=False)

    def _paso_evolutivo(self) -> None:
        """Selección → Cruce → Mutación → Reemplazo elitista."""
        n_hijos = int(self.tam_poblacion * self.tasa_reemplazo)
        if n_hijos % 2 != 0:
            n_hijos += 1

        hijos: List[Cromosoma] = []
        while len(hijos) < n_hijos:
            padres = self.seleccion.seleccionar(self.poblacion, 2)
            h1, h2 = self.reproduccion.cruzar(padres[0], padres[1])
            h1 = self.mutacion.mutar(h1)
            h2 = self.mutacion.mutar(h2)
            self.aptitud.evaluar_poblacion([h1, h2])
            hijos.extend([h1, h2])

        # Reemplazo elitista: eliminar los *n_hijos* peores
        self.poblacion.sort(key=lambda c: c.fitness)
        self.poblacion[:n_hijos] = hijos[:n_hijos]

    # ── API pública ────────────────────────────────────────────────────────

    def ejecutar(self) -> None:
        """Ejecuta el proceso evolutivo completo."""
        self._banner()
        self._inicializar_poblacion()
        self.aptitud.evaluar_poblacion(self.poblacion)

        for gen_num in range(1, self.n_generaciones + 1):
            datos = self._capturar_generacion(gen_num)

            self.historial_max.append(datos.max_fitness)
            self.historial_avg.append(datos.avg_fitness)
            self.historial_min.append(datos.min_fitness)

            if (
                self.mejor_generacion is None
                or datos.avg_fitness > self.mejor_generacion.avg_fitness
            ):
                self.mejor_generacion = datos
                self._guardar_mejor_generacion(datos)

            if gen_num % 25 == 0 or gen_num == 1:
                print(
                    f"  Gen {gen_num:>4} │ "
                    f"Max: {datos.max_fitness:>10.6f}  "
                    f"Avg: {datos.avg_fitness:>10.6f}  "
                    f"Min: {datos.min_fitness:>10.6f}"
                )

            self._paso_evolutivo()

        self.aptitud.evaluar_poblacion(self.poblacion)
        self.ultima_generacion = self._capturar_generacion(self.n_generaciones)

        print("\n  Evolución completada.")
        print(
            f"  ✔  Mejor generación (#{self.mejor_generacion.numero}) "
            f"persistida en: '{self.archivo_mejor}'"
        )

    # ── Métricas de diversidad ─────────────────────────────────────────────

    @staticmethod
    def similitud_coseno(c1: Cromosoma, c2: Cromosoma) -> float:
        """Similitud coseno ∈ [−1, 1] entre dos vectores de genes."""
        v1 = c1.to_array()
        v2 = c2.to_array()
        norm1, norm2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    @staticmethod
    def distancia_l2(c1: Cromosoma, c2: Cromosoma) -> float:
        """Distancia Euclidiana entre dos cromosomas."""
        return float(np.linalg.norm(c1.to_array() - c2.to_array()))

    @staticmethod
    def diversidad_poblacion(individuos: List[Cromosoma]) -> float:
        """
        Diversidad media de la población como distancia L2 normalizada
        entre todos los pares (promedio de distancias inter-individuo).
        """
        n = len(individuos)
        if n < 2:
            return 0.0
        total, count = 0.0, 0
        for i in range(n):
            for j in range(i + 1, n):
                total += AlgoritmoGenetico.distancia_l2(individuos[i], individuos[j])
                count += 1
        return total / count if count > 0 else 0.0

    # ── Reporte en consola ─────────────────────────────────────────────────

    def _banner(self) -> None:
        sel_name = type(self.seleccion).__name__
        rep_name = type(self.reproduccion).__name__
        mut_name = type(self.mutacion).__name__
        print("\n" + "═" * 68)
        print(f"  ALGORITMO GENÉTICO — {self.nombre_problema.upper()}")
        print("═" * 68)
        print(f"  Genes (dimensión)    : {self.n_genes}")
        print(f"  Tamaño de población  : {self.tam_poblacion}")
        print(f"  Generaciones         : {self.n_generaciones}")
        print(f"  Tasa de reemplazo    : {self.tasa_reemplazo * 100:.0f}%")
        print(f"  Rango de inicializ.  : {self.init_range}")
        print(f"  Selección            : {sel_name}")
        print(f"  Reproducción         : {rep_name}")
        print(f"  Mutación             : {mut_name}")
        print("═" * 68)

    def reporte_contraste(self) -> None:
        """
        Reporte comparativo entre la Mejor Generación y la Generación Final,
        incluyendo métricas de similitud genética basadas en coseno y norma L2.
        """
        if self.mejor_generacion is None or self.ultima_generacion is None:
            print("  [!] Ejecuta 'ejecutar()' antes de pedir el reporte.")
            return

        mg     = self.mejor_generacion
        ug     = self.ultima_generacion
        bi_mg  = mg.mejor_individuo()
        bi_ug  = ug.mejor_individuo()
        coseno = self.similitud_coseno(bi_mg, bi_ug)
        dist   = self.distancia_l2(bi_mg, bi_ug)

        delta_avg = ug.avg_fitness - mg.avg_fitness
        delta_max = ug.max_fitness - mg.max_fitness

        print("\n" + "═" * 68)
        print("  CONTRASTE  —  MEJOR GENERACIÓN  vs  GENERACIÓN FINAL")
        print("═" * 68)
        print(f"  {'Métrica':<30} {'Mejor Gen':>14} {'Gen Final':>14}")
        print(f"  {'─' * 30} {'─' * 14} {'─' * 14}")
        print(f"  {'Número de generación':<30} {mg.numero:>14d} {ug.numero:>14d}")
        print(f"  {'Fitness Máximo':<30} {mg.max_fitness:>14.6f} {ug.max_fitness:>14.6f}")
        print(f"  {'Fitness Promedio':<30} {mg.avg_fitness:>14.6f} {ug.avg_fitness:>14.6f}")
        print(f"  {'Fitness Mínimo':<30} {mg.min_fitness:>14.6f} {ug.min_fitness:>14.6f}")

        print(f"\n  {'── Δ desde mejor generación ──':^56}")
        print(f"  {'Δ Fitness Promedio':<30} {delta_avg:>+14.6f}")
        print(f"  {'Δ Fitness Máximo':<30} {delta_max:>+14.6f}")

        print(f"\n  {'── Similitud genética (genes reales) ──':^56}")
        print(f"  Similitud Coseno  : {coseno:>8.4f}   (1 = idénticos, -1 = opuestos)")
        print(f"  Distancia L2      : {dist:>8.4f}")

        bar_len = 40
        pct     = max(0.0, min(1.0, (coseno + 1) / 2))
        filled  = round(pct * bar_len)
        bar     = "█" * filled + "░" * (bar_len - filled)
        print(f"  Similitud coseno  : [{bar}]  {coseno:.4f}")
        print("═" * 68)

    # ── Dashboard visual ───────────────────────────────────────────────────

    def graficar(
        self,
        titulo_extra: str = "",
        guardar: bool = True,
        nombre_archivo: str = "",
    ) -> None:
        """
        Dashboard Matplotlib: evolución de fitness y error relativo.

        Plot 1 — Fitness Máximo / Promedio / Mínimo por generación.
        Plot 2 — Error relativo del promedio respecto al mejor global (%).
        """
        gens    = list(range(1, len(self.historial_avg) + 1))
        ref_max = max(self.historial_max)

        err_rel = [
            abs(ref_max - v) / ref_max * 100 if ref_max > 0 else 0.0
            for v in self.historial_avg
        ]
        best_gen_x = self.mejor_generacion.numero if self.mejor_generacion else 1

        BG_DARK  = "#0a0a12"
        BG_PANEL = "#12122a"
        C_AVG    = "#00e5ff"
        C_MAX    = "#ff6b35"
        C_MIN    = "#b2ff59"
        C_ERR    = "#ff4081"
        C_BEST   = "#ffd740"
        C_GRID   = "#1e1e3a"

        plt.rcParams.update({
            "font.family":      "monospace",
            "axes.facecolor":   BG_PANEL,
            "figure.facecolor": BG_DARK,
            "text.color":       "white",
            "axes.labelcolor":  "#cccccc",
            "xtick.color":      "#888888",
            "ytick.color":      "#888888",
            "axes.edgecolor":   "#333355",
            "axes.grid":        True,
            "grid.color":       C_GRID,
            "grid.linewidth":   0.5,
        })

        fig = plt.figure(figsize=(17, 7.5), facecolor=BG_DARK)
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)

        # ── Fitness vs Generaciones ────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.fill_between(
            gens, self.historial_min, self.historial_max,
            alpha=0.12, color=C_AVG, label="Rango [Min – Max]",
        )
        ax1.plot(gens, self.historial_max, color=C_MAX, lw=1.4,
                 ls="--", alpha=0.75, label="Fitness Máximo")
        ax1.plot(gens, self.historial_min, color=C_MIN, lw=1.0,
                 ls=":", alpha=0.55, label="Fitness Mínimo")
        ax1.plot(gens, self.historial_avg, color=C_AVG, lw=2.8,
                 label="Fitness Promedio")
        ax1.axvline(best_gen_x, color=C_BEST, lw=1.8, ls="--", alpha=0.9,
                    label=f"Mejor Gen (#{best_gen_x})")
        ax1.scatter([best_gen_x], [self.historial_avg[best_gen_x - 1]],
                    color=C_BEST, s=120, zorder=6)
        ax1.set_title("Fitness por Generación", color="white",
                      fontsize=14, fontweight="bold", pad=14)
        ax1.set_xlabel("Generación", fontsize=11)
        ax1.set_ylabel("Fitness", fontsize=11)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax1.legend(facecolor="#0d0d1e", labelcolor="white",
                   fontsize=8.5, framealpha=0.85, loc="lower right")

        idx_best = self.historial_max.index(ref_max)
        ax1.annotate(
            f" Global Max\n {ref_max:.4f}",
            xy=(gens[idx_best], ref_max),
            xytext=(20, -30), textcoords="offset points",
            color=C_MAX, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=C_MAX, lw=1.2),
        )

        # ── Error Relativo ─────────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.fill_between(gens, err_rel, alpha=0.20, color=C_ERR)
        ax2.plot(gens, err_rel, color=C_ERR, lw=2.8, label="Error Relativo (%)")
        ax2.axvline(best_gen_x, color=C_BEST, lw=1.8, ls="--", alpha=0.9,
                    label=f"Mejor Gen (#{best_gen_x})")
        ax2.axhline(0, color="#ffffff", lw=0.6, alpha=0.3)
        ax2.set_title("Error Relativo por Generación", color="white",
                      fontsize=14, fontweight="bold", pad=14)
        ax2.set_xlabel("Generación", fontsize=11)
        ax2.set_ylabel("Error Relativo (%)", fontsize=11)
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax2.legend(facecolor="#0d0d1e", labelcolor="white",
                   fontsize=8.5, framealpha=0.85)

        fig.suptitle(
            f"Algoritmo Genético — {self.nombre_problema}  {titulo_extra}",
            color="white", fontsize=15, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        if guardar:
            ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = nombre_archivo or f"ga_evolucion_{ts}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
            print(f"  ✔  Dashboard guardado en: '{fname}'")

        plt.show()
        plt.rcParams.update(plt.rcParamsDefault)


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE DATOS
# ══════════════════════════════════════════════════════════════════════════════

def cargar_iris() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, List[str]]:
    """
    Carga el dataset Iris (UCI) desde sklearn y devuelve un DataFrame
    junto con los arrays X, y y los nombres de clases.

    Returns
    -------
    df           : DataFrame con columnas de features + 'especie'
    X            : ndarray (150, 4) — features sin escalar
    y            : ndarray (150,)   — etiquetas enteras 0/1/2
    target_names : lista de nombres de clase ['setosa', 'versicolor', 'virginica']
    """
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df["especie"] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df, iris.data, iris.target, list(iris.target_names)


def cargar_diabetes() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Carga el dataset Diabetes (sklearn) para regresión.

    Returns
    -------
    df : DataFrame con 10 features + 'target'
    X  : ndarray (442, 10) — features sin escalar
    y  : ndarray (442,)    — valor cuantitativo de progresión
    """
    data = load_diabetes()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df, data.data, data.target


def preprocesar(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Aplica StandardScaler y divide en entrenamiento/prueba.

    Parameters
    ----------
    X            : features originales
    y            : etiquetas / valores objetivo
    test_size    : fracción del conjunto de prueba
    random_state : semilla para reproducibilidad

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=None
    )
    return X_train, X_test, y_train, y_test, scaler


# ══════════════════════════════════════════════════════════════════════════════
#  UTILIDADES DE VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════

def visualizar_pairplot(
    df: pd.DataFrame,
    hue_col: str,
    titulo: str = "Pairplot del Dataset",
    guardar: bool = True,
    nombre_archivo: str = "pairplot.png",
) -> None:
    """
    Genera un pairplot con seaborn para exploración inicial del dataset.

    Parameters
    ----------
    df            : DataFrame con features y columna de color (hue)
    hue_col       : nombre de la columna usada como color
    titulo        : título del gráfico
    guardar       : si True guarda la figura como PNG
    nombre_archivo: nombre del archivo de salida
    """
    print(f"\n  Generando pairplot ({hue_col}) …")
    sns.set_theme(style="ticks", palette="deep")
    grid = sns.pairplot(df, hue=hue_col, diag_kind="kde", plot_kws={"alpha": 0.6})
    grid.figure.suptitle(titulo, y=1.02, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if guardar:
        grid.figure.savefig(nombre_archivo, dpi=130, bbox_inches="tight")
        print(f"  ✔  Pairplot guardado en: '{nombre_archivo}'")
    plt.show()
    sns.reset_defaults()


def visualizar_confusion(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    nombres_clases: List[str],
    titulo: str = "Matriz de Confusión",
    guardar: bool = True,
    nombre_archivo: str = "confusion_matrix.png",
) -> None:
    """
    Genera y muestra la Matriz de Confusión con etiquetas de clase.

    Parameters
    ----------
    y_true         : etiquetas reales
    y_pred         : etiquetas predichas por el mejor cromosoma
    nombres_clases : lista de nombres de clase en orden numérico
    """
    cm   = confusion_matrix(y_true, y_pred)
    acc  = accuracy_score(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=nombres_clases,
    )
    disp.plot(ax=ax, colormap="Blues", values_format="d", xticks_rotation=30)
    ax.set_title(f"{titulo}\nAccuracy = {acc:.4f} ({acc * 100:.2f}%)",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    if guardar:
        plt.savefig(nombre_archivo, dpi=150, bbox_inches="tight")
        print(f"  ✔  Matriz de confusión guardada en: '{nombre_archivo}'")
    plt.show()


def visualizar_regresion(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    titulo: str = "Regresión — Predicho vs Real",
    guardar: bool = True,
    nombre_archivo: str = "regresion_resultado.png",
) -> None:
    """
    Genera dos gráficas para evaluar el modelo de regresión:
      1. Scatter: valores reales vs predichos (línea diagonal ideal).
      2. Residuos: índice de muestra vs error de predicción.

    Parameters
    ----------
    y_test         : valores reales del conjunto de prueba
    y_pred         : valores predichos por el mejor cromosoma
    """
    residuos = y_test - y_pred
    mse      = float(np.mean(residuos ** 2))
    rmse     = math.sqrt(mse)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ── Scatter: Real vs Predicho ──────────────────────────────────────────
    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.55, color="#4A90D9", edgecolors="white",
               linewidths=0.3, s=50, label="Muestras")
    lo = min(y_test.min(), y_pred.min()) * 0.95
    hi = max(y_test.max(), y_pred.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.5, label="Predicción perfecta")
    ax.set_xlabel("Valor Real", fontsize=11)
    ax.set_ylabel("Valor Predicho", fontsize=11)
    ax.set_title("Real vs Predicho", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Residuos ───────────────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.scatter(range(len(residuos)), residuos, alpha=0.55, color="#E67E22",
                edgecolors="white", linewidths=0.3, s=50)
    ax2.axhline(0, color="red", lw=1.5, ls="--", label="Error = 0")
    ax2.set_xlabel("Índice de muestra", fontsize=11)
    ax2.set_ylabel("Residuo (Real − Predicho)", fontsize=11)
    ax2.set_title("Residuos", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"{titulo}  |  MSE = {mse:.2f}  ·  RMSE = {rmse:.2f}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if guardar:
        plt.savefig(nombre_archivo, dpi=150, bbox_inches="tight")
        print(f"  ✔  Gráfica de regresión guardada en: '{nombre_archivo}'")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  EJEMPLO — CLASIFICACIÓN MULTICLASE (Iris)
# ══════════════════════════════════════════════════════════════════════════════

def main_clasificacion() -> None:
    """
    Ejemplo completo de clasificación multiclase con el dataset Iris.

    Configuración del cromosoma para 3 clases y 4 features:
      ┌─────────────────────────────────────────────────────┐
      │  genes = [W aplanado (4×3 = 12), b1, b2, b3]       │
      │  longitud total = 4 × 3 + 3 = 15 genes             │
      │  Modelo: ŷ = argmax(softmax(X · W + b))            │
      └─────────────────────────────────────────────────────┘
    """
    print("\n" + "═" * 68)
    print("  EJEMPLO — CLASIFICACIÓN IRIS (AG + Codificación Real)")
    print("═" * 68)

    # ── 1. Cargar y explorar el dataset ────────────────────────────────────
    df, X, y, nombres_clases = cargar_iris()

    print(f"\n  Dataset: Iris  |  {X.shape[0]} muestras  ×  {X.shape[1]} features")
    print(f"  Clases: {nombres_clases}")
    print(f"  Distribución: { {c: int(sum(y == i)) for i, c in enumerate(nombres_clases)} }")
    print(df.describe().to_string())

    visualizar_pairplot(
        df,
        hue_col         = "especie",
        titulo          = "Iris Dataset — Exploración de Features",
        nombre_archivo  = "iris_pairplot.png",
    )

    # ── 2. Preprocesamiento ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, scaler = preprocesar(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n  Entrenamiento: {X_train.shape[0]} muestras")
    print(f"  Prueba       : {X_test.shape[0]} muestras")
    print(f"  StandardScaler aplicado  (µ=0, σ=1 por feature)")

    # ── 3. Configuración del AG ────────────────────────────────────────────
    N_FEATURES  = X_train.shape[1]   # 4
    N_CLASES    = 3
    N_GENES     = N_FEATURES * N_CLASES + N_CLASES   # 15

    aptitud      = AptitudClasificacion(X_train, y_train, n_classes=N_CLASES)
    seleccion    = SeleccionRango()
    reproduccion = ReproduccionDosPuntos()
    mutacion     = MutacionGaussiana(prob_mutacion=0.20, sigma=0.25)

    ag = AlgoritmoGenetico(
        n_genes          = N_GENES,
        tam_poblacion    = 300,
        aptitud          = aptitud,
        seleccion        = seleccion,
        reproduccion     = reproduccion,
        mutacion         = mutacion,
        tasa_reemplazo   = 0.60,
        n_generaciones   = 400,
        init_range       = (-2.0, 2.0),
        nombre_problema  = "Clasificación Multiclase — Iris",
        archivo_mejor    = "iris_mejor_generacion.json",
        semilla          = 42,
    )

    # ── 4. Evolución ───────────────────────────────────────────────────────
    ag.ejecutar()
    ag.reporte_contraste()

    # ── 5. Evaluación en conjunto de prueba ───────────────────────────────
    mejor = ag.ultima_generacion.mejor_individuo()
    y_pred_train = aptitud.predecir(X_train, mejor)
    y_pred_test  = aptitud.predecir(X_test,  mejor)
    acc_train    = accuracy_score(y_train, y_pred_train)
    acc_test     = accuracy_score(y_test,  y_pred_test)

    print(f"\n  ── Evaluación del Mejor Individuo ──")
    print(f"  Accuracy entrenamiento : {acc_train:.4f}  ({acc_train * 100:.2f}%)")
    print(f"  Accuracy prueba        : {acc_test:.4f}  ({acc_test  * 100:.2f}%)")
    print(f"  Fitness final          : {mejor.fitness:.6f}")

    # ── 6. Visualizaciones ─────────────────────────────────────────────────
    ag.graficar(
        titulo_extra   = f"(N_genes={N_GENES} · Pop=300 · Gen=400)",
        nombre_archivo = "iris_evolucion.png",
    )

    visualizar_confusion(
        y_true          = y_test,
        y_pred          = y_pred_test,
        nombres_clases  = nombres_clases,
        titulo          = "Iris — Clasificación por Algoritmo Genético",
        nombre_archivo  = "iris_confusion_matrix.png",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  EJEMPLO — REGRESIÓN LINEAL MULTIVARIABLE (Diabetes)
# ══════════════════════════════════════════════════════════════════════════════

def main_regresion() -> None:
    """
    Ejemplo completo de regresión lineal con el dataset Diabetes (sklearn).

    Configuración del cromosoma (10 features):
      ┌───────────────────────────────────────────────────┐
      │  genes = [w1, w2, …, w10, b]  (11 genes)        │
      │  Modelo: ŷ = X · w + b                          │
      │  Fitness: 1 / (MSE + ε)                          │
      └───────────────────────────────────────────────────┘
    """
    print("\n" + "═" * 68)
    print("  EJEMPLO — REGRESIÓN DIABETES (AG + Codificación Real)")
    print("═" * 68)

    # ── 1. Cargar y explorar ───────────────────────────────────────────────
    df, X, y = cargar_diabetes()

    print(f"\n  Dataset: Diabetes  |  {X.shape[0]} muestras  ×  {X.shape[1]} features")
    print(f"  Variable objetivo: progresión de diabetes (continua)")
    print(df.describe().to_string())

    # Pairplot sobre las primeras 5 features para no saturar la figura
    df_vis = df[list(df.columns[:5]) + ["target"]].copy()
    visualizar_pairplot(
        df_vis,
        hue_col         = None,   # regresión: no hay hue categórico
        titulo          = "Diabetes Dataset — Primeras 5 Features",
        nombre_archivo  = "diabetes_pairplot.png",
    )

    # ── 2. Preprocesamiento ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, _ = preprocesar(
        X, y, test_size=0.20, random_state=42
    )
    print(f"\n  Entrenamiento: {X_train.shape[0]} muestras")
    print(f"  Prueba       : {X_test.shape[0]} muestras")

    # ── 3. Configuración del AG ────────────────────────────────────────────
    N_FEATURES = X_train.shape[1]   # 10
    N_GENES    = N_FEATURES + 1     # 11

    aptitud      = AptitudRegresion(X_train, y_train)
    seleccion    = SeleccionRango()
    reproduccion = ReproduccionDosPuntos()
    mutacion     = MutacionGaussiana(prob_mutacion=0.20, sigma=0.15)

    ag = AlgoritmoGenetico(
        n_genes          = N_GENES,
        tam_poblacion    = 300,
        aptitud          = aptitud,
        seleccion        = seleccion,
        reproduccion     = reproduccion,
        mutacion         = mutacion,
        tasa_reemplazo   = 0.60,
        n_generaciones   = 500,
        init_range       = (-2.0, 2.0),
        nombre_problema  = "Regresión Lineal — Diabetes",
        archivo_mejor    = "diabetes_mejor_generacion.json",
        semilla          = 42,
    )

    # ── 4. Evolución ───────────────────────────────────────────────────────
    ag.ejecutar()
    ag.reporte_contraste()

    # ── 5. Evaluación ─────────────────────────────────────────────────────
    mejor    = ag.ultima_generacion.mejor_individuo()
    y_pred   = aptitud.predecir(X_test, mejor)
    mse_test = aptitud.mse(X_test, y_test, mejor)
    rmse     = math.sqrt(mse_test)

    print(f"\n  ── Evaluación del Mejor Individuo ──")
    print(f"  MSE  prueba  : {mse_test:.4f}")
    print(f"  RMSE prueba  : {rmse:.4f}")
    print(f"  Fitness final: {mejor.fitness:.6f}")

    # ── 6. Visualizaciones ─────────────────────────────────────────────────
    ag.graficar(
        titulo_extra   = f"(N_genes={N_GENES} · Pop=300 · Gen=500)",
        nombre_archivo = "diabetes_evolucion.png",
    )
    visualizar_regresion(
        y_test         = y_test,
        y_pred         = y_pred,
        titulo         = "Diabetes — Regresión por Algoritmo Genético",
        nombre_archivo = "diabetes_regresion.png",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PUNTO DE ENTRADA
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    """
    Ejecuta el ejemplo de Clasificación Multiclase (Iris) por defecto.
    Para ejecutar regresión, llama a main_regresion() directamente.
    """
    main_clasificacion()


if __name__ == "__main__":
    main()
