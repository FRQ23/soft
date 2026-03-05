"""
╔══════════════════════════════════════════════════════════════════════════════╗
║       ADVANCED GENETIC ALGORITHM — KNAPSACK PROBLEM                        ║
║       Modular implementation based on AGmix.pdf / GAImp.pdf interfaces      ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture:
  • Cromosoma       — flexible bit-list chromosome (scales to any problem size)
  • Aptitud         — abstract fitness interface  → AptitudMochila (with penalty)
  • Seleccion       — abstract selection interface → SeleccionRango / SeleccionRuleta
  • Reproduccion    — abstract crossover interface → ReproduccionDosPuntos
  • Mutacion        — abstract mutation interface  → MutacionBitFlip
  • AlgoritmoGenetico — GA engine with persistence, contrast report, and plotting
"""

import abc
import json
import random
import datetime
import math
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator


# ══════════════════════════════════════════════════════════════════════════════
#  CHROMOSOME
# ══════════════════════════════════════════════════════════════════════════════

class Cromosoma:
    """
    Flexible bit-list chromosome.

    Uses a plain Python list of 0/1 integers so that the algorithm scales to
    *any* number of items without being limited to 32-bit integers.
    """

    def __init__(self, n_genes: int, bits: Optional[List[int]] = None):
        self.n_genes: int = n_genes
        if bits is not None:
            if len(bits) != n_genes:
                raise ValueError(
                    f"bits length {len(bits)} != n_genes {n_genes}"
                )
            self.bits: List[int] = bits[:]
        else:
            self.bits = [random.randint(0, 1) for _ in range(n_genes)]
        self.fitness: float = 0.0

    # ── Utilities ──────────────────────────────────────────────────────────

    def clone(self) -> "Cromosoma":
        """Return a deep copy preserving fitness."""
        c = Cromosoma(self.n_genes, self.bits)
        c.fitness = self.fitness
        return c

    def to_dict(self) -> Dict[str, Any]:
        return {"bits": self.bits, "fitness": self.fitness}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Cromosoma":
        c = cls(len(d["bits"]), d["bits"])
        c.fitness = d["fitness"]
        return c

    def __repr__(self) -> str:
        bit_str = "".join(map(str, self.bits))
        return f"Cromosoma(fitness={self.fitness:.4f}, bits={bit_str})"


# ══════════════════════════════════════════════════════════════════════════════
#  ABSTRACT INTERFACES  (following AGmix.pdf / GAImp.pdf modular design)
# ══════════════════════════════════════════════════════════════════════════════

class Aptitud(abc.ABC):
    """Abstract fitness evaluator."""

    @abc.abstractmethod
    def evaluar(self, cromosoma: Cromosoma) -> float:
        """Return the scalar fitness of a single chromosome."""

    def evaluar_poblacion(self, poblacion: List[Cromosoma]) -> None:
        """Evaluate and assign fitness to every chromosome in the population."""
        for c in poblacion:
            c.fitness = self.evaluar(c)


class Seleccion(abc.ABC):
    """Abstract parent-selection operator."""

    @abc.abstractmethod
    def seleccionar(self, poblacion: List[Cromosoma], n: int) -> List[Cromosoma]:
        """Return *n* selected (cloned) chromosomes from the population."""


class Reproduccion(abc.ABC):
    """Abstract crossover / recombination operator."""

    @abc.abstractmethod
    def cruzar(
        self, padre: Cromosoma, madre: Cromosoma
    ) -> Tuple[Cromosoma, Cromosoma]:
        """Produce two offspring from two parents."""


class Mutacion(abc.ABC):
    """Abstract mutation operator."""

    @abc.abstractmethod
    def mutar(self, cromosoma: Cromosoma) -> Cromosoma:
        """Return a (possibly mutated) copy of the chromosome."""


# ══════════════════════════════════════════════════════════════════════════════
#  FITNESS  —  Knapsack with Penalty
# ══════════════════════════════════════════════════════════════════════════════

class AptitudMochila(Aptitud):
    """
    Fitness function for the 0/1 Knapsack problem with soft penalty.

        score  = max(0,  total_value  −  penalty)
        penalty = (total_weight − capacity) × penalty_factor   [only if overweight]

    Feasible solutions are rewarded by their full value; infeasible solutions
    receive a proportional deduction scaled by how much they exceed capacity.
    """

    def __init__(
        self,
        weights: List[float],
        values: List[float],
        capacity: float,
        penalty_factor: float = 10.0,
    ):
        if len(weights) != len(values):
            raise ValueError("weights and values must have the same length.")
        self.weights = weights
        self.values = values
        self.capacity = capacity
        self.penalty_factor = penalty_factor

    def evaluar(self, cromosoma: Cromosoma) -> float:
        total_weight = sum(w * b for w, b in zip(self.weights, cromosoma.bits))
        total_value  = sum(v * b for v, b in zip(self.values,  cromosoma.bits))
        if total_weight > self.capacity:
            penalty = (total_weight - self.capacity) * self.penalty_factor
            return max(0.0, total_value - penalty)
        return float(total_value)

    def peso_total(self, cromosoma: Cromosoma) -> float:
        return sum(w * b for w, b in zip(self.weights, cromosoma.bits))

    def valor_total(self, cromosoma: Cromosoma) -> float:
        return sum(v * b for v, b in zip(self.values, cromosoma.bits))


# ══════════════════════════════════════════════════════════════════════════════
#  SELECTION  —  Range (Rank-Based) Selection
# ══════════════════════════════════════════════════════════════════════════════

class SeleccionRango(Seleccion):
    """
    Rank-based (Range) Selection.

    Steps:
      1. Sort population from worst (rank 1) to best (rank N).
      2. Compute:
            Solution_Percentage = rank / Σranks
            Expected_Count      = top_rank × Solution_Percentage
      3. Use the resulting probabilities as weights for random.choices.

    This avoids premature convergence caused by super-individuals dominating
    pure fitness-proportionate selection.
    """

    def seleccionar(self, poblacion: List[Cromosoma], n: int) -> List[Cromosoma]:
        sorted_pop = sorted(poblacion, key=lambda c: c.fitness)   # worst → best
        N         = len(sorted_pop)
        ranks     = list(range(1, N + 1))                          # 1 … N
        sum_ranks = sum(ranks)                                     # N(N+1)/2
        top_rank  = ranks[-1]                                      # N

        # Solution_Percentage  per individual
        sol_pct = [r / sum_ranks for r in ranks]
        # Expected_Count (used as probability weight)
        exp_cnt = [top_rank * sp for sp in sol_pct]

        selected = random.choices(sorted_pop, weights=exp_cnt, k=n)
        return [c.clone() for c in selected]


# ══════════════════════════════════════════════════════════════════════════════
#  SELECTION  —  Stochastic Roulette (Fitness-Proportionate)
# ══════════════════════════════════════════════════════════════════════════════

class SeleccionRuleta(Seleccion):
    """
    Stochastic Roulette-Wheel Selection.

    Each individual's probability of being chosen equals:
        P(i) = fitness(i) / Σ fitness

    When all fitnesses are zero (degenerate case) a uniform distribution
    is used as a fallback.
    """

    def seleccionar(self, poblacion: List[Cromosoma], n: int) -> List[Cromosoma]:
        total = sum(c.fitness for c in poblacion)
        if total == 0.0:
            return [random.choice(poblacion).clone() for _ in range(n)]
        weights  = [c.fitness / total for c in poblacion]
        selected = random.choices(poblacion, weights=weights, k=n)
        return [c.clone() for c in selected]


# ══════════════════════════════════════════════════════════════════════════════
#  CROSSOVER  —  Two-Point Crossover
# ══════════════════════════════════════════════════════════════════════════════

class ReproduccionDosPuntos(Reproduccion):
    """
    Two-Point Crossover.

    Two distinct cut points p1 < p2 are chosen uniformly at random from the
    interior of the chromosome.  The middle segment is swapped between parents
    to produce two offspring:

        h1 = padre[:p1] + madre[p1:p2] + padre[p2:]
        h2 = madre[:p1] + padre[p1:p2] + madre[p2:]
    """

    def cruzar(
        self, padre: Cromosoma, madre: Cromosoma
    ) -> Tuple[Cromosoma, Cromosoma]:
        n = padre.n_genes
        p1, p2 = sorted(random.sample(range(1, n), 2))

        bits_h1 = padre.bits[:p1] + madre.bits[p1:p2] + padre.bits[p2:]
        bits_h2 = madre.bits[:p1] + padre.bits[p1:p2] + madre.bits[p2:]

        return Cromosoma(n, bits_h1), Cromosoma(n, bits_h2)


# ══════════════════════════════════════════════════════════════════════════════
#  MUTATION  —  Bit-Flip Mutation
# ══════════════════════════════════════════════════════════════════════════════

class MutacionBitFlip(Mutacion):
    """
    Bit-Flip Mutation.

    Each gene is independently flipped (0→1 or 1→0) with probability
    *prob_mutacion*.  A low probability (≈ 1/n_genes) prevents disrupting
    well-adapted schemata while still maintaining genetic diversity.
    """

    def __init__(self, prob_mutacion: float):
        if not 0.0 <= prob_mutacion <= 1.0:
            raise ValueError("prob_mutacion must be in [0, 1].")
        self.prob_mutacion = prob_mutacion

    def mutar(self, cromosoma: Cromosoma) -> Cromosoma:
        nuevo = cromosoma.clone()
        for i in range(nuevo.n_genes):
            if random.random() < self.prob_mutacion:
                nuevo.bits[i] = 1 - nuevo.bits[i]
        return nuevo


# ══════════════════════════════════════════════════════════════════════════════
#  GENERATIONAL DATA  (persistence model)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DatosGeneracion:
    """Snapshot of one generation for persistence and contrast analysis."""

    numero:      int
    individuos:  List[Dict[str, Any]]
    max_fitness: float
    avg_fitness: float
    min_fitness: float

    # ── Serialisation ──────────────────────────────────────────────────────

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

    # ── Helpers ────────────────────────────────────────────────────────────

    def mejor_individuo(self) -> Cromosoma:
        best = max(self.individuos, key=lambda x: x["fitness"])
        return Cromosoma.from_dict(best)

    def todos_los_individuos(self) -> List[Cromosoma]:
        return [Cromosoma.from_dict(ind) for ind in self.individuos]


# ══════════════════════════════════════════════════════════════════════════════
#  GENETIC ALGORITHM ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class AlgoritmoGenetico:
    """
    Main GA engine.

    Responsibilities
    ─────────────────
    • Initialise a random population.
    • Run the evolutionary loop for *n_generaciones* generations.
    • Track and persist the Best Generation (highest avg fitness) to JSON.
    • Expose a generational contrast report (console + visual).
    • Produce a vibrant Matplotlib dashboard saved to PNG.
    """

    def __init__(
        self,
        n_genes:         int,
        tam_poblacion:   int,
        aptitud:         Aptitud,
        seleccion:       Seleccion,
        reproduccion:    Reproduccion,
        mutacion:        Mutacion,
        tasa_reemplazo:  float = 0.50,
        n_generaciones:  int   = 200,
        archivo_mejor:   str   = "mejor_generacion.json",
        semilla:         Optional[int] = None,
    ):
        if semilla is not None:
            random.seed(semilla)
            np.random.seed(semilla)

        self.n_genes        = n_genes
        self.tam_poblacion  = tam_poblacion
        self.aptitud        = aptitud
        self.seleccion      = seleccion
        self.reproduccion   = reproduccion
        self.mutacion       = mutacion
        self.tasa_reemplazo = tasa_reemplazo
        self.n_generaciones = n_generaciones
        self.archivo_mejor  = archivo_mejor

        self.poblacion: List[Cromosoma] = []

        # Per-generation history
        self.historial_avg: List[float] = []
        self.historial_max: List[float] = []
        self.historial_min: List[float] = []

        # Snapshots
        self.mejor_generacion: Optional[DatosGeneracion] = None
        self.ultima_generacion: Optional[DatosGeneracion] = None

    # ── Private helpers ────────────────────────────────────────────────────

    def _inicializar_poblacion(self) -> None:
        self.poblacion = [Cromosoma(self.n_genes) for _ in range(self.tam_poblacion)]

    def _estadisticas(self, poblacion: List[Cromosoma]) -> Tuple[float, float, float]:
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
        """Produce offspring via selection → crossover → mutation → replacement."""
        n_hijos = int(self.tam_poblacion * self.tasa_reemplazo)
        if n_hijos % 2 != 0:
            n_hijos += 1          # keep pairs

        hijos: List[Cromosoma] = []
        while len(hijos) < n_hijos:
            padres = self.seleccion.seleccionar(self.poblacion, 2)
            h1, h2 = self.reproduccion.cruzar(padres[0], padres[1])
            h1 = self.mutacion.mutar(h1)
            h2 = self.mutacion.mutar(h2)
            self.aptitud.evaluar_poblacion([h1, h2])
            hijos.extend([h1, h2])

        # Elitist replacement: remove the *n_hijos* worst individuals
        self.poblacion.sort(key=lambda c: c.fitness)
        self.poblacion[:n_hijos] = hijos[:n_hijos]

    # ── Public API ─────────────────────────────────────────────────────────

    def ejecutar(self) -> None:
        """Run the full evolutionary process."""
        self._banner()
        self._inicializar_poblacion()
        self.aptitud.evaluar_poblacion(self.poblacion)

        for gen_num in range(1, self.n_generaciones + 1):
            datos = self._capturar_generacion(gen_num)

            self.historial_max.append(datos.max_fitness)
            self.historial_avg.append(datos.avg_fitness)
            self.historial_min.append(datos.min_fitness)

            # Best Generation = generation with highest average fitness
            if (
                self.mejor_generacion is None
                or datos.avg_fitness > self.mejor_generacion.avg_fitness
            ):
                self.mejor_generacion = datos
                self._guardar_mejor_generacion(datos)    # persist to disk

            if gen_num % 20 == 0 or gen_num == 1:
                print(
                    f"  Gen {gen_num:>4} │ "
                    f"Max: {datos.max_fitness:>8.2f}  "
                    f"Avg: {datos.avg_fitness:>8.2f}  "
                    f"Min: {datos.min_fitness:>8.2f}"
                )

            self._paso_evolutivo()

        # Evaluate and capture the state AFTER the last evolutionary step
        self.aptitud.evaluar_poblacion(self.poblacion)
        self.ultima_generacion = self._capturar_generacion(self.n_generaciones)

        print("\n  Evolución completada.")
        print(f"  ✔  Mejor generación (#{self.mejor_generacion.numero}) "
              f"persistida en: '{self.archivo_mejor}'")

    # ── Analysis helpers ───────────────────────────────────────────────────

    @staticmethod
    def similitud_bits(c1: Cromosoma, c2: Cromosoma) -> float:
        """Return the percentage of identical genes between two chromosomes."""
        same = sum(1 for a, b in zip(c1.bits, c2.bits) if a == b)
        return same / c1.n_genes * 100.0

    @staticmethod
    def diversidad_poblacion(individuos: List[Cromosoma]) -> float:
        """
        Average pairwise Hamming distance (normalised to [0, 1]).
        Higher → more diverse gene pool.
        """
        n = len(individuos)
        if n < 2:
            return 0.0
        total_diff = 0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = sum(a != b for a, b in zip(individuos[i].bits, individuos[j].bits))
                total_diff += diff
                count += 1
        return total_diff / (count * individuos[0].n_genes)

    # ── Console Report ─────────────────────────────────────────────────────

    def _banner(self) -> None:
        sel_name = type(self.seleccion).__name__
        rep_name = type(self.reproduccion).__name__
        mut_name = type(self.mutacion).__name__
        print("\n" + "═" * 64)
        print("  ALGORITMO GENÉTICO  —  PROBLEMA DE LA MOCHILA")
        print("═" * 64)
        print(f"  Genes              : {self.n_genes}")
        print(f"  Tamaño de población: {self.tam_poblacion}")
        print(f"  Generaciones       : {self.n_generaciones}")
        print(f"  Tasa de reemplazo  : {self.tasa_reemplazo * 100:.0f}%")
        print(f"  Selección          : {sel_name}")
        print(f"  Reproducción       : {rep_name}")
        print(f"  Mutación           : {mut_name}")
        print("═" * 64)

    def reporte_contraste(self) -> None:
        """
        Print a detailed side-by-side contrast between the Best Generation
        and the Final Generation, including bit-level genetic similarity.
        """
        if self.mejor_generacion is None or self.ultima_generacion is None:
            print("  [!] Execute 'ejecutar()' before requesting the contrast report.")
            return

        mg = self.mejor_generacion
        ug = self.ultima_generacion
        bi_mg = mg.mejor_individuo()
        bi_ug = ug.mejor_individuo()
        sim   = self.similitud_bits(bi_mg, bi_ug)

        # Convergence indicator: how much the population improved from best gen to end
        delta_avg = ug.avg_fitness - mg.avg_fitness
        delta_max = ug.max_fitness - mg.max_fitness

        print("\n" + "═" * 64)
        print("  CONTRASTE  —  MEJOR GENERACIÓN  vs  GENERACIÓN FINAL")
        print("═" * 64)
        print(f"  {'Métrica':<28} {'Mejor Gen':>12} {'Gen Final':>12}")
        print(f"  {'─' * 28} {'─' * 12} {'─' * 12}")
        print(f"  {'Número de generación':<28} {mg.numero:>12d} {ug.numero:>12d}")
        print(f"  {'Fitness Máximo':<28} {mg.max_fitness:>12.4f} {ug.max_fitness:>12.4f}")
        print(f"  {'Fitness Promedio':<28} {mg.avg_fitness:>12.4f} {ug.avg_fitness:>12.4f}")
        print(f"  {'Fitness Mínimo':<28} {mg.min_fitness:>12.4f} {ug.min_fitness:>12.4f}")

        print(f"\n  {'── Δ desde mejor gen ──':^52}")
        print(f"  {'Δ Fitness Promedio':<28} {delta_avg:>+12.4f}")
        print(f"  {'Δ Fitness Máximo':<28} {delta_max:>+12.4f}")

        print(f"\n  {'── Similitud genética (bit-a-bit) ──':^52}")
        print(f"  Entre mejores individuos: {sim:.2f}% de genes idénticos")
        bar_len = 40
        filled  = round(sim / 100 * bar_len)
        bar     = "█" * filled + "░" * (bar_len - filled)
        print(f"  [{bar}]  {sim:.1f}%")

        print(f"\n  Mejor Gen  #{mg.numero:>4}  bits: {''.join(map(str, bi_mg.bits))}")
        print(f"  Gen Final  #{ug.numero:>4}  bits: {''.join(map(str, bi_ug.bits))}")
        print("═" * 64)

    def reporte_mejor_individuo(self, aptitud_mochila: "AptitudMochila") -> None:
        """Print the best individual's knapsack statistics."""
        if self.ultima_generacion is None:
            return
        bi = self.ultima_generacion.mejor_individuo()
        items_sel = [i for i, b in enumerate(bi.bits) if b == 1]
        print(f"\n  ── Mejor individuo (Generación Final) ──")
        print(f"  Fitness  : {bi.fitness:.4f}")
        print(f"  Peso     : {aptitud_mochila.peso_total(bi):.1f} kg "
              f"/ {aptitud_mochila.capacity} kg")
        print(f"  Valor    : ${aptitud_mochila.valor_total(bi):.2f}")
        print(f"  Items seleccionados ({len(items_sel)}): {items_sel}")

    # ── Visual Report ──────────────────────────────────────────────────────

    def graficar(
        self,
        titulo_extra: str = "",
        guardar: bool = True,
    ) -> None:
        """
        High-impact Matplotlib dashboard with vibrant neon-on-dark styling.

        Plot 1  —  Average / Max / Min Fitness vs Generations
        Plot 2  —  Relative Error vs Generations
        """
        gens    = list(range(1, len(self.historial_avg) + 1))
        ref_max = max(self.historial_max)          # best known solution quality

        # Relative error: how far is the average from the global best found
        err_rel = [
            abs(ref_max - v) / ref_max * 100 if ref_max > 0 else 0.0
            for v in self.historial_avg
        ]

        best_gen_x = self.mejor_generacion.numero if self.mejor_generacion else 1

        # ── Style ──────────────────────────────────────────────────────────
        BG_DARK    = "#0a0a12"
        BG_PANEL   = "#12122a"
        C_AVG      = "#00e5ff"    # cyan
        C_MAX      = "#ff6b35"    # orange
        C_MIN      = "#b2ff59"    # lime
        C_ERR      = "#ff4081"    # hot-pink
        C_BEST     = "#ffd740"    # amber
        C_GRID     = "#1e1e3a"

        plt.rcParams.update({
            "font.family":       "monospace",
            "axes.facecolor":    BG_PANEL,
            "figure.facecolor":  BG_DARK,
            "text.color":        "white",
            "axes.labelcolor":   "#cccccc",
            "xtick.color":       "#888888",
            "ytick.color":       "#888888",
            "axes.edgecolor":    "#333355",
            "axes.grid":         True,
            "grid.color":        C_GRID,
            "grid.linewidth":    0.5,
        })

        fig = plt.figure(figsize=(17, 7.5), facecolor=BG_DARK)
        gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.30)

        # ── Plot 1: Fitness vs Generations ────────────────────────────────
        ax1 = fig.add_subplot(gs[0, 0])

        ax1.fill_between(
            gens, self.historial_min, self.historial_max,
            alpha=0.12, color=C_AVG, label="Rango [Min – Max]",
        )
        ax1.plot(gens, self.historial_max, color=C_MAX, lw=1.4,
                 ls="--", alpha=0.75, label="Fitness Máximo")
        ax1.plot(gens, self.historial_min, color=C_MIN, lw=1.0,
                 ls=":",  alpha=0.55, label="Fitness Mínimo")
        ax1.plot(gens, self.historial_avg, color=C_AVG, lw=2.8,
                 label="Fitness Promedio")

        # Best generation marker
        ax1.axvline(best_gen_x, color=C_BEST, lw=1.8, ls="--", alpha=0.9,
                    label=f"Mejor Gen (#{best_gen_x})")
        ax1.scatter(
            [best_gen_x], [self.historial_avg[best_gen_x - 1]],
            color=C_BEST, s=120, zorder=6,
        )

        ax1.set_title("Fitness por Generación", color="white",
                      fontsize=14, fontweight="bold", pad=14)
        ax1.set_xlabel("Generación", fontsize=11)
        ax1.set_ylabel("Fitness Score", fontsize=11)
        ax1.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=8))
        ax1.legend(facecolor="#0d0d1e", labelcolor="white",
                   fontsize=8.5, framealpha=0.85, loc="lower right")

        # Annotation: global best
        ax1.annotate(
            f" Global Max\n {ref_max:.0f}",
            xy=(gens[self.historial_max.index(ref_max)], ref_max),
            xytext=(20, -30), textcoords="offset points",
            color=C_MAX, fontsize=8,
            arrowprops=dict(arrowstyle="->", color=C_MAX, lw=1.2),
        )

        # ── Plot 2: Relative Error vs Generations ─────────────────────────
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

        # ── Super-title ────────────────────────────────────────────────────
        fig.suptitle(
            f"Algoritmo Genético  —  Problema de la Mochila  {titulo_extra}",
            color="white", fontsize=15, fontweight="bold", y=1.02,
        )

        plt.tight_layout()

        if guardar:
            ts    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"ga_reporte_{ts}.png"
            plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=BG_DARK)
            print(f"  ✔  Gráfica guardada en: '{fname}'")

        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
#  PROBLEM INSTANCE  —  50-item Knapsack
# ══════════════════════════════════════════════════════════════════════════════

def generar_instancia(
    n_items:  int = 50,
    seed:     int = 42,
) -> Tuple[List[int], List[int]]:
    """
    Generate a reproducible Knapsack instance.

    Returns:
        weights  — list of integers in [1, 10]
        values   — list of integers in [10, 100]
    """
    rng = random.Random(seed)
    weights = [rng.randint(1,  10)  for _ in range(n_items)]
    values  = [rng.randint(10, 100) for _ in range(n_items)]
    return weights, values


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Problem parameters ─────────────────────────────────────────────────
    N_ITEMS          = 50
    CAPACITY         = 150      # kg
    POP_SIZE         = 100
    N_GEN            = 200
    MUTATION_PROB    = 0.05     # 5 %
    REPLACEMENT_RATE = 0.50     # 50 %
    PENALTY_FACTOR   = 10.0
    SEED             = 42

    # ── Problem instance ───────────────────────────────────────────────────
    weights, values = generar_instancia(N_ITEMS, seed=SEED)

    print("\n  Instancia del problema")
    print("  " + "─" * 40)
    print(f"  Items             : {N_ITEMS}")
    print(f"  Capacidad         : {CAPACITY} kg")
    print(f"  Peso acumulado    : {sum(weights)} kg  (todos los ítems)")
    print(f"  Valor acumulado   : ${sum(values)}  (todos los ítems)")
    print(f"  Densidad media    : {sum(values)/sum(weights):.2f} $/kg")

    # ── Build components ───────────────────────────────────────────────────
    aptitud      = AptitudMochila(weights, values, CAPACITY, PENALTY_FACTOR)
    seleccion    = SeleccionRango()          # swap with SeleccionRuleta() freely
    reproduccion = ReproduccionDosPuntos()
    mutacion     = MutacionBitFlip(MUTATION_PROB)

    ag = AlgoritmoGenetico(
        n_genes         = N_ITEMS,
        tam_poblacion   = POP_SIZE,
        aptitud         = aptitud,
        seleccion       = seleccion,
        reproduccion    = reproduccion,
        mutacion        = mutacion,
        tasa_reemplazo  = REPLACEMENT_RATE,
        n_generaciones  = N_GEN,
        archivo_mejor   = "mejor_generacion.json",
        semilla         = SEED,
    )

    # ── Run ────────────────────────────────────────────────────────────────
    ag.ejecutar()

    # ── Reports ────────────────────────────────────────────────────────────
    ag.reporte_contraste()
    ag.reporte_mejor_individuo(aptitud)

    # ── Visual dashboard ───────────────────────────────────────────────────
    ag.graficar(titulo_extra=f"(N={N_ITEMS} ítems · Cap={CAPACITY} kg · Pop={POP_SIZE})")


if __name__ == "__main__":
    main()
