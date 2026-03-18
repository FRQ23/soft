import random
import copy
from typing import List, Dict

from CromosomaBin import CromosomaBinario
from interfaz import (
    AptitudMochila,
    SeleccionRank, SeleccionRuleta, SeleccionTorneo,
    ReproduccionUnCruce,
    MutacionFlip,
)


class GA(object):
    def __init__(self, num_inv: int, num_genes: int, hyperparams: Dict):
        cromosoma_class = hyperparams["Cromosoma"] if "Cromosoma" in hyperparams else CromosomaBinario
        cromosoma_class_params = hyperparams["Cromosoma_params"].copy() \
            if "Cromosoma_params" in hyperparams else [num_genes]
        self.poblacion: List[CromosomaBinario] = [
            cromosoma_class(*cromosoma_class_params) for _ in range(num_inv)
        ]
        self.num_inv: int = num_inv
        self.hyperparams: Dict = hyperparams
        self.evaluar = hyperparams["Aptitud"]
        self.seleccion = hyperparams["Seleccion"]
        self.reproducion = hyperparams["Reproduccion"]
        self.mutacion = hyperparams["Mutacion"]
        self.num_remplazo = int(hyperparams["por_remplazo"] * num_inv)
        self.max_generaciones = hyperparams["generaciones"]
        self.objetivo = hyperparams["objetivo"]
        self.stop = hyperparams["stop"]
        self.mejor_generacion = None

        # Inicializar solo si se usa CromosomaBinario (mochila)
        if "Cromosoma" not in hyperparams:
            for i in range(num_inv):
                self.poblacion[i].llenar_random_data()

    def buscar_solucion(self, tolerancia: float, var: Dict):
        avg_tol: float = 99999999.99
        sum_score_prev: float = 0
        sum_score: float = 1
        generacion: int = 1
        mating_pool = None
        generacion_nueva: List = None
        fail: bool = False
        top_score = 999999 if self.objetivo == "minimizar" else -1
        top_gen_id = 0

        while not fail and generacion <= self.max_generaciones:
            # 1. Evaluar población
            self.evaluar.evaluar_poblacion(self.poblacion, var)

            # 1.2 Calcular avg_score
            sum_scores: float = sum(cromosoma.score for cromosoma in self.poblacion)
            sum_scores /= self.num_inv

            if self.stop == "relativo":
                avg_tol = abs(sum_scores - sum_score_prev) / sum_score
                sum_score_prev = sum_scores
            elif self.stop == "loss":
                avg_tol = sum_scores

            # Guardar mejor generación
            if self.objetivo == "minimizar":
                if sum_scores < top_score:
                    top_score = sum_scores
                    self.mejor_generacion = copy.deepcopy(self.poblacion)
                    top_gen_id = generacion
            else:
                if sum_scores > top_score:
                    top_score = sum_scores
                    self.mejor_generacion = copy.deepcopy(self.poblacion)
                    top_gen_id = generacion

            # Reporte cada 10% de generaciones
            if (generacion + 1) % max(1, int(self.max_generaciones * 0.1)) == 0:
                print(f"[{generacion + 1:05d}] avg_score: {sum_scores:.6f}  "
                      f"avg_tol: {avg_tol:.6f}")

            # Criterio de parada
            if (avg_tol <= tolerancia and self.objetivo == "minimizar") \
                    or (avg_tol == 1.0 and self.objetivo == "maximizar"):
                print(f"Terminó por tolerancia en gen {generacion}  "
                      f"(avg_tol={avg_tol:.6f})")
                break

            # 1.4 Ordenar mejor → peor
            mayormenor = True if self.objetivo == "maximizar" else False
            self.poblacion = sorted(self.poblacion, key=lambda c: c.score, reverse=mayormenor)

            # 2. Selección
            mating_pool = self.seleccion.seleccion_poblacion(self.poblacion)

            # 3. Nueva generación (elite + hijos)
            generacion_nueva = copy.deepcopy(self.poblacion[:self.num_remplazo])
            self.reproducion.reproduccion_poblacion(generacion_nueva, mating_pool, self.num_inv)

            # 4. Mutación
            self.mutacion.mutacion_poblacion(generacion_nueva)

            generacion += 1
            self.poblacion = copy.deepcopy(generacion_nueva)

        # Evaluación final
        self.evaluar.evaluar_poblacion(self.poblacion, var)
        print(f"[{generacion:05d}] avg_score: {sum_scores:.6f}  avg_tol: {avg_tol:.6f}")
        print(f"[Top generación: {top_gen_id}]  top_score: {top_score:.6f}")
