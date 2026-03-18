import random
from abc import ABC, abstractmethod
from typing import List

from CromosomaBin import CromosomaBinario


class Seleccion(ABC):
    @abstractmethod
    def seleccion_poblacion(self, poblacion: List[CromosomaBinario]) -> List[CromosomaBinario]:
        pass


class SeleccionTorneo(Seleccion):
    def __init__(self, participantes=3, objetivo="minimizar") -> None:
        super().__init__()
        self.participantes = participantes
        self.objetivo = objetivo

    def seleccion_poblacion(self, poblacion: List[CromosomaBinario]) -> List[CromosomaBinario]:
        ganadores = []
        for i in range(len(poblacion)):
            competidores = random.sample(poblacion, self.participantes)
            ganador = max(competidores, key=lambda cromosoma: cromosoma.score) \
                if self.objetivo == "maximizar" else \
                min(competidores, key=lambda c: c.score)
            ganadores.append(ganador)
        return ganadores


class SeleccionRank(Seleccion):
    def __init__(self, objetivo):
        self.objetivo = objetivo

    def seleccion_poblacion(self, poblacion: List[CromosomaBinario]) -> List[CromosomaBinario]:
        mayormenor = True if self.objetivo == "maximizar" else False
        lista_ordenada = sorted(poblacion, key=lambda cromosoma: cromosoma.score, reverse=mayormenor)
        lista_score = [c.score for c in lista_ordenada]
        score_unicos = sorted(set(lista_score), reverse=mayormenor)
        rangos = {}
        for r, v in enumerate(score_unicos):
            rangos[v] = r + 1
        rn = len(rangos.items())
        sum_rank = rn * (rn + 1) / 2
        top_rank = rangos[lista_ordenada[-1].score]

        n = len(poblacion)
        seleccion = []
        c = 0
        min_prob = rangos[lista_ordenada[0].score] / sum_rank
        max_prob = top_rank / sum_rank
        while c < n:
            prob_rank_ok = random.uniform(min_prob, max_prob)
            for cromosoma in lista_ordenada:
                rank = rangos[cromosoma.score]
                por = rank / sum_rank
                cuenta = round(top_rank * por)
                if por >= prob_rank_ok and cuenta > 0 and c < n:
                    seleccion.append(cromosoma)
                    c += 1
                    break
        return seleccion


class SeleccionRuleta(Seleccion):
    def seleccion_poblacion(self, poblacion: List[CromosomaBinario]) -> List[CromosomaBinario]:
        sum_fit = sum([c.score for c in poblacion])
        n = len(poblacion)
        avg_fit = sum_fit / n

        seleccion = []
        c = 0
        min_fit = avg_fit / 2
        max_fit = avg_fit * 1.5
        while c < n:
            fit_ok = random.uniform(min_fit, max_fit)
            for crom in poblacion:
                if crom.score >= fit_ok and c < n:
                    seleccion.append(crom)
                    c += 1
        return seleccion
