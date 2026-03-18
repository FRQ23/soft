from abc import ABC, abstractmethod
from typing import List

from CromosomaBin import CromosomaBinario, CromosomaBinRes
import random


class Reproduccion(ABC):
    @abstractmethod
    def reproduccion_poblacion(
        self,
        poblacion: List[CromosomaBinario],
        potencial: List[CromosomaBinario],
        max_poblacion: int,
    ) -> None:
        pass


class ReproduccionUnCruce(Reproduccion):
    def __init__(self, num_genes):
        super().__init__()
        self.num_genes = num_genes

    def reproduccion_poblacion(
        self,
        poblacion: List[CromosomaBinario],
        potencial: List[CromosomaBinario],
        max_poblacion: int,
    ) -> None:
        curr_po = len(poblacion)
        ord_padres = sorted(potencial, key=lambda c: c.score, reverse=True)
        id_padres: int = 0

        while curr_po < max_poblacion:
            hijoA = CromosomaBinario(self.num_genes)
            hijoB = CromosomaBinario(self.num_genes)
            punto_cruce = random.randint(1, self.num_genes - 1)
            padreA = ord_padres[id_padres % max_poblacion]
            padreB = ord_padres[(id_padres + 1) % max_poblacion]

            for i in range(1, self.num_genes):
                if i <= punto_cruce:
                    hijoA.set_bits(padreA.get_bits(i), i)
                    hijoB.set_bits(padreB.get_bits(i), i)
                else:
                    hijoA.set_bits(padreB.get_bits(i), i)
                    hijoB.set_bits(padreA.get_bits(i), i)
            if curr_po < max_poblacion:
                poblacion.append(hijoA)
                curr_po += 1
            if curr_po < max_poblacion:
                poblacion.append(hijoB)
                curr_po += 1
            id_padres += 1


class ReproduccionUnCruceRes(Reproduccion):
    def __init__(self, objetivo):
        self.objetivo = objetivo

    def reproduccion_poblacion(
        self,
        poblacion: List[CromosomaBinRes],
        potencial: List[CromosomaBinRes],
        max_poblacion: int,
    ) -> None:
        mayormenor = True if self.objetivo == "maximizar" else False
        orden_padres = sorted(potencial, key=lambda c: c.score, reverse=mayormenor)
        curr_po = len(poblacion)

        ngenes = poblacion[0].num_genes
        nbits  = poblacion[0].num_bits
        minv   = poblacion[0].min_valor_dec
        maxv   = poblacion[0].max_valor_dec
        id_padres: int = 0

        while curr_po < max_poblacion:
            hijoA = CromosomaBinRes(ngenes, nbits, minv, maxv)
            hijoB = CromosomaBinRes(ngenes, nbits, minv, maxv)

            punto_cruce = random.randint(1, ngenes - 1)
            padreA = orden_padres[id_padres % max_poblacion]
            padreB = orden_padres[(id_padres + 1) % max_poblacion]

            hijoA.data = padreA.data[:punto_cruce] + padreB.data[punto_cruce:]
            hijoB.data = padreB.data[:punto_cruce] + padreA.data[punto_cruce:]

            if curr_po < max_poblacion:
                poblacion.append(hijoA)
                curr_po += 1
            if curr_po < max_poblacion:
                poblacion.append(hijoB)
                curr_po += 1
            id_padres += 1


class ReproduccioDosCruce(Reproduccion):
    def reproduccion_poblacion(
        self,
        poblacion: List[CromosomaBinario],
        potencial: List[CromosomaBinario],
        max_poblacion: int,
    ) -> None:
        return
