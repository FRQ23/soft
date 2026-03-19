from abc import ABC, abstractmethod
from typing import List, Dict

from CromosomaBin import CromosomaBinario
from CromosomaBin import CromosomaBinRes
from interfaz.regresion_classi import regresion_cromosoma, entropia_cruzada, accuracy, onehot_encode
import numpy as np
from joblib import Parallel, delayed


def _eval_mnist_uno(cromosoma, X_tr, Y_tr, caracteristicas, clases, objetivo):
    YH = regresion_cromosoma(cromosoma, X_tr, caracteristicas, clases)
    loss = entropia_cruzada(YH, Y_tr)
    acc = accuracy(YH, Y_tr)
    return loss if objetivo == "minimizar" else acc


class Aptitud(ABC):
    @abstractmethod
    def evaluar_poblacion(self,poblacion:List[CromosomaBinario],var:Dict):
        """interfaz encargada para darle un score a la población"""
        pass

class AptitudMochila(Aptitud):
    def evaluar_poblacion(self,poblacion:List[CromosomaBinario],var:Dict):
        for cromosoma in poblacion:
            peso_total:int = 0
            valor_total:int =0
            pesos:List[int] = var["pesos"]
            valores:List[int] = var["valores"]
            capacidad_maxima:int = var["capacidad"]
            factor_penalizacion:int = 70 #%

            #recorrer los genes
            for i in range(1,cromosoma.n_genes+1):
                if cromosoma.get_bits(i) == 1:
                    peso_total += pesos[i-1]
                    valor_total+= valores[i-1]

            #asignar el score
            if peso_total > capacidad_maxima:
                penalizacion = (peso_total - capacidad_maxima) * factor_penalizacion
                score = valor_total- penalizacion
                cromosoma.score = max(0, score)
            else:
                cromosoma.score =valor_total


class AptitudMNIST(Aptitud):
    def __init__(self, pixeles:int, clases:int,objetivo:str="minimizar"):
        self.caracteristicas = pixeles
        self.clases = clases
        self.objetivo = objetivo

    def evaluar_poblacion(self, poblacion:List[CromosomaBinRes], var:Dict):
        X_tr, Y_tr = var["X_TRAIN"], var["Y_TRAIN"]
        scores = Parallel(n_jobs=-1, prefer="threads")(
            delayed(_eval_mnist_uno)(c, X_tr, Y_tr, self.caracteristicas, self.clases, self.objetivo)
            for c in poblacion
        )
        for c, s in zip(poblacion, scores):
            c.score = s
