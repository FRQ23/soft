import numpy as np
from CromosomaBin import CromosomaBinRes
import seaborn as sns
import matplotlib.pyplot as plt

def softmax(z:np.ndarray)->np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z/np.sum(exp_z, axis=1,keepdims=True)


def regresion_cromosoma(c:CromosomaBinRes, X,ATRIBUTOS:int,CLASES:int)->np.ndarray:
    # Usa get_all_valores() si está disponible (vectorizado, ~7× más rápido).
    # Fallback al bucle original para compatibilidad con CromosomaBinario.
    if hasattr(c, "get_all_valores"):
        genes = c.get_all_valores()
        W = genes[:ATRIBUTOS * CLASES].reshape(ATRIBUTOS, CLASES)
        B = genes[ATRIBUTOS * CLASES:ATRIBUTOS * CLASES + CLASES].reshape(1, CLASES)
    else:
        W = np.array([c.get_gen_valor(i) for i in range(ATRIBUTOS * CLASES)]).reshape(ATRIBUTOS, CLASES)
        B = np.array([c.get_gen_valor(b) for b in range(ATRIBUTOS * CLASES, ATRIBUTOS * CLASES + CLASES)]).reshape(1, CLASES)
    return softmax(X @ W + B)


def entropia_cruzada(yh,Y): #loss function
    return -np.mean(np.sum(Y*np.log(yh.clip(1e-12)),axis=1))

def SSE(yh,y):
    return np.sum(yh-y)**2
def MSE(yh,y):
    return np.mean(np.sum((yh-y)**2,axis=1))/yh.shape[1]
def RMSE(yh,y):
    return np.sqrt(np.mean(np.sum((yh-y)**2,axis=1))/yh.shape[1])

def accuracy(yh,Y): # presición
    return np.mean(np.argmax(yh,axis=1)==np.argmax(Y,axis=1))


def onehot_encode(Y,num_clases:int):
    N= len(Y)
    ye= np.zeros((N,num_clases))
    ye[np.arange(N),Y]=1
    return ye

def matriz_confusion(y,yh,CLASES:int):
    m = np.zeros((CLASES,CLASES),dtype=np.int8)
    yc = np.argmax(y,axis=1)
    yhc = np.argmax(yh,axis=1)
    for t,p in zip(yc,yhc):
        m[t][p] +=1

    return m

def display_matriz_confusion(matriz:np.ndarray):
    sns.heatmap(matriz,annot=True,fmt='d',cmap='Blues')
    plt.xlabel("Modelo")
    plt.ylabel("Real")
    plt.show()
