import random


class CromosomaBinario(object):
    def __init__(self, genes: int):
        assert genes in [1, 2, 4, 8, 16, 32], \
            "la información se guarda en 32bits por lo que la cantidad de genes que se puede " \
            "guardar está límitado a los tamaños de un entero"
        self.n_genes: int = genes
        self.score: int = -1
        self.gen_width: int = 32 // genes  # en bits
        self.mascaras = {}
        self.bits: int = 0x00000000
        max_value: int = 2 ** self.gen_width - 1
        for g, i in enumerate(reversed(range(0, 32, self.gen_width))):
            self.mascaras["p_" + str(g + 1)] = i
            self.mascaras["m_" + str(g + 1)] = max_value << i

    def llenar_random_data(self):
        max_value = 2 ** self.gen_width - 1
        for i in range(self.n_genes):
            valor = random.randint(0, max_value)
            self.set_bits(valor, i + 1)

    def registrar_mascaras(self, mascara: int, gen: int) -> None:
        self.mascaras["m_" + str(gen)] = mascara

    def get_bits(self, gen: int) -> int:
        return (self.bits & self.mascaras["m_" + str(gen)]) \
               >> self.mascaras["p_" + str(gen)]

    def set_bits(self, valor: int, gen: int) -> None:
        self.bits = (self.bits & ~self.mascaras["m_" + str(gen)]) | \
                    (valor << self.mascaras["p_" + str(gen)])


class CromosomaBinRes(object):
    def __init__(self, c_genes: int, c_bits: int, min_valor: float, max_valor: float):
        self.num_bits = c_bits
        self.num_genes = c_genes
        self.bits_max_valor = 2 ** self.num_bits
        self.min_valor_dec = min_valor
        self.max_valor_dec = max_valor
        self.data = [random.randint(0, self.bits_max_valor - 1) for _ in range(self.num_genes)]
        self.score: int = -1

    def mostrar_tabla(self):
        step = (self.max_valor_dec - self.min_valor_dec) / (2 ** self.num_bits - 1)
        for i in range(2 ** self.num_bits):
            valor = self.min_valor_dec + i * step
            print(f"{bin(i):>012} -> {valor}")

    def get_gen_valor(self, id_gen: int):
        binario = self.data[id_gen]
        step = (self.max_valor_dec - self.min_valor_dec) / (2 ** self.num_bits - 1)
        return self.min_valor_dec + binario * step

    def flip_gen_bit(self, id_gen: int, bit: int):
        if 0 < bit <= self.num_bits:
            pos = 1 << (bit - 1)
            self.data[id_gen] ^= pos
        else:
            print(f"El bit ({bit}) se sale de los bits máximos que acepta el gen, "
                  f"0 < bit <= {self.num_bits}")

    def set_gen_valor(self, id_gen: int, valor: int):
        if valor < 2 ** self.num_bits:
            self.data[id_gen] = valor
        else:
            print("El valor se sale de los valores máximos que acepta el gen")

    def get_all_valores(self):
        """
        Decodifica TODOS los genes a float en una sola operación NumPy.
        ~7× más rápido que llamar get_gen_valor() en un bucle Python.
        Requerido para datasets de alta dimensión (MNIST, etc.).
        """
        import numpy as np
        data = np.array(self.data, dtype=np.float64)
        step = (self.max_valor_dec - self.min_valor_dec) / (2 ** self.num_bits - 1)
        return self.min_valor_dec + data * step
