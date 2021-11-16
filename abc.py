#!/usr/bin/env python
# ALGORITMO ABC (ARTIFICIAL BEE COLONY) PARA OPTIMIZACIÓN
import numpy as np
from random import random

def fitness(x):
    """Función benchmark (absolute)"""
    return np.abs(x)

def generate_matrix(min: float, max: float, sol: int, dim: int):
    """Genera la matriz de solucionesXdimensiones"""
    x = np.zeros(shape=(sol, dim))
    offset = abs(min) + abs(max)
    for i in range(sol):
        for j in range(dim - 1):
            x[i, j] = ((random() * 100) % offset) + min
    return x

# Parámetros del ABC
min = -10
max = 10
pop_size = 50
gen = 1000
dim = 2
# Límite de estancamiento generacional
L = 5
# Forager population size (Tamaño de la población de abejas obreras)
fps = int(pop_size * (2 / 3))
# Onlooker population size (Tamaño de la poblacińo de abejas espectadoras)
ops = pop_size - fps
# Matriz de población de hormigas obreras (filas: soluciones, col1: abeja, 
# col2: estancamiento de la abeja)
fp = generate_matrix(min, max, pop_size, dim)

best_fp = 0

for g in range(gen):
    # Abejas obreras
    for i in range(fps):
        # Posición de la abeja obrera distinta a la actual
        k = int(random() * 100) % fps
        while k == i:
            k = int(random() * 100) % fps
        # Factor de combinación aleatorio (de entre -1 a 1)
        r = -1 + (random() * 100) % 2
        v = fp[i, 0] + r * (fp[i, 0] - fp[k, 0])
        if fitness(v) < fitness(fp[i, 0]):
            fp[i, :] = np.array([ v, 0 ])
        else:
            fp[i, 1] += 1
        # Se guarda el índice de la mejor abeja obrera para que después sea
        # analizado su fitness por la abeja espectadora
        if fitness(fp[i, 0]) < fitness(fp[best_fp, 0]):
            best_fp = i

    # Abejas espectadoras
    for i in range(ops):
        # Posición de la abeja espectadora distinta a la actual
        k = int(random() * 100) % ops
        while k == i:
            k = int(random() * 100) % ops
        # Factor de combinación aleatorio (de entre -1 a 1)
        r = -1 + (random() * 100) % 2
        v = fp[best_fp, 0] + r * (fp[best_fp, 0] - fp[k, 0])
        if fitness(v) < fitness(fp[best_fp, 0]):
            fp[best_fp, :] = np.array([ v, 0 ])
        else:
            fp[best_fp, 1] += 1

    # Abeja reconocedora
    for i in range(fps):
        if fp[i, 1] > L:
            offset = abs(min) + abs(max)
            fp[i] = np.array([ ((random() * 100) % offset) + min, 0 ])
    print(np.mean(fp[:, 0]))
