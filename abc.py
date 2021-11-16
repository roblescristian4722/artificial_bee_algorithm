#!/usr/bin/env python
# ALGORITMO ABC (ARTIFICIAL BEE COLONY) PARA OPTIMIZACIÓN
import numpy as np
from random import random
from matplotlib import pyplot as plt

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
gen = 100
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

# Mejor abeja obrera de cada generación
best_fp = 0
# Array en el que se almacena el promedio de soluciones de cada generación con
# la intención de graficarlas al final de la ejecución
gen_mean = []
best_mean = max
best_gen = 0

# Impresión de datos iniciales en la terminal
print("Datos iniciales:")
print(f"Espacio de búsqueda: [{min}, {max}]")
print(f"Tamaño de la población: {pop_size}")
print(f"Cantidad de generaciones: {gen}")
print(f"Cantidad de dimensiones (características x): {dim - 1}")
print(f"Tamaño de la población de abejas obreras (forager bees): {fps}")
print(f"Tamaño de la población de abejas espectadoras (onlooker bees): {ops}")

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
    gen_mean.append(np.mean(fp[:, 0]))
    if fitness(best_mean) > fitness(np.mean(fp[:, 0])):
        best_mean = np.mean(fp[:, 0])
        best_gen = g

# Impresión de resultados en pantalla
print(f"\nMejor resultado encontrado (media): {best_mean}")
print(f"Encontrado en la generación: {best_gen}")
print(f"Último resultado encontrado (media): {gen_mean[len(gen_mean) - 1]}")

# Graficación
fig = plt.figure()
ax = fig.add_subplot()
fig.suptitle("Optimización por colonia de abejas (Artificial Bee Colony)")
ax.set_xlabel("Generaciones")
ax.set_ylabel("Costo")
plt.plot(gen_mean, 'b-', label='Promedio de las soluciones')
plt.plot(best_gen, best_mean, 'r+', label='Mejor promedio de soluciones obtenido')
plt.plot(gen - 1, gen_mean[gen - 1], 'r*', label='Último promedio de soluciones obtenido')
plt.plot([ 0 for _ in range(gen) ], 'g--', label='Solución óptima (mínimo global)')
ax.legend()
plt.show()
