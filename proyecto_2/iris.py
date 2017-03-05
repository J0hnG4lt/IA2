#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,cos,sin
from mlp import *
from generarPatrones import normalizar, logistica, derivada_logistica, tanh, derivada_tanh


# Lectura de datos
with open("datosP2EM2017/iris_setosa.data","r") as file :
    lines = file.readlines()
    flores = []
    for l in lines:
        flores.append(list(map(float,l.strip("\n\r").split(","))))
    file.close()

tamanoTotal = len(flores)
porcentajesEntrenamiento = [0.5,0.6,0.7,0.8,0.9]

# Se mezclan los datos
flores = np.random.permutation(flores)


# Se particionan los datos
totalDatosEntrenamiento = int(len(flores)*porcentajesEntrenamiento[4])
totalDatosValidacion = tamanoTotal - totalDatosEntrenamiento

datasetEntrenamiento = flores[0:totalDatosEntrenamiento]
datasetValidacion = flores[totalDatosEntrenamiento:]

# Entrenamiento del MLP
resultadosValidacion,errorPorIteracion = MLP(nroCapas = 2,
                    data=np.array(datasetEntrenamiento ),
                    datasetValidacion=np.array(datasetValidacion),
                    funcionPorCapa=[logistica,logistica],
                    derivadaFuncionPorCapa=[derivada_logistica ,derivada_logistica],
                    nroNeuronasPorCapa = [2,1],
                    maxIter = 1000,
                    aprendizaje = 0.1)

setosa = []
versicolor = []
virginica = []
no_setosa = []
errorDePrueba = 0
cantCasos = 0

# Verificacion de resultados
for flor in resultadosValidacion :
    errorDePrueba += sum(flor["error"])
    cantCasos += len(flor["error"])
    if flor["respuestaSalida"][0] > 0.5 :
        setosa.append(flor["respuestaCorrecta"] == 1)
    else :
        no_setosa.append(flor["respuestaCorrecta"] == 0)

print("Setosa", setosa)
print("No Setosa", no_setosa)

print("Cantidad de Instancias (Entrenamiento): ", totalDatosEntrenamiento)
print("Error de Prueba: ", errorDePrueba/cantCasos)
print("Falsos Positivos: ", sum(1 for x in setosa if not x))
print("Falsos Negativos: ", sum(1 for x in no_setosa if not x))

y1 = plt.plot(range(len(errorPorIteracion)),errorPorIteracion)
plt.title("Curva de Convergencia -Flores-")
plt.xlabel("Numero de Iteraciones")
plt.ylabel("Error")
plt.show()
plt.show()

