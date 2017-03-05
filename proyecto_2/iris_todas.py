#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,cos,sin
from mlp import *
from generarPatrones import normalizar, logistica, derivada_logistica, tanh, derivada_tanh


with open("datosP2EM2017/iris_todas.data","r") as file :
    lines = file.readlines()
    flores = []
    for l in lines:
        flores.append(list(map(float,l.strip("\n\r").split(","))))
    file.close()

tamanoTotal = len(flores)
porcentajesEntrenamiento = [0.5,0.6,0.7,0.8,0.9]

flores = np.random.permutation(flores)

totalDatosEntrenamiento = int(len(flores)*porcentajesEntrenamiento[3])
totalDatosValidacion = tamanoTotal - totalDatosEntrenamiento

datasetEntrenamiento = flores[0:totalDatosEntrenamiento]
datasetValidacion = flores[totalDatosEntrenamiento:]

resultadosValidacion,errorPorIteracion = MLP(nroCapas = 3,
                    data=np.array(datasetEntrenamiento ),
                    datasetValidacion=np.array(datasetValidacion),
                    funcionPorCapa=[lambda x:x,logistica, logistica],
                    derivadaFuncionPorCapa=[lambda x:1,derivada_logistica,derivada_logistica],
                    nroNeuronasPorCapa = [4,15,3],
                    maxIter = 1000,
                    aprendizaje = 0.01)

setosa = []
versicolor = []
virginica = []

errorDePrueba = 0
cantCasos = 0
for flor in resultadosValidacion :
    errorDePrueba += sum(flor["error"])
    cantCasos += len(flor["error"])
    if flor["respuestaSalida"][0] > 0.5 :
        setosa.append(flor["respuestaCorrecta"] == 1)
    
    print(flor["respuestaSalida"])
    
"""
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

"""
