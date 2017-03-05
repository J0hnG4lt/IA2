#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,cos,sin
from mlpMultiClase import *
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

totalDatosEntrenamiento = int(len(flores)*porcentajesEntrenamiento[0])
totalDatosValidacion = tamanoTotal - totalDatosEntrenamiento

datasetEntrenamiento = flores[0:totalDatosEntrenamiento]
datasetValidacion = flores[totalDatosEntrenamiento:]

resultadosValidacion = MLPMultiClass(nroCapas = 2,
                    data=np.array(datasetEntrenamiento ),
                    datasetValidacion=np.array(datasetValidacion),
                    funcionPorCapa=[logistica, logistica],
                    derivadaFuncionPorCapa=[derivada_logistica,derivada_logistica],
                    nroNeuronasPorCapa = [3,3],
                    maxIter = 1000,
                    aprendizaje = 0.1)

setosa = []
versicolor = []
virginica = []
no_setosa = []
out = []
print(resultadosValidacion)
for flor in resultadosValidacion :
    out.append([flor["respuestaCorrecta"],flor["respuestaSalida"],flor["error"]])

print("Out", out)
#print("No Setosa", no_setosa)


