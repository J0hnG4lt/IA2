#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,cos,sin
from mlp import *
from generarPatrones import normalizar, logistica, derivada_logistica, tanh, derivada_tanh


with open("datosP2EM2017/iris_setosa.data","r") as file :
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

resultadosValidacion = MLP(nroCapas = 1,
                    data=np.array(datasetEntrenamiento ),
                    datasetValidacion=np.array(datasetValidacion),
                    funcionPorCapa=[logistica],
                    derivadaFuncionPorCapa=[derivada_logistica],
                    nroNeuronasPorCapa = [2],
                    maxIter = 1000,
                    aprendizaje = 0.1)

setosa = []
versicolor = []
virginica = []
no_setosa = []
for flor in resultadosValidacion :
    if flor["respuestaSalida"] > 0.5 :
        setosa.append(flor["respuestaCorrecta"] == 1)
    else :
        no_setosa.append(flor["respuestaCorrecta"] == 0)

print("Setosa", setosa)
print("No Setosa", no_setosa)


