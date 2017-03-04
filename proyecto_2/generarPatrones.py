#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from mlp import *

# Taken from mpl2
def logistica(x):
    return 1/(1 + np.exp(-x))

def derivada_logistica(x):
    return np.exp(-x)/((1+np.exp(-x))**2)

def tanh(x):
    return np.tanh(x)

def derivada_tanh(x):
    return 1.0 - np.tanh(x)**2

def dentroDeCircunferencia(*coords) :
    Xcoord = coords[0]
    Ycoord = coords[1]
    if abs(Xcoord - 10.0) >  6.0:
        return False
    temp = sqrt(36.0 - (10.0-Xcoord)**2)
    if Ycoord >= 10.0 :
        return Ycoord < (10.0 + temp)
    else :
        return Ycoord >= (10.0 - temp)

def generarPatrones(numeroPuntos = 2000) :
    
    puntosX = np.random.uniform(0, 20, numeroPuntos)
    puntosY = np.random.uniform(0, 20, numeroPuntos)
    
    puntos = zip(puntosX, puntosY)
    areas = map(lambda coords: dentroDeCircunferencia(coords[0], coords[1]), puntos)
    return zip(puntosX,
               puntosY,
               areas)
    

def normalizar(data):
    # the last one is assumed to be the result
    for i in range(len(data[0]) - 1):
        mean = sum(instancia[i] for instancia in data) / len(data)
        stddev = sqrt(sum((instancia[i] - mean) **2 for instancia in data) / len(data))
        for j in range(len(data)):
            data[j][i] = (data[j][i] - mean)/stddev
    return data


with open("datosP2EM2017/datos_P2_EM2017_N2000.txt","r") as file :
    lines = file.readlines()
    patrones = []
    for l in lines:
        patrones.append(tuple(l.strip("\n\r").split(" ")))
    file.close()

patrones_array = np.array([[float(x),float(y),float(z)] for (x,y,z) in patrones])
patrones_array = normalizar(patrones_array)


#patrones_array = 

unos =sum(1 for i in patrones_array if i[2] == 1)
ceros = sum(1 for i in patrones_array if i[2] == 0)
print(unos,ceros)

resultadosValidacion = MLP(nroCapas = 3,
                    data=patrones_array,
                    funcionPorCapa=[lambda x: x, lambda x: x**2, logistica],
                    derivadaFuncionPorCapa=[lambda x : 1, lambda x : 2*x ,derivada_logistica],
                    nroNeuronasPorCapa = [1,2,1],
                    maxIter = 1000,
                    aprendizaje = 0.1)

fuera = []
dentro = []
for instancia in resultadosValidacion :
    print("RESPUESTA: ", 1 if instancia["respuestaSalida"] > 0.5 else 0, instancia["respuestaCorrecta"])
    if instancia["respuestaSalida"] < 0.5:
        aux = [instancia["punto"], 0 == instancia["respuestaCorrecta"]]
        dentro.append(aux)
    else:
        aux = [instancia["punto"], 1 == instancia["respuestaCorrecta"]]
        fuera.append(aux)
    

#plt.figure(0)
print(dentro[0])
plt.scatter([x[0][0] for x in dentro if x[1] == 0],
             [x[0][1] for x in dentro if x[1] == 0], color="blue", marker = "x")
plt.scatter([x[0][0] for x in fuera if x[1] == 0 ], 
            [x[0][1] for x in fuera if x[1] == 0 ], color="red", marker= "x")
plt.scatter([x[0][0] for x in dentro if x[1] == 1 ], 
            [x[0][1] for x in dentro if x[1] == 1], color="blue" , marker = "o")
plt.scatter([x[0][0] for x in fuera if x[1] == 1],
            [x[0][1] for x in fuera if x[1] == 1], color="red" , marker = "o")
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.add_artist(plt.Circle((10,10),6,color= "red", fill = False))
plt.show()

"""
plt.figure(1)
plt.scatter([x[0] for x in patrones_array if x[2]], [x[1] for x in patrones_array if x[2]], color="blue")
plt.scatter([x[0] for x in patrones_array if not x[2]], [x[1] for x in patrones_array if not x[2]], color="red")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
"""
