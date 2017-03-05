#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt,log
from mlp import *

# Taken from mpl2
def logistica(x):
    return 1/(1 + np.exp(-x))

def derivada_logistica(x):
    return logistica(x)* (1-logistica(x))

def tanh(x):
    return np.tanh(x)

def softplus(x):
    return log(1 + np.exp(x))

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

def generarPatronesMismasAreas(numeroDentro=200,numeroFuera=300) :
    nFuera = 0
    nDentro = 0
    puntos = []
    while((nFuera < numeroFuera) or (nDentro < numeroDentro) ) :
        
        puntoX = np.random.uniform(0, 20)
        puntoY = np.random.uniform(0, 20)
        estaAdentro = dentroDeCircunferencia(puntoX,puntoY)
        instancia = (puntoX,puntoY,estaAdentro)
        if  estaAdentro and (nDentro < numeroDentro):
            puntos.append(instancia)
            nDentro += 1
        if  (not estaAdentro) and (nFuera < numeroFuera):
            puntos.append(instancia)
            nFuera += 1
    return puntos
    

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

#patrones_array = np.array([[float(x),float(y),float(z)] for (x,y,z) in patrones])
#patrones_array = normalizar(patrones_array)


patrones_entrenamiento = generarPatrones(numeroPuntos = 2000)
patrones_array = np.array([[float(x),float(y),float(z)] for (x,y,z) in patrones_entrenamiento])
patrones_array = normalizar(patrones_array)

unos =sum(1 for i in patrones_array if i[2] == 1)
ceros = sum(1 for i in patrones_array if i[2] == 0)

patrones_validacion = generarPatronesMismasAreas(numeroFuera = ceros, numeroDentro = unos)
puntos_generados = np.array([[float(x),float(y),float(z)] for (x,y,z) in patrones_validacion])
puntos_generados = normalizar(puntos_generados)


resultadosValidacion = MLP(nroCapas = 4,
                    data=patrones_array,
                    datasetValidacion=puntos_generados,
                    funcionPorCapa=[lambda x: x , logistica  , lambda x : x**2 , softplus],
                    derivadaFuncionPorCapa=[lambda x: 1 , derivada_logistica , lambda x : 2*x ,logistica],
                    nroNeuronasPorCapa = [1,2,5,1],
                    maxIter = 1000,
                    aprendizaje = 0.1)

fuera = []
dentro = []
fueraT = []
dentroT = []
for instancia in resultadosValidacion :
    if instancia["respuestaSalida"] < 0.5:
        aux = [instancia["punto"], instancia["respuestaCorrecta"]]
        dentro.append(aux)
    else:
        aux = [instancia["punto"], instancia["respuestaCorrecta"]]
        fuera.append(aux)    

x1 = plt.scatter([x[0][0] for x in dentro if x[1] == 0],
             [x[0][1] for x in dentro if x[1] == 0], color="red", marker = "x")
x2 = plt.scatter([x[0][0] for x in fuera if x[1] == 0 ], 
            [x[0][1] for x in fuera if x[1] == 0 ], color="blue", marker= "o")
x3 = plt.scatter([x[0][0] for x in dentro if x[1] == 1 ], 
            [x[0][1] for x in dentro if x[1] == 1], color="blue" , marker = "x")
x4 = plt.scatter([x[0][0] for x in fuera if x[1] == 1],
            [x[0][1] for x in fuera if x[1] == 1], color="red" , marker = "o")
ax = plt.gca()
ax.set_aspect('equal', adjustable='box')
ax.add_artist(plt.Circle((0,0),1,color= "red", fill = False))
plt.legend((x1, x2, x3, x4),
           ('Clasificado mal como interno', 
            'Clasificado bien como externo',
            'Clasificado bien como interno', 
            'Clasificado mal como externo'),
           scatterpoints=1,
           loc='lower left',
           ncol=2,
           fontsize=8)
plt.show()

