#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from mlp import *

# Taken from mpl2
def logistica(x):
    return 1/(1 + np.exp(-x))

def derivada_logistica(y):
    return np.exp(-x)/((1+np.exp(-x))**2)


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
    

patrones = generarPatrones(numeroPuntos = 2000)
patrones = generarPatrones(numeroPuntos = 2000)

patrones_array = np.array([[x,y,float(z)] for (x,y,z) in patrones])
MLP(nroCapas = 1,
    data=patrones_array,
    funcionPorCapa=[logistica],
    derivadaFuncionPorCapa=[derivada_logistica])
#plt.scatter([x[0] for x in patrones if x[2]], [x[1] for x in patrones if x[2]], color="blue")
#plt.scatter([x[0] for x in patrones if not x[2]], [x[1] for x in patrones if not x[2]], color="red")
#plt.gca().set_aspect('equal', adjustable='box')
#plt.show()
