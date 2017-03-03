import numpy as np
import matplotlib.pyplot as plt
from math import sqrt

def dentroDeCircunferencia((Xcoord, Ycoord)) : 
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
    areas = map(dentroDeCircunferencia, puntos)
    return zip(puntosX,
               puntosY,
               areas)
    

patrones = generarPatrones(numeroPuntos = 2000)
plt.scatter([x[0] for x in patrones if x[2]], [x[1] for x in patrones if x[2]], color="blue")
plt.scatter([x[0] for x in patrones if not x[2]], [x[1] for x in patrones if not x[2]], color="red")
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
