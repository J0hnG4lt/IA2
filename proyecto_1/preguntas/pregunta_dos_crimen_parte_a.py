#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from proyecto import normalizarZ, regresion_lineal_multiple

if __name__ == '__main__':
    
    # Se abre el archivo de prueba limpiado
    with open("x08_copia.txt", "r") as f : 
        datos_x08 = np.loadtxt( fname=f, 
                                dtype=float ,
                                comments="#" ,
                                delimiter=",",
                                usecols=(1,2,3,4))
        f.close()
    
    datos_x08 = normalizarZ(datos_x08)
    
    dominio = datos_x08[:,0:3]
    rango = datos_x08[:,3]
    aprendizaje = 0.1
    coeficientes,iteraciones,errorPorIteracion = regresion_lineal_multiple( dom=dominio, 
                                                                            rango=rango,
                                                                            coeficiente_aprendizaje=aprendizaje)
    errorPorIteracion = np.array(errorPorIteracion)
    mejor_iter = np.where(errorPorIteracion == errorPorIteracion.min())      # Mejor Iteracion
    print("Mejor Iteracion", mejor_iter[0][0])
    print(coeficientes[mejor_iter[0][0]], iteraciones)
    
    plt.plot(range(iteraciones), errorPorIteracion)
    plt.title("Convergencia -Homicidios- (normalizado)")
    plt.xlabel("Numero de Iteraciones")
    plt.ylabel("Error")
    plt.text(5,0.2, 'Aprendizaje = {0:.1f}'.format(aprendizaje))
    
    plt.show()
