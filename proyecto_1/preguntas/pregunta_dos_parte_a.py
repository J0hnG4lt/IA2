#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from proyecto import normalizarZ, regresion_lineal_multiple

if __name__ == '__main__':
    
    # Se abre el archivo de prueba limpiado
    with open("../datasets/x01_copia.txt", "r") as f : 
        datos_x01 = np.loadtxt( fname=f, 
                                dtype=float ,
                                comments="#" ,
                                delimiter=",",
                                usecols=(1,2))
        f.close()
    
    datos_x01 = normalizarZ(datos_x01)
    
    dominio = datos_x01[:,0]
    rango = datos_x01[:,1]
    
    aprendizaje = 0.01
    coeficientes,iteraciones,errorPorIteracion = regresion_lineal_multiple( dom=dominio, 
                                                                            rango=rango,
                                                                            coeficiente_aprendizaje=aprendizaje)
    
    errorPorIteracion = np.array(errorPorIteracion)
    mejor_iter = np.where(errorPorIteracion == errorPorIteracion.min())     # Mejor Iteracion
    print("Mejor Iteracion: ", mejor_iter[0][0])
    print(coeficientes[mejor_iter[0][0]], iteraciones)
    
    plt.plot(range(iteraciones), errorPorIteracion)
    plt.title("Curva de Convergencia -Peso- (normalizado)")
    plt.xlabel("Numero de Iteraciones")
    plt.ylabel("Error")
    plt.text(5,0.5, 'Aprendizaje = {:.2f}'.format(aprendizaje))
    plt.show()
