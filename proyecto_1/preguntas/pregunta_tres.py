#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from proyecto import normalizarZ, regresion_lineal_multiple, evaluar_modelo, h

if __name__ == '__main__':
    
    nombre_dataset = "../datasets/dataActividad3.csv"

    #usecols = (i for j in (range(0,80),range(81,82)) for i in j)
    usecols = range(0,20)
    # Se abre el archivo de prueba limpiado
    with open(nombre_dataset, "r") as f:
        datos = np.loadtxt( fname=f, 
                                dtype= float,
                                comments="#" ,
                                delimiter=",",
                                usecols=usecols,
                                skiprows=1)
        
        
    datos = normalizarZ(datos)
    
    nro_datos = len(datos[:,1])
    
    # El número de datos del entrenamiento es el 80% de los datos totales
    nro_datos_entrenamiento = int(round(0.8*nro_datos))
    
    # El número datos para pruebas es el 20% de los datos totales
    nro_datos_prueba = nro_datos - nro_datos_entrenamiento

    # Se permutan los datos de forma aleatoria
    datos = np.random.permutation(datos)
    
    # Subconjunto de datos para la prueba
    prueba = datos[0:nro_datos_prueba,:]
    
    # Subconjunto de datos para el entrenamiento
    entrenamiento = datos[nro_datos_prueba:nro_datos,:]
    
    dominio = entrenamiento[:,0:19]
    rango = entrenamiento[:,19]    
    dominio_p = prueba[:,0:19]
    rango_p = prueba[:,19]
    
    aprendizaje = 0.001
    
    coeficientes,iteraciones,errorPorIteracion, errorPorIteracionPrueba= regresion_lineal_multiple( dom=dominio, 
                                                                            rango=rango,
                                                                            coeficiente_aprendizaje=aprendizaje,
                                                                            dom_p = dominio_p,
                                                                            rang_p = rango_p)
    
    error = np.array(errorPorIteracionPrueba)
    mejor_iter = np.where(error == error.min())     # Mejor Iteracion
    evaluacion = evaluar_modelo(dominio, rango, coeficientes[mejor_iter[0][0]])
    print(evaluacion)
    # Grafica de error de entrenamiento y de validación por iteración    
    plt.plot(range(iteraciones), errorPorIteracion,'r-',label="Error de entrenamiento")
    plt.plot(range(iteraciones), errorPorIteracionPrueba,'g-',label="Error de prueba")
    plt.title("Convergencia Ventas")
    plt.xlabel("Numero de Iteraciones")
    plt.ylabel("Error")
    plt.text(60,0.95, 'Aprendizaje = {0:.3f}'.format(aprendizaje))
    plt.legend()
    plt.show()    
    

    '''
    Comparacion Visual
    '''

    resultados = []
    for index,instancia in enumerate(dominio_p) :
        resultados.append(np.dot(np.insert(instancia, 0,1), coeficientes[mejor_iter[0][0]]))
    
    plt.plot(0,0,"ro",label="Resultado Actual")
    plt.plot(0,0,"bx",label="Resultado Obtenido")

    for c,i in enumerate(rango_p):
        plt.plot(c,i,"ro")

    for c,i in enumerate(resultados):
        plt.plot(c,i,"bx")
    plt.xlabel("Numero de Instancia")
    plt.ylabel("Valor (Normalizado) de instancia")
    plt.legend()
    plt.show()
