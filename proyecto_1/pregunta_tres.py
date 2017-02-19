#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from proyecto import normalizarZ, regresion_lineal_multiple, evaluar_modelo

if __name__ == '__main__':
    
    nombre_dataset = "AmesHousing_sin_missing_values_filtrados_segun_paper_80_cortado.csv"
    #usecols = (i for j in (range(0,80),range(81,82)) for i in j)
    usecols = range(0,20)
    # Se abre el archivo de prueba limpiado
    with open(nombre_dataset, "r") as f:
        datos = np.loadtxt( fname=f, 
                                dtype=float ,
                                comments="#" ,
                                delimiter=",",
                                usecols=usecols,
                                skiprows=1)
        f.close()
        
       
    datos = normalizarZ(datos)

    nro_datos = len(datos[:,1])
    
    # El número de datos del entrenamiento es el 80% de los datos totales
    nro_datos_entrenamiento = int(round(0.8*nro_datos))
    
    # El número datos para pruebas es el 20% de los datos totales
    nro_datos_prueba = nro_datos - nro_datos_entrenamiento
    
    # Se permutan los datos de forma aleatoria
    datos = np.random.permutation(datos)
    
    prueba = datos[0:nro_datos_prueba,:]
    entrenamiento = datos[nro_datos_prueba:nro_datos,:]
    
    dominio = entrenamiento[:,0:19]
    rango = entrenamiento[:,19]
    
    
    aprendizaje = 0.001
    
    coeficientes,iteraciones,errorPorIteracion = regresion_lineal_multiple( dom=dominio, 
                                                                            rango=rango,
                                                                            coeficiente_aprendizaje=aprendizaje)
    
    
    errorPorIteracion = np.array(errorPorIteracion)
    mejor_iter = np.where(errorPorIteracion == errorPorIteracion.min())     # Mejor Iteracion
    print(mejor_iter)
    evaluacion = evaluar_modelo(dominio, rango, coeficientes[mejor_iter[0][0]])
    print(evaluacion)
    
    plt.figure(1)
    plt.plot(range(iteraciones), errorPorIteracion)
    plt.title("Convergencia Ventas")
    plt.xlabel("Numero de Iteraciones")
    plt.ylabel("Error")
    plt.text(60,0.95, 'Aprendizaje = {0:.3f}'.format(aprendizaje))
    plt.show()
    
    
    dominio_p = prueba[:,0:19]
    rango_p = prueba[:,19]
        
    coeficientes_p,iteraciones_p,errorPorIteracion_p = regresion_lineal_multiple( dom=dominio_p, 
                                                                            rango=rango_p,
                                                                            max_iter = 1,
                                                                            coeficiente_aprendizaje=aprendizaje)
                                                                            
    
    print(coeficientes_p)                                                                        
    evaluacion_p = evaluar_modelo(dominio_p, rango_p, coeficientes_p[0])
    print("Evaluacion con el conjunto de prueba: ", evaluacion_p)
    #plt.figure(2)
    #plt.plot(rango_p,range(20)'*g')
    #plt.plot(coeficientes,'x')
    #plt.plot(coeficientes_p,'xr')
    #plt.title("")
    #plt.text(60,0.95, 'Aprendizaje = {0:.8f}'.format(aprendizaje))
    #plt.text(60,0.9, 'Inicial = {0:.3f}'.format(valor_inicial))
    #plt.show()
    
                                                                            
