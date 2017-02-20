#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from random import uniform


    
"""
#    Evaluar Funcion objetivo
#    
#    objetivo : solo hace producto punto de las dos entradas
#    
#    @coeficientes :  coeficientes de la hipotesis
#    @instancia : valores del dominio
"""
def h(coeficientes, instancia) : 
    return np.dot(coeficientes, instancia) 


"""
#    Funcion de Costo
#    
#    Objetivo :  calcula el error de la funcion hipotesis al tratar de 
#                aproximar los datos de entrenamiento usando la norma
#                dada como input
#    
#    @coeficientes : coeficientes de la funcion hipotesis
#    @dom : matriz cuyas columnas representan los valores de su feature
#    @rango : valores que deberia tener la funcion objetivo para una fila
#             de dom
#    @norma : norma a usar con el vector de error
#    
"""
def error_n(coeficientes,dom,rango,norma=2):
    errorAcum = 0
    for i,instancia in enumerate(dom) :
        errorAcum += (h(coeficientes, instancia) - rango[i])**norma
    errorAcum = errorAcum/float(len(rango))
    return errorAcum

"""
#    
#    Objetivo: aproximar un conjunto de datos teniendo como hipotesis que la
#              funcion que genero esos datos tiene la forma:
#                
#                h(<1, x_1, x_2, ... , x_len(dom)>) = sigma{i = 0..len(dom)} coef_i * x_i
#    
#    @dom : matriz de entrenamiento de numpy donde cada columna es un feature y 
#           cada fila una instancia.
#           
#           dom = [[instancias de feature1], [instancias de feature2]...[instancias de feature(N-1)]]
#           
#    @rango : vector columna de entrenamiento con el feature que se quiere predecir.
#             La fila de cada instancia corresponde con la fila de su dominio en
#             'dom'
#             
#             rango = [instancias de valor objetivo]
#
#    @dom : análogo a 'dom', pero para el subconjunto de datos de prueba. 
#           Tiene valor None por defecto
#           
#    @rango : análogo a 'rang_p', pero para el subconjunto de datos de prueba. 
#             Tiene valor None por defecto
#             
#    @max_iter : maximo numero de instancias a usar para el entrenamiento
#    @coeficiente_aprendizaje : que tanto cambia el aprendizaje si encuentra un error
#    @valor_inicial :  valor inicial del vector de coeficientes de la funcion hipotesis
#    
#    @return (coeficientes_por_iteracion,iteraciones,errorPorIteracion)
#        
#        @coeficientes_por_iteracion : coeficientes de la funcion objetivo por cada
#                                      instancia. Los mejores son aquellos para los
#                                      que el errorPorIteracion es el mínimo.
#        @iteraciones : iteracion por cada instancia
#        @errorPorIteracion : error por cada instancia
#        
"""
def regresion_lineal_multiple(dom,
                              rango,
                              max_iter=1000,
                              coeficiente_aprendizaje = 0.01, 
                              dom_p = None, 
                              rang_p = None) :
    
    # Se agrega una columna de 1.0 para x_0
    if dom.ndim == 1:
        dom = np.array([[1.0,instancia] for instancia in dom])
        if dom_p is not None:
            dom_p = np.array([[1.0,instancia] for instancia in dom_p])
    else :
        dom = np.insert(dom, 0, values=1.0, axis=1)
        if dom_p is not None:
            dom_p = np.insert(dom_p, 0, values=1.0, axis=1)
    
    numero_instancias = len(rango)
    numero_atributos = dom.shape[1]
    
  
    coeficientes = np.array([uniform(-0.3,0.3) for i in range(numero_atributos)]) # Se generan los valores iniciales para los pesos
         
    inverso_num_atributos = (1.0/numero_atributos)

    
    coef_anteriores = np.copy(coeficientes)     # Vector de coeficientes 
                                                # antes de la iteracion actual
                                                
    coeficientes_por_iteracion = []
    coeficientes_por_iteracion_p = []  #coeficientes que se obtienen en la validación             

    # Declaramos una constante que es el aprendizaje / el numero de atributos
    constante = (coeficiente_aprendizaje * inverso_num_atributos)

    iteraciones = 0
    errorPorIteracion = [] # Almacena el error de entrenamiento de cada iteración
    errorPorIteracion_p = [] #Almacena el error de prueba de cada iteración

    # Cuando se disminuya el error por menos de esto detenemos
    epsilon = 10**-6
    errorAnt = 1000000
    errorIter = 0

    while (iteraciones < max_iter and abs(errorAnt - errorIter) > epsilon) :

        errorAnt = errorIter
        errorIter = 0

        if (iteraciones % 100) == 0:
            print("Iteraciones = " + str(iteraciones))
        error = []
        for i in range(numero_instancias) :
            # Se obtiene el vector de error

            error.append(h(coef_anteriores,dom[i]) - rango[i])

        # Se actualiza cada coeficiente con el vector de error
        for j in range(numero_atributos): 
            derivada = np.dot(error,dom[:,j])
            coeficientes[j] = coef_anteriores[j] - constante * derivada

        

        # Calculamos el error de la iteracion
        errorIter = error_n(coeficientes,dom,rango)    

        # Seran utiles al graficar
        coeficientes_por_iteracion.insert(iteraciones, coef_anteriores)
        errorPorIteracion.insert(iteraciones,errorIter)
        iteraciones += 1
        
        coef_anteriores = np.copy(coeficientes)
        
        
        # VALIDAMOS
        # Si se pasaron datos para la validación
        if dom_p is not None:
            salidas_p = np.zeros(len(rang_p))
                    
            # Calculamos el error de la validación para la iteración
           
            errorIter_p = error_n(coeficientes,dom_p,rang_p) 
            
            errorPorIteracion_p.insert(iteraciones,errorIter_p)
        
    return (coeficientes_por_iteracion,iteraciones,errorPorIteracion,errorPorIteracion_p)

"""
#    Funcion de Normalizacion 
#    
#    Objetivo: calcula la desviacion estandar y la media por columna
#    y los usa para normalizar todos las instancias e   n ese feature usando la
#    siguiente formula (vector_feature_i - promedio_feature_i) / desviacion_feature_i
#    
#    @matriz : matriz de numpy donde cada columna representa un feature y cada
#              fila es una instancia
#
#   @return : retorna la misma matriz con las columnas normalizadas
#
"""
def normalizarZ(matriz) :
    promedio_por_columna = np.mean(matriz,axis=0)
    desviacion_por_columna = np.std(matriz,axis=0)
    (m,n) = matriz.shape
    for i in range(n):
        matriz[:,i] = (matriz[:,i] - promedio_por_columna[i])/desviacion_por_columna[i]
    
    return matriz

"""
#    Funcion de evaluacion de modelo
#    
#    Objetivo: evalua al modelo usando los cuatro estadisticos sugeridos por
#    Dean De Cock en "Ames, Iowa: Alternative to the Boston Housing Data as an
#    End of Semester Regression Project", pagina septima.
#    
#    @dominio : matriz de numpy con los valores para varias instancias
#    de los features usados como dominio del modelo
#    
#    @rango : arreglo de numpy con los valores que debio haber aproximado
#    el modelo a partir de los valores del dominio
#    
#    @coeficientes : coeficientes del modelo de regresion lineal multiple
#    
#    @return : diccionario con los cuatro estadisticos
"""
def evaluar_modelo(dominio, rango, coeficientes) :
    diferencias = []
    for index,instancia in enumerate(dominio) :
        diferencias.append(np.dot(np.insert(instancia, 0,1), coeficientes) - rango[index])
    cantidad_elementos = float(len(diferencias))
    bias = sum(diferencias)/cantidad_elementos
    max_deviation = max(diferencias)
    mean_absolute_deviation = sum(map(abs,diferencias))/cantidad_elementos
    mean_square_error = sum(map((lambda x : x**2),diferencias))/cantidad_elementos
    
    return {"bias":bias, 
            "max_deviation":max_deviation, 
            "mean_absolute_deviation":mean_absolute_deviation, 
            "mean_square_error": mean_square_error}




