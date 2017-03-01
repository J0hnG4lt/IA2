#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


def logistica(x):
    return 1/(1 + np.exp(-x))

# y : resultado de aplicar la función logística
def derivada_logistica(y):
    return y*(1-y)

"""
#   MPL: Perceptrón multicapa
#
#   @data : Conjunto de datos: filas intancias, columnas features
#   @porcentaje_prueba: porcentaje de la data que será usado para la validación,
#                       es cero cuando no se hace validación
#   @max_iter: La cantidad máxima de iteraciones
#   @eta:  tasa de aprendizaje
#   @nro_capas_ocultas: Número de capas ocultas de la red
#   @nro_neuronas_capa_oculta1: Número de neuronas de la primera capa oculta
#                               
#   @nro_neuronas_capa_oculta2: Número de neuronas de la 2da capa oculta
#                               Es cero cuando hay una sola capa

"""

def MPL(data,
        porcentaje_prueba,
        max_iter,
        eta, 
        nro_capas_ocultas, 
        nro_neuronas_capa_oculta1, 
        nro_neuronas_capa_oculta2,
        nro_neuronas_capa_salida):
    
    nro_instancias = len(data) # Número de instancias que tiene el conjunto de datos completo
    nro_atributos = len(data[0]) -1
    
    ''' SE PARTICIONAN LOS DATOS EN ENTRENAMIENTO y VALIDACIÓN '''
    
    # Se permutan los datos de forma aleatoria
    data = np.random.permutation(data)
           
    # El número de datos para la validación según el porcentaje dato
    nro_datos_prueba = int(round(porcentaje_prueba*nro_instancias/100))
    
    # El número datos para el entrenamiento
    nro_datos_entrenamiento = nro_instancias - nro_datos_prueba
    
    # Subconjunto de datos para la prueba
    prueba = data[0:nro_datos_prueba,:]
    
    # Subconjunto de datos para el entrenamiento
    entrenamiento = data[nro_datos_prueba:nro_instancias,:]
    
    
    ''' SE INICIALIZA LA RED '''
    
    W  = [0,0,0] # Pesos (coeficientes)
    b  = [0,0,0] # Sesgos
    y  = [0,0,0] # Salida de las capas para una iteración 
    derivadas = [0,0,0]  # Derivadas de la funcion de activación en cada capa
    deltas = [0,0,0] # Deltas para cada capa
    salidas = [] # Salida de la red para cada iteración
    errores = [] # Error de la red para cada iteración
    
    # Primera capa oculta
    W[0] = (np.random.rand(nro_neuronas_capa_oculta1,nro_atributos))
    b[0] = (np.random.rand(nro_neuronas_capa_oculta1,1))
    
    # Segunda capa oculta en caso de que exista
    if nro_capas_ocultas == 2:
        W[1] = (np.random.rand(nro_neuronas_capa_oculta2,nro_neuronas_capa_oculta1))
        b[1] = (np.random.rand(nro_neuronas_capa_oculta2,1))
    
        # Capa de salida
        W[2] = (np.random.rand(nro_neuronas_capa_salida,nro_neuronas_capa_oculta2))
        b[2] = (np.random.rand(nro_neuronas_capa_salida,1))
    else:
        # Capa de salida
        W[1] = (np.random.rand(nro_neuronas_capa_salida,nro_neuronas_capa_oculta1))
        b[1] = (np.random.rand(nro_neuronas_capa_salida,1))
   
    Dominio = (entrenamiento[:,:-1])
    '''
    for atributo in range(nro_atributos):
        maximo = max(Dominio[:,atributo])
        minimo = min(Dominio[:,atributo])
        Dominio = (maximo -  Dominio)/(maximo -minimo)'''
        
    
    Rango = entrenamiento[:,-1]
    
    ''' Ciclo principal '''
    for i in range(max_iter):
        salidas = []
        
        errores += [0] # Se inicializa el error de la iteración actual
        
               
        # Para cada instancia en el conjunto de datos de entrenamiento
        for dom,rango in zip(Dominio,Rango):
            
            # PROPAGACIÓN HACIA ADELANTE ---------------------------------------
            
            entradas = [np.transpose(np.array([dom]))]
            
            # Para cada capa oculta
            for capa in range(nro_capas_ocultas):
                y[capa] = (logistica(np.dot(W[capa],entradas[capa])+b[capa])) # Salida de capa actual        
                entradas += [y[capa]] # La entrada para la capa siguiente es la salida de la actual
               
            capa += 1 # Se incrementa el nro de capa actual, 
                      # para trabajar con la capa de salida
                      
            y[capa] = np.dot(W[capa],entradas[capa])+b[capa] # Salida de capa    
            
            salidas += [y[capa]]
            
            error = rango - y[capa] # Error de la muestra actual para la iteracion
            
            errores[i] += error**2 # Se almacena en la lista de errores para cada iteración
            
            # PROPAGACION HACIA ATRAS ------------------------------------------
            
            # Cálculo de las derivadas de la funcion de activación en cada capa
            
            for capa in range(nro_capas_ocultas): # Para cada capa oculta
                derivadas[capa] = (derivada_logistica(y[capa]))        
            
            capa += 1
            derivadas[capa] = W[capa] # Para la capa de salida
                        
            # se calculan los deltas
            
            deltas[capa] = (error)* (derivadas[capa])
                        
            for capa in reversed(range(nro_capas_ocultas)): # Para cada capa oculta
                if np.shape(deltas[capa+1]) == (1,1):
                    deltas[capa] = derivadas[capa] * deltas[capa+1] * (W[capa])
                else:
                    deltas[capa] = derivadas[capa] * np.dot(deltas[capa+1], (W[capa]))
            
            # Actualizacion de los pesos y sesgos
            for capa in reversed(range(nro_capas_ocultas+1)): # Para cada capa
                if np.shape(deltas[capa]) == (1,1):
                     W[capa] +=  eta * deltas[capa] * entradas[capa]
                else:
                    W[capa] +=  eta * np.dot(deltas[capa], entradas[capa])
                
                b[capa] += eta * error
        
        
        
        # Si el error de la iteración actual es mayor al de la iteración anterior   
         
        if (i > 2) and (errores[i] > errores[i-1]):
            break    
    
    
    
    dom = entrenamiento[:,:-1]
    rango = entrenamiento[:,-1]  
    #print(entrenamiento)
    count = 0
    
    ceros = 0
    unos = 0
    
    for salida in salidas:
        if salida >=0.5:
            unos += 1
        else:
            ceros += 1
            
    print("ceros: ",ceros,", unos: ",unos)
    '''
    #  print(errores)
    for k in dom:
        print("aja",salidas[count][0][0])
        if salidas[count][0][0] > 0.5:
        
            plt.plot(k[0],k[1],'*r')
        else:
            plt.plot(k[0],k[1],'*g')
        count += 1'''
                        
    print("hola")
    return dom,salidas,w,b
    #print(y)
    

    
