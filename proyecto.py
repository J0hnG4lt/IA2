#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt


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
                              max_iter=100,
                              coeficiente_aprendizaje = 0.000000001,
                              valor_inicial = 0.1) :
    
    # Se agrega una columna de 1.0 para x_0
    if dom.ndim == 1:
        dom = np.array([[1.0,instancia] for instancia in dom])
    else :
        dom = np.insert(dom, 0, values=1.0, axis=1)
    
    numero_instancias = len(rango)
    numero_atributos = dom.shape[1]
    
    # Se crea el vector de coeficientes usando el valor inicial dado
    coeficientes = np.array([valor_inicial for i in range(numero_atributos)]) # Se generan los valores iniciales para los pesos
    inverso_num_atributos = (1.0/numero_atributos)
    
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
    
    coef_anteriores = np.copy(coeficientes)     # Vector de coeficientes 
                                                # antes de la iteracion actual
    coeficientes_por_iteracion = []
    constante = (coeficiente_aprendizaje * inverso_num_atributos)
    iteraciones = 0
    errorPorIteracion = []
    
    for i in range(numero_instancias) :
        
        # Se obtiene el vector de error
        error = h(coef_anteriores,dom[i,:]) - rango
        
        # Se actualiza cada coeficiente con el vector de error
        for j in range(numero_atributos) : 
            derivada = np.dot(error,dom[:,j])
            coeficientes[j] = coef_anteriores[j] - constante * derivada
        
        coef_anteriores = np.copy(coeficientes)
        
        # Seran utiles al graficar
        coeficientes_por_iteracion.insert(iteraciones, coef_anteriores)
        errorPorIteracion.insert(iteraciones,error_n(coeficientes,dom,rango))
        iteraciones += 1
        
        # Condicion de parada si hay muchas instancias
        if iteraciones > max_iter :
            return (coeficientes_por_iteracion,iteraciones,errorPorIteracion)
    
    return (coeficientes_por_iteracion,iteraciones,errorPorIteracion)


################################################################################
#                                Ejecucion
################################################################################


if __name__ == '__main__':
    
    # Se graficaran los resultados de la regresion lineal multiple con los datasets
    # de ejemplo. Notar que se normalizaron y no se quitaron los outliers
    
    # Se abre el archivo de prueba limpiado
    with open("x01_copia.txt", "r") as f : 
        datos_x01 = np.loadtxt( fname=f, 
                                dtype=float ,
                                comments="#" ,
                                delimiter=",",
                                usecols=(1,2))
        f.close()
    
    datos_x01 = datos_x01 / datos_x01.max(axis=0)   # Normalizacion
    
    dominio = datos_x01[:,0]
    rango = datos_x01[:,1]
    
    aprendizaje = 0.01
    inicial = 0.1
    coeficientes,iteraciones,errorPorIteracion = regresion_lineal_multiple( dom=dominio, 
                                                                            rango=rango,
                                                                            coeficiente_aprendizaje=aprendizaje,
                                                                            valor_inicial=inicial)

    errorPorIteracion = np.array(errorPorIteracion)
    mejor_iter = np.where(errorPorIteracion == errorPorIteracion.min())     # Mejor Iteracion
    print("Mejor Iteracion: ", mejor_iter[0][0])
    print(coeficientes[mejor_iter[0][0]], iteraciones)

    def linea(x,coef):
        return np.array([coef[0] + elem*coef[1] for elem in x])
    
    x = np.linspace(0,1,10000)
    
    plt.subplot(1, 3, 1)
    plt.scatter(dominio, rango)
    plt.title("Scatterplot (normalizado)")
    plt.xlabel("Peso Cerebral")
    plt.ylabel("Peso Corporal")
    plt.text(0.1,1.1, 'Aprendizaje = {:f}'.format(aprendizaje))
    plt.text(0.1,1.0, 'Inicial = {:f}'.format(inicial))
    plt.plot(x,linea(x,coeficientes[mejor_iter[0][0]]))
    
    plt.subplot(1, 3, 2)
    plt.plot(range(iteraciones), errorPorIteracion)
    plt.title("Convergencia -Peso- (normalizado)")
    plt.xlabel("Numero de Iteraciones")
    plt.ylabel("Error")
    plt.text(3,0.023, 'Aprendizaje = {:f}'.format(aprendizaje))
    plt.text(3,0.022, 'Inicial = {:f}'.format(inicial))
    
    
    # Se abre el archivo de prueba limpiado
    with open("x08_copia.txt", "r") as f:
        datos_x08 = np.loadtxt( fname=f, 
                                dtype=float ,
                                comments="#" ,
                                delimiter=",",
                                usecols=(1,2,3,4))
        f.close()
    
    
    datos_x08 = datos_x08 / datos_x08.max(axis=0)   # Normalizacion
    
    dominio2 = datos_x08[:,0:3]
    rango2 = datos_x08[:,3]
    aprendizaje2 = 0.01
    inicial2 = 1.0
    coeficientes2,iteraciones2,errorPorIteracion2 = regresion_lineal_multiple( dom=dominio2, 
                                                                            rango=rango2,
                                                                            coeficiente_aprendizaje=aprendizaje2,
                                                                            valor_inicial=inicial2)
    errorPorIteracion2 = np.array(errorPorIteracion2)
    mejor_iter2 = np.where(errorPorIteracion2 == errorPorIteracion2.min())      # Mejor Iteracion
    print("Mejor Iteracion 2", mejor_iter2[0][0])
    print(coeficientes2[mejor_iter2[0][0]], iteraciones2)
    
    plt.subplot(1, 3, 3)
    plt.plot(range(iteraciones2), errorPorIteracion2)
    plt.title("Convergencia -Homicidios- (normalizado)")
    plt.xlabel("Numero de Iteraciones")
    plt.ylabel("Error")
    plt.text(5,3.7, 'Aprendizaje = {:f}'.format(aprendizaje2))
    plt.text(5,3.5, 'Inicial = {:f}'.format(inicial2))
    
    plt.show()

