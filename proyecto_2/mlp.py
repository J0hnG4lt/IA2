#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import uniform
import numpy as np
import matplotlib.pyplot as plt

''' 
#   MLP : Perceptrón multicapa
#
#	@nroCapas : Numero de capas en el perceptron.
#   @funcionPorCapa: Funcion de cada capa.
#   @derivadaFuncionPorCapa: Derivada de las funciones de cada capa.
#	@nroNeuronasPorCapa: Numero de neuronas de cada capa.
#	@data: Data de entrenamiento.
#	@datasetValidacion: Data de validacion.
#   @maxIter: La cantidad máxima de iteraciones.
#   @aprendizaje:  tasa de aprendizaje.
#   @momentum: Constante de momentum.
'''

def MLP(nroCapas = 1,
		funcionPorCapa = None,
		derivadaFuncionPorCapa = None,
		nroNeuronasPorCapa = [1],
		data = None,
		datasetValidacion = None,
		maxIter = 1000,
		aprendizaje = 0.1,
		momentum = 0):
	
	# Chequeo de que todos los datos esten bien
	flag = False	
	if (data is None):
		print("Se necesita data de entrenamiento")
		flag = True
	if (len(nroNeuronasPorCapa) != nroCapas):
		print("El largo de nroNeuronasPorCapa debe ser igual al numero de capas")
		flag = True
	if (funcionPorCapa is None):
		funcionPorCapa = [lambda x: x for i in range(nroCapas)]
	if (len(funcionPorCapa) != nroCapas):
		flag = True
		print("El largo de las funciones por capa debe ser igual al numero de capas")
	if (len(derivadaFuncionPorCapa) != nroCapas):
		flag = True
		print("El largo de las derivadas de funciones por capa debe ser igual al numero de capas")

	if (flag):
		print("Hubo errores, se retornara 0")
		return 0


	nroAtributos = len(data[0]) - 1

	totalDatos = len(data)

	# mezclamos la data
	data = np.random.permutation(data)

	totalDatosEntrenamiento = totalDatos
	totalDatosValidacion = len(datasetValidacion)
	dataEntrenamiento = data
	dataValidacion = datasetValidacion

	# Se definen las estructuras necesarias.
	mlp = []

	# Para el momentum, guarda el update anterior.
	previousUpdate = []
	print("Creando perceptron con " + str(nroCapas) + " capas.")

	# Salida, entrada y gradiente de las neuronas para facilitar backpropagation
	salidaNeuronas = [ [0 for i in range(nroNeuronasPorCapa[j])] for j in range(nroCapas)]
	entradaNeuronas = [ [0 for i in range(nroNeuronasPorCapa[j])] for j in range(nroCapas)]
	gradiente = [ [0 for i in range(nroNeuronasPorCapa[j])] for j in range(nroCapas)]


	# Bias de cada neurona
	bias = [ [uniform(-1,1) for i in range(nroNeuronasPorCapa[j])] for j in range(nroCapas)]

	# Para el momentum, guarda el update del bias anterior
	previousBiasUpdate = [ [0 for i in range(nroNeuronasPorCapa[j])] for j in range(nroCapas)]

	# Llenamos el perceptron y el array de momentum
	for i in range(nroCapas):
		print("Capa " + str(i+1) + " con " + str(nroNeuronasPorCapa[i]) + " neuronas")
		if (i == 0):
			mlp += [[[uniform(-1,1) for i in range(nroAtributos)]\
					for neurona in range(nroNeuronasPorCapa[i])]]
			previousUpdate += [[[0 for i in range(nroAtributos)]\
					for neurona in range(nroNeuronasPorCapa[i])]]
		else:
			mlp += [[[uniform(-1,1) for i in range(nroNeuronasPorCapa[i-1])]\
					for neurona in range(nroNeuronasPorCapa[i])]]
			previousUpdate += [[[0 for i in range(nroNeuronasPorCapa[i-1])]\
					for neurona in range(nroNeuronasPorCapa[i])]]



	# Inicializamos los valores.
	iteraciones = 0
	error = 0
	errorAnt = 10**20
	eps = 10**-5
	errorPorIteracion = []

	print("Entrenando... Puede tardar unos momentos")
	# Ciclo principal de entrenamiento
	while (iteraciones < maxIter and abs(error - errorAnt) > eps):
		iteraciones += 1
		
		if iteraciones % 50 == 0:
			print("Iteracion: ",iteraciones," error Actual: ",error)
		
		errorAnt = error
		error = 0
		ceros = 0
		unos = 0

		for indexEstimulo in range(totalDatosEntrenamiento):
			# Forward propagation
			estimulo = dataEntrenamiento[indexEstimulo][:-1]
			respuesta = dataEntrenamiento[indexEstimulo][-1]
			for capa in range(nroCapas):
				for neurona in range(nroNeuronasPorCapa[capa]):
					# Si es la capa de salida se calcula el error
					if (capa == 0):
						entradaNeuronas[capa][neurona] = np.dot(mlp[capa][neurona],estimulo)+\
												 bias[capa][neurona]
						salidaNeuronas[capa][neurona] = funcionPorCapa[capa](entradaNeuronas[capa][neurona])
					else:
						entradaNeuronas[capa][neurona] = np.dot(mlp[capa][neurona],salidaNeuronas[capa-1])+\
												 bias[capa][neurona]
						salidaNeuronas[capa][neurona] = funcionPorCapa[capa](entradaNeuronas[capa][neurona])
					if (capa == nroCapas-1):
						error += (respuesta - salidaNeuronas[capa][neurona])**2 

			# Back Propagation
			for i in range(nroCapas):
				capa = nroCapas - (i+1)
				for neurona in range(nroNeuronasPorCapa[capa]):
					if (capa == nroCapas-1):
						gradiente[capa][neurona] =  derivadaFuncionPorCapa[capa](entradaNeuronas[capa][neurona])*\
														(respuesta - salidaNeuronas[capa][neurona])
					else:
						sumaGradientes = sum(gradiente[capa+1][i]*mlp[capa+1][i][neurona]\
											 for i in range(nroNeuronasPorCapa[capa+1])) 
						gradiente[capa][neurona] =  derivadaFuncionPorCapa[capa](entradaNeuronas[capa][neurona])*\
														sumaGradientes
			# Actualizacion Pesos
			for capa in range(nroCapas):
				for neurona in range(nroNeuronasPorCapa[capa]):
					# Se actualiza cada peso con el gradiente*su entrada + el momentum.
					bias[capa][neurona] += aprendizaje * gradiente[capa][neurona] +\
											momentum*previousBiasUpdate[capa][neurona]
					previousBiasUpdate[capa][neurona] = aprendizaje * gradiente[capa][neurona] +\
														momentum*previousBiasUpdate[capa][neurona]
					for peso in range(len(mlp[capa][neurona])):	
						if capa == 0:
							mlp[capa][neurona][peso] += aprendizaje*gradiente[capa][neurona]*estimulo[peso] + \
														momentum*previousUpdate[capa][neurona][peso]
							previousUpdate[capa][neurona][peso] = aprendizaje*gradiente[capa][neurona]*estimulo[peso] + \
																	momentum*previousUpdate[capa][neurona][peso]
						else:
							mlp[capa][neurona][peso] += aprendizaje*gradiente[capa][neurona]*salidaNeuronas[capa-1][peso] +\
																	momentum*previousUpdate[capa][neurona][peso]
							previousUpdate[capa][neurona][peso] = aprendizaje*gradiente[capa][neurona]*salidaNeuronas[capa-1][peso] + \
																	momentum*previousUpdate[capa][neurona][peso]

		# Dividimos el error entre los datos de entrenamiento para 
		# tener el error medio.
		error = error/totalDatosEntrenamiento
		# Guardamos el error por iteracion		
		errorPorIteracion.append(error)

	
	resultados = []
	# Validacion
	for indexEstimulo in range(totalDatosValidacion):
		# Forward propagation
		estimulo = dataValidacion[indexEstimulo][:-1]
		respuesta = dataValidacion[indexEstimulo][-1]
		resultadoValidacion = dict()
		resultadoValidacion["punto"] = estimulo
		resultadoValidacion["respuestaCorrecta"] = respuesta
		resultadoValidacion["respuestaSalida"] = []
		resultadoValidacion["error"] = []
		for capa in range(nroCapas):
			for neurona in range(nroNeuronasPorCapa[capa]):
				error_neurona = 0
				if (capa == 0):
					entradaNeuronas[capa][neurona] = np.dot(mlp[capa][neurona],estimulo)+\
											 bias[capa][neurona]
					salidaNeuronas[capa][neurona] = funcionPorCapa[capa](entradaNeuronas[capa][neurona])
				else:
					entradaNeuronas[capa][neurona] = np.dot(mlp[capa][neurona],salidaNeuronas[capa-1])+\
											 bias[capa][neurona]
					salidaNeuronas[capa][neurona] = funcionPorCapa[capa](entradaNeuronas[capa][neurona])
				if (capa == nroCapas-1):
					error_neurona += (respuesta - salidaNeuronas[capa][neurona])**2 
					resultadoValidacion["respuestaSalida"].append(salidaNeuronas[capa][neurona])
					resultadoValidacion["error"].append(error_neurona)
		resultados.append(resultadoValidacion)


	return (resultados,errorPorIteracion)

	

		
