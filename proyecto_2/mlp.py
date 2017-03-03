

from random import uniform
import numpy as np

def MLP(nroCapas = 1,
		funcionPorCapa = None,
		derivadaFuncionPorCapa = None,
		nroNeuronasPorCapa = [1],
		data = None,
		porcentajeValidacion = None,
		maxIter = 1000,
		aprendizaje = 0.01):

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
	if (porcentajeValidacion is None):
		porcentajeValidacion = 20

	if (flag):
		print("Hubo errores, se retornara 0")
		return 0


	nroAtributos = len(data[0]) - 1
	
	totalDatos = len(data)
    totalDatosValidacion = len(data)*porcentajeValidacion//100
    totalDatosEntrenamiento = totalDatos - totalDatosValidacion

    data = np.random.permutation(data)

    dataEntrenamiento = data[0:totalDatosEntrenamiento]
    dataValidacion = data[totalDatosEntrenamiento:]

	mlp = []

	print("Creando perceptron con " + str(nroCapas) + " capas.")

	salidaNeuronas = [ [0 for i in range(nroNeuronasPorCapa[j])] for j in range(nroCapas)]
	bias = [ [uniform(-0.3,0.3) for i in range(nroNeuronasPorCapa[j])] for j in range(nroCapas)]
	gradiente = [ [0 for i in range(nroNeuronasPorCapa[j])] for j in range(nroCapas)]

	for i in range(nroCapas):
		print("Capa " + (i+1) + " con " + nroNeuronasPorCapa[i] + " neuronas")
		if (i == 0):
			mlp += [[[uniform(-0.3,0.3) for entrada in range(nroAtributos)]\
					for neurona in range(nroNeuronasPorCapa[i])]]
		else:
			mlp += [[[uniform(-0.3,0.3) for entrada in range(nroAtributos[i-1])]\
					for neurona in range(nroNeuronasPorCapa[i])]]




	iteraciones = 0
	error = 1000
	errorAnt = 0
	eps = 10**-5

	while (iteraciones < maxIter and abs(error - errorAnt) < eps):
		iteraciones += 1
		errorAnt = error
		error = 0

		for estimulo,respuesta in zip(nroAtributos[:-1],data[-1:]):
			# Forward propagation
			for capa in range(nroCapas):
				for neurona in range(nroNeuronasPorCapa[capa]):
					if (capa == 0):
						salida[capa][neurona] = funcionPorCapa[capa](\
												np.dot(mlp[capa][neurona],estimulo)+\
												bias[capa][neurona])
					else:
						salida[capa][neurona] = funcionPorCapa[capa](\
												np.dot(mlp[capa][neurona],salida[capa-1])+\
												bias[capa][neurona])
			# Back Propagation
			for i in range(nroCapas):
				capa = nroCapas - (i+1)
				for neurona in range(nroNeuronasPorCapa[capa]):
					