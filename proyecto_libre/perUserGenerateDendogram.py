import pandas as pd
from sklearn import cluster
from sklearn import preprocessing
from extractFeatures import readLanguages,getFeatureVectors
import json
from similarityMatrix import calculateCondProbMatrix
from showClusters import showClusters
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

import scipy.spatial.distance as ssd

def convertToMatrix(data):
	M = [[0 for i in range(len(data))] for j in range(len(data))]
	mapping = {}
	count = 0
	for lang in data.keys():
		mapping[count] = lang
		count+=1
	for i in range(count):
		for j in range(count):
			M[i][j] = data[mapping[i]][mapping[j]]

	return M,mapping

if __name__ == '__main__':
	print("Reading JSON file with repo and language info")
	data = readLanguages("DATASET_FINAL/languagesUsersGithub.json")

	nroLenguajes = int(input("Number of languages to use: "))

	print("Building Feature Matrix")
	condMatrix = calculateCondProbMatrix(data,pruneLanguages = nroLenguajes)

	print("Saving conditional probability matrix")
	dataset1 = pd.DataFrame.from_dict(condMatrix,orient="index")

	dataset1.to_csv("corridas/ProbabilidadCondicional/MatrizProbabilidadCondicional_" + str(nroLenguajes)+".csv",
						float_format="%.2f")



	#Inverting the matrix probabilities 
	#  to have a distance matrix instead of similarity
	for lang in condMatrix:
		for lang2 in condMatrix[lang]:
	 		condMatrix[lang][lang2] = 1 - condMatrix[lang][lang2]

	# Making the distance matrix symmetric using max "distance"
	distDicc = {lang:{} for lang in condMatrix}
	for lang in condMatrix:
		for lang2 in condMatrix[lang]:
			distDicc[lang][lang2] = max(condMatrix[lang][lang2],
											condMatrix[lang2][lang])


	dataset = pd.DataFrame.from_dict(distDicc,orient="index")

	dataset.to_csv("corridas/ProbabilidadCondicional/MatrizDistProbabilidadCondicional.csv",
						float_format="%.2f")

	distMatrix, mapping = convertToMatrix(distDicc)
	# convert the n*n square matrix form into a condensed nC2 array
	# distArray[{n choose 2}-{n-i choose 2} + (j-i-1)] is the distance between points i and j
	distArray = ssd.squareform(distMatrix) 

	# clustering
	print("Applying cluster analysis algorithm")
	#Dendograma
	dendogramData = linkage(distArray,method="single")
	language = [mapping[i] for i in range(len(condMatrix))]
	dendrogram(dendogramData,
			labels=language,
			leaf_rotation = 90,
			leaf_font_size=10)
	plt.gcf().subplots_adjust(bottom=0.20)

	print ("Dendograma guardado en corridas/ProbabilidadCondicional/dendograma_" + str(nroLenguajes))
	plt.savefig("corridas/ProbabilidadCondicional/dendograma_" + str(nroLenguajes))