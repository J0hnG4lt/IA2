import pandas as pd
from sklearn import cluster
from sklearn import preprocessing
from extractFeatures import readLanguages,getFeatureVectors
import json
from similarityMatrix import calculateCondProbMatrix
from showClusters import showClusters
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

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
	nroClusters = int(input("Number of clusters to use: "))


	print("Building Feature Matrix")
	condMatrix = calculateCondProbMatrix(data,pruneLanguages = nroLenguajes)


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

	distMatrix, mapping = convertToMatrix(distDicc)

	# clustering
	print("Applying cluster analysis algorithm")
	modelo = cluster.DBSCAN(eps=0.5,min_samples=2,metric="precomputed")
	modelo.fit(distMatrix)

	# Cluster names
	labels = modelo.labels_

	# as dataframes 
	print("Saving clusters to corridas/ProbabilidadCondicional/clusterValues.txt")
	results = pd.DataFrame()
	results["label"] = labels
	results["modelo"] = 1
	for i in range(len(condMatrix)):
		results.loc[i,('modelo')] = mapping[i]
	results.to_csv("corridas/ProbabilidadCondicional/clusterValues.txt", 
		           sep=',',
		           encoding='utf-8',
		           columns=["modelo","label"])

	print("Showing clusters")
	showClusters("corridas/ProbabilidadCondicional/clusterValues.txt")

