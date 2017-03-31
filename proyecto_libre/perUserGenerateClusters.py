import pandas as pd
from sklearn import cluster
from sklearn import preprocessing
from extractFeatures import readLanguages,getFeatureVectors
import json
from similarityMatrix import calculateCondProbMatrix
from showClusters import showClusters
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt



if __name__ == '__main__':
	print("Reading JSON file with repo and language info")
	data = readLanguages("DATASET_FINAL/languagesUsersGithub.json")

	nroLenguajes = int(input("Number of languages to use: "))
	nroClusters = int(input("Number of clusters to use: "))


	print("Building Feature Matrix")
	condMatrix = calculateCondProbMatrix(data,pruneLanguages = nroLenguajes)

	dataset = pd.DataFrame.from_dict(condMatrix,orient="index")

	# clustering
	print("Applying cluster analysis algorithm")
	modelo = cluster.AgglomerativeClustering(n_clusters=nroClusters,affinity="precomputed",linkage="average")
	#k_means = cluster.DBSCAN(eps=0.5,min_samples=3)
	modelo.fit(dataset.as_matrix())

	# Cluster names
	labels = modelo.labels_

	# as dataframes 
	print("Saving clusters to clustersConditionalProbability.txt")
	results = pd.DataFrame([dataset.index,labels]).T
	results.to_csv("clustersConditionalProbability.txt", sep=',',encoding='utf-8')

	print("Showing clusters")
	showClusters("clustersConditionalProbability.txt")

