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

	print("Building Feature Matrix")
	condMatrix = calculateCondProbMatrix(data,pruneLanguages = nroLenguajes)

	dataset = pd.DataFrame.from_dict(condMatrix,orient="index")

	# clustering
	print("Applying cluster analysis algorithm")
	#Dendograma
	dendogramData = linkage(dataset.as_matrix())
	language = [lang for lang in condMatrix]
	dendrogram(dendogramData,
			labels=language,
			leaf_rotation = 90,
			leaf_font_size=10)
	print ("Dendograma guardado en corridas/ProbabilidadCondicional/dendograma_" + str(nroLenguajes))
	plt.savefig("corridas/ProbabilidadCondicional/dendograma_" + str(nroLenguajes))