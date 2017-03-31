import pandas as pd
from sklearn import cluster
from sklearn import preprocessing
from extractFeatures import readLanguages,getFeatureVectors
import json
from similarityMatrix import calculateCondProbMatrix

from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt


print("Reading JSON file with repo and language info")
data = readLanguages("DATASET_FINAL/languagesUsersGithub.json")
print("Building Feature Matrix")
condMatrix = calculateCondProbMatrix(data,pruneLanguages = 40)

#featureMatrix = getFeatureVectors(data)

print("Saving featureMatrix to instancesConditionalProbability.txt")
ff = open("instancesConditionalProbability.txt","w")
ff.write(json.dumps(condMatrix,indent=4))
ff.close()

dataset = pd.DataFrame.from_dict(condMatrix,orient="index")

# k-means
print("Applying cluster analysis algorithm")
k_means = cluster.AgglomerativeClustering(n_clusters=4,affinity="precomputed",linkage="average")
#k_means = cluster.DBSCAN(eps=0.5,min_samples=3)
k_means.fit(dataset.as_matrix())

# Cluster names
labels = k_means.labels_

# as dataframes 
print("Saving clusters to clustersConditionalProbability.txt")
results = pd.DataFrame([dataset.index,labels]).T
results.to_csv("clustersConditionalProbability.txt", sep=',',encoding='utf-8')


#Dendograma
dendogramData = linkage(dataset.as_matrix())
language = [lang for lang in condMatrix]
dendrogram(dendogramData,labels=language)
plt.show()  