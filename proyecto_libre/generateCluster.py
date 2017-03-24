import pandas as pd
from sklearn import cluster
from sklearn import preprocessing
from extractFeatures import readLanguages,getFeatureVectors
import json

print("Reading JSON file with repo and language info")
data = readLanguages("languagesUsersGithub.json")
print("Building Feature Matrix")
featureMatrix = getFeatureVectors(data)

print("Saving featureMatrix to featureVectors.csv for Weka")
ofile = open("featureVectors.csv","w")
ofile.write(",".join(["lang_column"]+list(featureMatrix.keys())))
for lang in featureMatrix.keys():
    ofile.write("\n")
    ofile.write(",".join([lang]+list(map(str,featureMatrix[lang].values()))))
ofile.close()

print("Saving featureMatrix to instances.txt")
ff = open("instances.txt","w")
ff.write(json.dumps(featureMatrix,indent=4))
ff.close()

langs = [lang for lang in featureMatrix]
dataset = preprocessing.normalize(pd.DataFrame.from_dict(featureMatrix,orient="index"),axis=0)
dataset = {lang:dataset[i] for i,lang in enumerate(langs)}
dataset = pd.DataFrame.from_dict(dataset,orient="index")

# k-means
print("Applying cluster analysis algorithm")
k_means = cluster.AgglomerativeClustering(n_clusters=60,affinity="cosine",linkage="average")
k_means.fit(dataset.as_matrix())

# Cluster names
labels = k_means.labels_

# as dataframes
print("Saving clusters to clusters.txt")
results = pd.DataFrame([dataset.index,labels]).T
results.to_csv("clusters.txt", sep=',',encoding='utf-8')

