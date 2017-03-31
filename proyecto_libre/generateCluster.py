import pandas as pd
from sklearn import cluster
from sklearn import preprocessing
from extractFeatures import readLanguages,getFeatureVectors,getFeatureVectors_perRepo
import json
from sklearn import metrics
#from sklearn.metrics import pairwise_distances
#from sklearn import datasets




print("Reading JSON file with repo and language info")
data = readLanguages("DATASET_FINAL/languagesUsersGithub.json")
print("Building Feature Matrix")
featureMatrix = getFeatureVectors(data)


print("Saving featureMatrix to featureVectors.csv for Weka")
ofile = open("featureVectors.csv","w")
ofile.write(",".join(["lang_column"]+list(map(str,featureMatrix.keys()))))
for lang in featureMatrix.keys():
    ofile.write("\n")
    ofile.write(",".join([str(lang)]+list(map(str,featureMatrix[lang].values()))))
ofile.close()

#print("Saving featureMatrix to instances.txt")
#ff = open("instances.txt","w")
#ff.write(json.dumps(featureMatrix,indent=4))
#ff.close()

langs = [lang for lang in featureMatrix]
dataset = preprocessing.normalize(pd.DataFrame.from_dict(featureMatrix,orient="index"),axis=0)
dataset = {lang:dataset[i] for i,lang in enumerate(langs)}
dataset = pd.DataFrame.from_dict(dataset,orient="index")


# k-means
print("Applying cluster analysis algorithm")
model = cluster.AgglomerativeClustering(n_clusters=20,affinity="cosine",linkage="average")
datasetM = dataset.as_matrix()
model.fit(datasetM)

# Cluster names
labels = model.labels_

# Cluster Evaluation
clusterEvaluation = metrics.silhouette_score(datasetM, labels, metric='euclidean')
print("Cluster quality using silhouette_score: {}".format(clusterEvaluation))

# as dataframes
print("Saving clusters to clusters.txt")
results = pd.DataFrame([dataset.index,labels]).T
results.to_csv("corridas/DBSCAN/eps5minsamples3.txt", sep=',',encoding='utf-8')

