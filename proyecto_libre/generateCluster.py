import pandas as pd
from sklearn import cluster
from similarityMatrix import readLanguages,calculateCondProbMatrix

data = readLanguages("languagesUsersGithub.json")
condProbMatrix = calculateCondProbMatrix(data)



dataset = pd.DataFrame.from_dict(condProbMatrix)

# Similarity to Distance matrix
dataset = 1.0-dataset

# k-means
k_means = cluster.KMeans(n_clusters=10,precompute_distances=False)
k_means.fit(dataset.as_matrix())

# Cluster names
labels = k_means.labels_

# as dataframes
results = pd.DataFrame([dataset.index,labels]).T
results.to_csv("clusters.txt", sep=',',encoding='utf-8')

