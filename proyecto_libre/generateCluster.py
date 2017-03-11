import pandas as pd
from sklearn import cluster
from similarityMatrix import readLanguages,calculateCondProbMatrix

print("Reading JSON file with repo and language info")
data = readLanguages("languagesUsersGithub.json")
print("Building Similarity Matrix")
condProbMatrix = calculateCondProbMatrix(data)

for lang1,row in condProbMatrix.items() :
    for lang2,condProb in row.items() :
        if condProb > 1.0 :
            print("Warning: CondProb="+str(condProb)+" btwn "+lang1+" and "+lang2+" bigger than 1.")

dataset = pd.DataFrame.from_dict(condProbMatrix)


# Similarity to Distance matrix
dataset = 1.0-dataset
print("Saving distance matrix to PrecomputedDistanceMatrix.txt")
dataset.to_csv("PrecomputedDistanceMatrix.txt", sep=',',encoding='utf-8')

# k-means
print("Applying cluster analysis algorithm")
k_means = cluster.KMeans(n_clusters=10,precompute_distances=False)
k_means.fit(dataset.as_matrix())

# Cluster names
labels = k_means.labels_

# as dataframes
print("Saving clusters to clusters.txt")
results = pd.DataFrame([dataset.index,labels]).T
results.to_csv("clusters.txt", sep=',',encoding='utf-8')

