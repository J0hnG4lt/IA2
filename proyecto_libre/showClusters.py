

clusterFile = open("clusters.txt","r")
clusters = clusterFile.readlines()


clusterN = 0
for line in clusters[1:] :
    lang_and_cluster = line.strip("\n\r").split(",")
    clusterN = max(clusterN,int(lang_and_cluster[2]))

lang_by_cluster = {i:[] for i in range(clusterN+1)}

for line in clusters[1:] :
    lang_and_cluster = line.strip("\n\r").split(",")
    lang_by_cluster[int(lang_and_cluster[2])].append(lang_and_cluster[1])

for lang in lang_by_cluster:
	print(lang, ", ".join(lang_by_cluster[lang]))
