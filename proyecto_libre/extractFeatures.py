import json
import itertools
from statistics import mean

def readLanguages(jsonFilename) :
    with open(jsonFilename,"r") as dataFile :
        data = json.load(dataFile)
        dataFile.close()
    return data

def getFeatureVectors(data) :
    
    langs = []
    numberRepos = 0
    for userAccount in data :
        for repo in data[userAccount] :
            numberRepos += 1
            for lang in data[userAccount][repo] :
                if lang not in langs :
                    langs.append(lang)
    print("There are {} repositories".format(numberRepos))
    featureMatrix_ = {lang:{lang2:[0.0] \
                        for lang2 in langs} \
                            for lang in langs}
    
    for userAccount in data :
        for repo in data[userAccount] :
            for pair in itertools.combinations_with_replacement(set(data[userAccount][repo].keys()),2):
                featureMatrix_[pair[0]][pair[1]].append(1.0) #data[userAccount][repo][pair[1]])
                featureMatrix_[pair[1]][pair[0]].append(1.0) #data[userAccount][repo][pair[0]])
    
    featureMatrix = {lang:{lang2:sum(featureMatrix_[lang][lang2]) \
                        for lang2 in langs} \
                            for lang in langs}
    
    
    return featureMatrix


def getFeatureVectors_perRepo(data) :
    
    langs = []
    numberOfRepos = 0
    for userAccount in data :
        for repo in data[userAccount] :
            numberOfRepos += 1
            for lang in data[userAccount][repo] :
                if lang not in langs :
                    langs.append(lang)
    
    featureMatrix_ = {i:{lang:0.0 \
                        for lang in langs} \
                            for i in range(numberOfRepos)}
    
    for userAccount in data :
        for i,repo in enumerate(data[userAccount]) :
            for lang in data[userAccount][repo] :
                featureMatrix_[i][lang] = data[userAccount][repo][lang] #data[userAccount][repo][pair[0]])
    
    
    return featureMatrix_


def frecuenciaLenguajes(data) :
    langs = {}
    numberOfRepos = 0
    for userAccount in data :
        for repo in data[userAccount] :
            numberOfRepos += 1
            for lang in data[userAccount][repo] :
                if lang in langs :
                    langs[lang] += 1
                else :
                    langs[lang] = 1
    
    return [langs,numberOfRepos]