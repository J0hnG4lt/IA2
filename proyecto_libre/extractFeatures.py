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
    for userAccount in data :
        for repo in data[userAccount] :
            for lang in data[userAccount][repo] :
                if lang not in langs :
                    langs.append(lang)
    
    featureMatrix_ = {lang:{lang2:[0.0] \
                        for lang2 in langs} \
                            for lang in langs}
    
    for userAccount in data :
        for repo in data[userAccount] :
            for pair in itertools.combinations_with_replacement(set(data[userAccount][repo].keys()),2):
                featureMatrix_[pair[0]][pair[1]].append(data[userAccount][repo][pair[1]])
                featureMatrix_[pair[1]][pair[0]].append(data[userAccount][repo][pair[0]])
    
    featureMatrix = {lang:{lang2:mean(featureMatrix_[lang][lang2]) \
                        for lang2 in langs} \
                            for lang in langs}
    
    return featureMatrix


