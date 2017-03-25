import json
import itertools

def readLanguages(jsonFilename) :
    with open(jsonFilename,"r") as dataFile :
        data = json.load(dataFile)
        dataFile.close()
    return data

def calculateCondProbMatrix(data) :
    
    langAllBytes = dict()
    for userAccount in data :
        for repo in data[userAccount] :
            
            # Total amount of bytes per language
            for lang in data[userAccount][repo] :
                if lang in langAllBytes :
                    langAllBytes[lang] += float(data[userAccount][repo][lang])
                else :
                    langAllBytes[lang] = float(data[userAccount][repo][lang])
            
    totalBytes = sum(langAllBytes.values())
    probAllLangs = {lang:(langAllBytes[lang]/totalBytes) for lang in langAllBytes}
    
    condProbMatrix = {lang:{lang2:0.0 \
                        for lang2 in langAllBytes} \
                            for lang in langAllBytes}
    
    for userAccount in data :
        for repo in data[userAccount] :
            
            if len(data[userAccount][repo]) < 2 :
                continue
            
            # Total amount of bytes for each pair of languages
            for pair in itertools.combinations(data[userAccount][repo].keys(),2):
                intersection = float(data[userAccount][repo][pair[0]]) + float(data[userAccount][repo][pair[1]])
                condProbMatrix[pair[0]][pair[1]] += intersection
            
    # Normalize with second language
    for lang in condProbMatrix :
        for lang2 in condProbMatrix[lang] :
            if lang == lang2 :
                condProbMatrix[lang][lang2] = 1.0
            else :
                condProbMatrix[lang][lang2] /= totalBytes
                condProbMatrix[lang][lang2] /= probAllLangs[lang2]
    
    return condProbMatrix