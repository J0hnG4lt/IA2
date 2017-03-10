import json
import itertools
import pandas as pd

def readLanguages(jsonFilename) :
    with open(jsonFilename,"r") as dataFile :
        data = json.load(dataFile)
        dataFile.close()
    return data

def calculateCondProbMatrix(data) :
    condProbMatrix = dict()
    langAllBytes = dict()
    for userAccount in data :
        for repo in data[userAccount] :
            
            # Total amount of bytes per language
            for lang in data[userAccount][repo] :
                if lang in langAllBytes :
                    langAllBytes[lang] += float(data[userAccount][repo][lang])
                else :
                    langAllBytes[lang] = float(data[userAccount][repo][lang])
            
            if len(data[userAccount][repo]) < 2 :
                continue
            
            # Total amount of bytes for each pair of languages
            for pair in itertools.combinations(data[userAccount][repo].keys(),2):
                intersection = float(data[userAccount][repo][pair[0]]) + float(data[userAccount][repo][pair[1]])
                if pair[0] in condProbMatrix :
                    if pair[1] in condProbMatrix[pair[0]] :
                        condProbMatrix[pair[0]][pair[1]] += intersection
                    else :
                        condProbMatrix[pair[0]][pair[1]] = intersection
                else :
                    condProbMatrix[pair[0]] = dict()
                    condProbMatrix[pair[0]][pair[1]] = intersection
            
    # Normalize with second language
    for lang in condProbMatrix :
        for lang2 in condProbMatrix[lang] :
            if (langAllBytes[lang]+langAllBytes[lang2]) < condProbMatrix[lang][lang2] :
                print(langAllBytes[lang],langAllBytes[lang2],condProbMatrix[lang][lang2])
            condProbMatrix[lang][lang2] /= (langAllBytes[lang]+langAllBytes[lang2])
    
    return condProbMatrix


data = readLanguages("languagesUsersGithub.json")
condProbMatrix = calculateCondProbMatrix(data)

#d = pd.DataFrame.from_dict(condProbMatrix)
#print(d)