import json
import itertools

def readLanguages(jsonFilename) :
    with open(jsonFilename,"r") as dataFile :
        data = json.load(dataFile)
        dataFile.close()
    return data

def calculateCondProbMatrix(data) :
    
    langAllUses = dict()
    langUsers = dict()
    amountOfUsers = len(data)
    for userAccount in data:
        for repo in data[userAccount] :
            # we check each language that was used.
            for lang in data[userAccount][repo]:
                if lang in langAllUses:
                    langAllUses[lang] += 1
                else :
                    langAllUses[lang] = 1
                langUsers[userAccount][lang] = 1


            
    totalBytes = sum(langAllUses.values())
    probAllLangs = {lang:(langAllUses[lang]/amountOfUsers) for lang in langAllUses}
    
    usedTogether = {lang:lang2:0 \
                        for lang2 in langAllUses} \
                            for lang in langAllUses}


    # Abnormally long if and for chain 
    #  to check amount of times two languages
    #  are used together by an user. 
    for user in langUsers:
        for lang in langAllUses:
            if lang in langUsers[user]:
                for lang2 in langAllUses:
                    if lang2 in langUsers[user]:
                        usedTogether[lang][lang2] += 1 

    condProbMatrix = {lang:{lang2:0.0 \
                        for lang2 in langAllUses} \
                            for lang in langAllUses}
    
    for lang in langAllUses:
        uses = langAllUses[lang] # The uses of this language
        for lang2 in langAllUses:
            # conditional probability of lang given lang2
            condProbMatrix[lang][lang2] = 0 if uses == 0 \
                                            else (usedTogether[lang][lang2]/uses)



    return condProbMatrix