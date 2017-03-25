import pandas as pd
from extractFeatures import readLanguages
import json
from showClusters import *


def frecuenciaLenguajes():
    data = readLanguages("languagesUsersGithub.json")

    langs = dict()
    for userAccount in data :
        for repo in data[userAccount] :
            for lang in data[userAccount][repo] :
                if lang not in langs :
                	langs[lang] = 1 
                else:
                	langs[lang] += 1 
    return langs
    
def wordCloud():
    langs = frecuenciaLenguajes()
    clusters = showClusters()
    
    for cluster in range(len(clusters)):
        print("------- Cluster" + str(cluster) + "------- ")
        for lang in clusters[cluster]: 
            print(langs[lang],lang)

wordCloud()    

