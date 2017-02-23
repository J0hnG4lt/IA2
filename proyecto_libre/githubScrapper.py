#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
from auth import username, password
from requests.auth import HTTPBasicAuth
import requests

"""
#   USAR python3
#
#   Para correr la función de scrapping es necesario que registre un token
#   de acceso en un archivo llamado "auth". Dicho archivo sólo va a contener
#   las variables username y password. Estos son su nombre de usuario de Github
#   y su token de acceso generado en la configuración. Usar esto le permitirá
#   contar con un rate limit más generoso para hacer requests.
#
"""


languages = dict()

def getRepos(user):
    response = requests.get('https://api.github.com/users/{0}/repos'.format(user), 
                            auth=HTTPBasicAuth(username, password)).text
    responseJson = json.loads(response)
    return responseJson


def getLanguage(user, repo):
    response = requests.get('https://api.github.com/repos/{0}/{1}/languages'.format(user,repo), 
                            auth=HTTPBasicAuth(username, password)).text
    responseJson = json.loads(response)
    return responseJson



def findUserLanguages(user):
    languages[user] = []
    jsonReposResponse = getRepos(user)
    for repo in jsonReposResponse :
        jsonLangsResponse = getLanguage(user, repo["name"])
        languages[user] += list(jsonLangsResponse.keys())
    return languages


languages.update(findUserLanguages("J0hnG4lt"))
languages.update(findUserLanguages("leslierodrigues"))
languages.update(findUserLanguages("Imme2"))

print(languages)