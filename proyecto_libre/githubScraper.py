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
#   Hay que guardar la lista de usuarios en un archivo llamado "usuariosGithub.txt".
#   Dicho archivo debe tener un usuario por línea.
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
    # Comentar si se quiere repetir lenguajes
    languages[user] = list(set(languages[user]))
    return languages


with open("usuariosGithub.txt", "r") as usersFile :
    # Si se encontrase un rate limit no se debe permitir que se pierda
    # el trabajo hecho
    try :
        for user in usersFile :
            print(user.strip("\n\r"))
            languages.update(findUserLanguages(user.strip("\n\r")))
    except requests.exceptions.HTTPError as err:
        print(err)
    finally :
        usersFile.close()

with open("languagesUsersGithub.json","w") as langsFile :
    json.dump(languages, langsFile, indent=4)
    langsFile.close()
print(languages)