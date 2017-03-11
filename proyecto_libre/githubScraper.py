#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
from auth import username, password
from requests.auth import HTTPBasicAuth
import requests
from collections import deque

"""
#   USAR python3
#
#   Para correr la función de scraping es necesario que registre un token
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

class RepoException(Exception): pass

class GithubUserException(Exception) : pass

def findUserLanguages(user):
    languages[user] = []
    jsonReposResponse = getRepos(user)
    if "message" in jsonReposResponse :
        print(jsonReposResponse["message"])
        print("Se ha dejado de recolectar repos")
        raise RepoException
    
    languages[user] = dict()
    for repo in jsonReposResponse :
        jsonLangsResponse = getLanguage(user, repo["name"])
        if "message" in jsonLangsResponse :
            print(jsonLangsResponse["message"])
            print("Se ha dejado de recolectar lenguajes")
            raise RepoException
        languages[user][repo["name"]] = jsonLangsResponse
    # Comentar si se quiere repetir lenguajes
    #print(languages)
    #languages[user] = list(set(languages[user]))
    #print(languages)
    return languages



def githubUserSpider(user, depth) :
    users = deque()
    users.append(user)
    newLevel = deque()
    done = deque()
    lengthUsers = 1
    lengthNewLevel = 0
    currentLevel = 0
    while (currentLevel < depth) :
        if lengthUsers == 0 :
            if lengthNewLevel == 0 :
                return done
            users = newLevel
            newLevel = deque()
            currentLevel += 1
            lengthUsers = lengthNewLevel
            lengthNewLevel = 0
        currentUser = users.popleft()
        print(currentUser)
        lengthUsers -= 1
        done.append(currentUser)
        response = requests.get('https://api.github.com/users/{0}/followers'.format(currentUser), 
                                auth=HTTPBasicAuth(username, password)).text
        responseJson = json.loads(response)
        if "message" in responseJson :
            print(responseJson["message"])
            print("Se ha dejado de recolectar usuarios")
            return done
        
        for follower in responseJson :
            if (users.count(follower["login"]) == 0) \
                and (done.count(follower["login"]) == 0) \
                and (newLevel.count(follower["login"]) == 0):
                newLevel.append(follower["login"])
                lengthNewLevel += 1
        
    
    return done

def findUsers(rootUserName, depth, userListFileName) :
    usersCrawled = githubUserSpider(rootUserName, depth)
    with open(userListFileName, "w") as userList:
        userList.writelines(map(lambda x : x+"\n",usersCrawled))
        userList.close()


if __name__ == '__main__':
    
    # Encuentro los usuarios
    
    print("\nRecolectando Usernames: ")
    findUsers("donnemartin", 10, "usuariosGithub.txt")
    
    # Encuentro los lenguajes
    
    print("\nRecolectando Lenguajes: ")
    with open("usuariosGithub.txt", "r") as usersFile :
        # Si se encontrase un rate limit no se debe permitir que se pierda
        # el trabajo hecho
        try :
            for user in usersFile :
                print("Lenguajes: ",user.strip("\n\r"))
                languages.update(findUserLanguages(user.strip("\n\r")))
        except requests.exceptions.HTTPError as err:
            print(err)
        except RepoException :
            pass
        finally :
            usersFile.close()
    
    with open("languagesUsersGithub.json","w") as langsFile :
        json.dump(languages, langsFile, indent=4)
        langsFile.close()
    
