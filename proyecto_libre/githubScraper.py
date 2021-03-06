#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
from auth import username, password
from requests.auth import HTTPBasicAuth
import requests
from collections import deque
from pathlib import Path
import sys
import getopt
import signal
import os.path



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
    """
    Obtiene los repositorios del usuario @user
    """
    response = requests.get('https://api.github.com/users/{0}/repos'.format(user), 
                            auth=HTTPBasicAuth(username, password)).text
    responseJson = json.loads(response)
    return responseJson


def getLanguage(user, repo):
    """
    Obtiene los lenguajes del repositorio @repo del usuario @user
    """
    response = requests.get('https://api.github.com/repos/{0}/{1}/languages'.format(user,repo), 
                            auth=HTTPBasicAuth(username, password)).text
    responseJson = json.loads(response)
    return responseJson

# Excepciones personalizadas
class RepoException(Exception): pass
class GithubUserException(Exception) : pass


def findUserLanguages(user):
    """
    Encuentra los lenguajes usados por @user en su
    cuenta de GitHub
    """
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
    """
    Va encontrando los usuarios que forman el grafo de followers desde
    el usuario @user en GitHub siguiendo un BFS hasta llegar a la 
    profundidad @depth
    """
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
    """
    Encuentra los usuarios con un spider a partir de
    @rootUserName con profundidad @depth y los guarda en @userListFileName
    """
    usersCrawled = githubUserSpider(rootUserName, depth)
    with open(userListFileName, "w") as userList:
        userList.writelines(map(lambda x : x+"\n",usersCrawled))
        userList.close()


def manejador_de_signals(signal, frame):
        print('\nCtrl+C has been pressed!')
        print("New users cannot be appended in this case")
        sys.exit(0)


if __name__ == '__main__':
    
    signal.signal(signal.SIGINT, manejador_de_signals)
    
    if (len(sys.argv) < 2) or (len(sys.argv) > 3) :
        print("Usage: ")
        print('githubScaper.py [ repos | users <usuario>]')
        sys.exit(2)
    elif sys.argv[1] == "repos" :
        getUsers = False
    elif sys.argv[1] == "users":
        if (len(sys.argv) != 3) :
            print("Usage: ")
            print('githubScaper.py [ repos | users <usuario>]')
            sys.exit(2)
        getUsers = True
    else :
        print("Uso correcto: ")
        print('githubScaper.py [ repos | users <usuario>]')
        sys.exit(2)
    
    if getUsers :
        
        # Encuentro los usuarios
        usuarioI = sys.argv[2]
        
        response = requests.get('https://api.github.com/users/{0}'.format(usuarioI), 
                                auth=HTTPBasicAuth(username, password)).text
        responseJson = json.loads(response)
        if "message" in responseJson :
            print(responseJson["message"])
            print("Usuario introducido no existe en GitHub")
            sys.exit(2)
        
        print("\nRecolectando Usernames: ")
        findUsers(usuarioI, 10, "usuariosGithub.txt")
        print("Ya se tienen suficientes usuarios guardados.")
        
    else :
        # Encuentro los lenguajes de los usuarios
        
        if not os.path.isfile("usuariosGithub.txt") :
            print("There are no users to get repos from")
            print("Execute first: \npython3 githubScraper.py users <user>")
            sys.exit(2)
        
        print("\nRecolectando Lenguajes: ")
        with open("usuariosGithub.txt", "r") as usersFile :
            # Si se encontrase un rate limit no se debe permitir que se pierda
            # el trabajo hecho
            lastUserFile = Path("lastUser.txt")
            lastUser = 0
            if lastUserFile.is_file():
                lastUserFile = open("lastUser.txt","r")
                lastUser = int(lastUserFile.readline())
                print("Retomando ultima corrida.")
            i = 0
            try :
                for user in usersFile :
                    if (lastUser != 0) and (i < (lastUser+1)) :
                        i += 1
                        continue
                    
                    print("Lenguajes: ",user.strip("\n\r"))
                    languages.update(findUserLanguages(user.strip("\n\r")))
                    i += 1
            except requests.exceptions.HTTPError as err:
                print(err)
            except RepoException :
                pass
            finally :
                lastUserFile = Path("lastUser.txt")
                if lastUserFile.is_file():
                    with open("languagesUsersGithub.json","a") as langsFile :
                        json.dump(languages, langsFile, indent=4)
                        langsFile.close()
                        print("Appending found repos for the next attempt.")
                else :
                    with open("languagesUsersGithub.json","w") as langsFile :
                        json.dump(languages, langsFile, indent=4)
                        langsFile.close()
                        print("Saving found repos")
                usersFile.close()
                ff = open("lastUser.txt","w")
                ff.write(str(i))
    
