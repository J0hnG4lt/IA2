''' Parsing the input '''

dataDict = {}
keys = []

with open("data.txt","r") as f:
	lines = f.readlines()
	print(len(lines)-1)
	for i in lines[0].split("\t"):
		# Shaving off newlines
		i = i.replace("\n","").replace("\r","")
		keys  += [i.lower()]
		dataDict[i.lower()] = []
	for line in lines[1:]:
		counter = 0
		for word in line.split("\t"):
			# Shaving off newlines
			word = word.replace("\n","").replace("\r","")
			dataDict[keys[counter]].append(word)
			counter += 1


for key in keys:
	# Prints missing data
	print(key,sum(1 for i in dataDict[key] if i == ""))


''' Applying custom values from a file '''


with open("customValues.txt","r") as f:
	lines = f.readlines()
	turnToNumbers = False
	for line in lines:
		proccessedLine = line.replace("\r","").replace("\n","").split("\t")
		word = proccessedLine[0]
		if word == "#NUMERIC":
			turnToNumbers = True
			continue
		elif word == "#VALUE":
			key = proccessedLine[1].lower()
			turnToNumbers = False
			continue
		try:
			if turnToNumbers:
				for i in range(len(dataDict[word])):
					if dataDict[word][i] != "":
						dataDict[word][i] = float(dataDict[word][i])
				print("Turned all " + word + " to numbers")
			else:
				value = float(proccessedLine[1])
				for i in range(len(dataDict[key])):
					if dataDict[key][i] == word:
						dataDict[key][i] = value
				print("Turned all " + word + " to " + str(value))
		except:
			if turnToNumbers:
				print("Couldn't Turn " + proccessedLine + " into numbers")
			else:
				print("Couldn't swap " + proccessedLine[0] + " for " + proccessedLine[1])
				

import numpy as np
import csv


features = dataDict.keys()
row = []
data = []

archivoSalida = open("data.csv",'w')
wr  = csv.writer(archivoSalida, delimiter=",",lineterminator='\n')

features = list(features)

for i in range(len(features)):
    if features[i] == "saleprice":
        features[i] = features[len(features) - 1]
        features[len(features) - 1] = "saleprice"
        break

features_seleccionados = ["overall qual", "overall cond", "year built", 
                          "year remod/add", "exter qual", "foundation", 
                          "bsmt qual", "bsmt exposure", "bsmtfin type 1", 
                          "bsmtfin sf 1", "total bsmt sf", "heating qc", 
                          "central air", "1st flr sf", "gr liv area", 
                          "garage yr blt", "garage finish", "garage cars", 
                          "garage area", "saleprice"]

wr.writerow(features_seleccionados)

for i in range(len(dataDict["foundation"])):
    row = []
    for feature in features_seleccionados:
        row.append(dataDict[feature][i])
    wr.writerow(row)

archivoSalida.close()

