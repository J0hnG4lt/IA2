

dataDict = {}
keys = []

with open("data.txt","r") as f:
	lines = f.readlines()
	print(len(lines)-1)
	for i in lines[0].split("\t"):
		# Shaving off newlines
		i = i.replace("\n","").replace("\r","")
		keys  += [i]
		dataDict[i] = []
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


