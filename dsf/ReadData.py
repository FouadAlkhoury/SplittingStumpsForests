# The script loads data from dataset files.
import numpy as np
import csv
import operator
import sys
import os
from sklearn.datasets import load_svmlight_file

def readData(dataset, type, path):
	if dataset == 'satlog':
		return readDataSatlog(type, path)
	if dataset == 'magic':
		return readDataMagic(type, path)
	if dataset == 'spambase':
		return readDataSpambase(type, path)
	if dataset == 'letter':
		return readDataLetter(type, path)
	if dataset == 'bank':
		return readDataBank(type, path)
	if dataset == 'adult':
		return readDataAdult(type, path)
	if dataset == 'room':
		return readDataRoom(type, path)
	if dataset == 'credit':
		return readDataCredit(type, path)
	if dataset == 'drybean':
		return readDataDryBean(type, path)
	if dataset == 'rice':
		return readDataRice(type, path)
	if dataset == 'shopping':
		return readDataShopping(type, path)
	if dataset == 'aloi':
		return readDataAloi(type, path)
	if dataset == 'waveform':
		return readDataWaveform(type, path)

	# error
	raise Exception('Name of dataset unknown: ' + dataset)


    
def readDataSatlog(type, path="../data/"):
	X = []
	Y = []

	D = np.genfromtxt(os.path.join(path, "satlog", "sat_all.trn"), delimiter=' ')

	X = D[:,0:-1].astype(dtype=np.int32)
	Y = D[:,-1]
	Y = [y-1 if y != 7 else 5 for y in Y]

	Y = np.array(Y)
	X = np.array(X)    
	return np.array(X).astype(dtype=np.int32), np.array(Y).astype(dtype=np.int32)




def readDataMagic(type, path="../data/"):

	filename = os.path.join(path, "magic", "magic04.train")

	X = np.loadtxt(filename, delimiter=',',dtype = np.float64)
	Y = X[:, -1]
	X = X[:, :-1]

	return X, Y


def readDataRoom(type, path="../data/"):

	filename = os.path.join(path, "room", "room.train")

	X = np.loadtxt(filename, delimiter=',',dtype = np.float64)
	Y = X[:, -1]
	X = X[:, :-1]

	return X, Y


def readDataAloi(type, path="../data/"):
	filename = os.path.join(path, "aloi", "aloi.train")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float64)
	Y = X[:, 0]
	X = X[:, 1:-1]

	return X, Y

def readDataWaveform(type, path="../data/"):
	filename = os.path.join(path, "waveform", "waveform.train")

	X = np.loadtxt(filename, delimiter=',', dtype=np.float64)
	Y = X[:,-1]
	X = X[:, :-2]

	return X, Y

def readDataCredit(type, path="../data/"):

	filename = os.path.join(path, "credit", "credit.train")

	X = np.loadtxt(filename, delimiter=',',dtype = np.float64)
	Y = X[:, -1]
	X = X[:, 1:-1]

	return X, Y



def readDataSpambase(type, path="../data/"):


	filename = os.path.join(path, "spambase", "spambase.train")

	X = np.loadtxt(filename, delimiter=',',dtype = np.float64)
	Y = X[:, -1]
	X = X[:, :-1]

	return X, Y



def readDataLetter(type,path="../data/"):

	f = open(os.path.join(path, "letter", "letter-recognition.data"),'r')
	X = []
	Y = []
	for row in f:
		entries = row.strip().split(",")
		x = [int(e) for e in entries[1:]]
		# Labels are capital letter has. 'A' starts in ASCII code with 65
		# We map it to '0' here, since SKLearn internally starts with mapping = 0 
		# and I have no idea how it produces correct outputs in the first place
		if (type =='train'):
			y = ord(entries[0]) - 65
		if (type =='test'):
			y = int(entries[0])
            
		X.append(x)
		Y.append(y)

	return np.array(X).astype(dtype=np.int32), np.array(Y)


def readDataAdult(type, path="../data/"):
	
	X = []
	Y = []       

	f = open(os.path.join(path, "adult", "adult."+type), 'r')
	counter = 0    
	for row in f:
		if (len(row) > 1):
			entries = row.replace("\n", "").replace(" ", "").split(",")

			x = getFeatureVectorAdult(entries)

			if (entries[-1] == "<=50K." or entries[-1] == "<=50K"):
				counter += 1
				if (counter %3 == 0):
					y = 0
					Y.append(y)
					X.append(x) 
                
                
			else:
					y = 1
					Y.append(y)
					X.append(x) 
                    
	X = np.array(X).astype(dtype=np.int32)
	Y = np.array(Y)
	f.close()
	return X, Y


def readDataDryBean(type,path="../data/"):
	X = []
	Y = []
	filename = os.path.join(path, "drybean", "drybean.train")

	X = np.loadtxt(filename, delimiter=',',dtype = np.float64)
	Y = X[:, -1]
	X = X[:, :-1]

	return X, Y

def readDataRice(type, path="../data/"):
	X = []
	Y = []
	filename = os.path.join(path, "rice", "rice.train")

	X = np.loadtxt(filename, delimiter=',',dtype = np.float64)
	Y = X[:, -1]
	X = X[:, :-1]

	return X, Y


def readDataBank(type, path="../data/"):

	X = []
	Y = []

	f = open(os.path.join(path, "bank/bank.train"))

        
	next(f)
	for row in f:
		if (len(row) > 1):
			entries = row.replace("\n", "").replace(" ", "").replace("\"","").split(";")

			x = getFeatureVectorBank(entries)

			if (entries[-1] == "no"):
				y = 0
			else:
				y = 1

			Y.append(y)
			X.append(x)

	X = np.array(X).astype(dtype=np.int32)
	Y = np.array(Y)

	f.close()
	return (X, Y)


def readDataShopping(type,path="../data/"):
	X = []
	Y = []


	f = open(os.path.join(path, "shopping/shopping_all.train"))

	next(f)
	for row in f:
		if (len(row) > 1):
			entries = row.replace("\n", "").replace(" ", "").replace("\"", "").split(",")

			x = getFeatureVectorShopping(entries)

			if (entries[-1] == "FALSE"):
				y = 0
			else:
				y = 1

			Y.append(y)
			X.append(x)
	print(X[0])
	print(Y[0])
	print(np.shape(X))
	X = np.array(X)
	Y = np.array(Y)

	f.close()
	return (X, Y)




def getFeatureVectorShopping(entries):
	x = []
	for i in range(10):
		x.append(float(entries[i]))

	month = [0 for i in range(12)]
	if entries[10] == "Jan":
		month[0] = 1
	elif entries[10] == "Feb":
		month[1] = 1
	elif entries[10] == "Mar":
		month[2] = 1
	elif entries[10] == "Apr":
		month[3] = 1
	elif entries[10] == "May":
		month[4] = 1
	elif entries[10] == "Jun":
		month[5] = 1
	elif entries[10] == "Jul":
		month[6] = 1
	elif entries[10] == "Aug":
		month[7] = 1
	elif entries[10] == "Sep":
		month[8] = 1
	elif entries[10] == "Oct":
		month[9] = 1
	elif entries[10] == "Nov":
		month[10] = 1
	elif entries[10] == "Dec":
		month[11] = 1

	x.extend(month)
	x.append(float(entries[11]))
	x.append(float(entries[12]))
	x.append(float(entries[13]))
	x.append(float(entries[14]))

	visitor = [0 for i in range(2)]
	if entries[15] == "Returning_Visitor":
		visitor[0] = 1
	elif entries[15] == "New_Visitor":
		visitor[1] = 1

	x.extend(visitor)

	weekend = [0 for i in range(2)]
	if entries[16] == "FALSE":
		weekend[0] = 1
	elif entries[16] == "TRUE":
		weekend[1] = 1
	x.extend(weekend)

	return x



def getFeatureVectorAdult(entries):
	x = []
	x.append(float(entries[0])) # age = continous

	workclass = [0 for i in range(8)]
	if entries[1] == "Private":
		workclass[0] = 1
	elif entries[1] == "Self-emp-not-inc":
		workclass[1] = 1
	elif entries[1] == "Self-emp-inc":
		workclass[2] = 1
	elif entries[1] == "Federal-gov":
		workclass[3] = 1
	elif entries[1] == "Local-gov":
		workclass[4] = 1
	elif entries[1] == "State-gov":
		workclass[5] = 1
	elif entries[1] == "Without-pay":
		workclass[6] = 1
	else: #Never-worked
		workclass[7] = 1
	x.extend(workclass)
	x.append(float(entries[2]))

	education = [0 for i in range(16)]
	if entries[3] == "Bachelors":
		education[0] = 1
	elif entries[3] == "Some-college":
		education[1] = 1
	elif entries[3] == "11th":
		education[2] = 1
	elif entries[3] == "HS-grad":
		education[3] = 1
	elif entries[3] == "Prof-school":
		education[4] = 1
	elif entries[3] == "Assoc-acdm":
		education[5] = 1
	elif entries[3] == "Assoc-voc":
		education[6] = 1
	elif entries[3] == "9th":
		education[7] = 1
	elif entries[3] == "7th-8th":
		education[8] = 1
	elif entries[3] == "12th":
		education[9] = 1
	elif entries[3] == "Masters":
		education[10] = 1
	elif entries[3] == "1st-4th":
		education[11] = 1
	elif entries[3] == "10th":
		education[12] = 1
	elif entries[3] == "Doctorate":
		education[13] = 1
	elif entries[3] == "5th-6th":
		education[14] = 1
	if entries[3] == "Preschool":
		education[15] = 1

	x.extend(education)
	x.append(float(entries[4]))

	marital = [0 for i in range(7)]
	if entries[5] == "Married-civ-spouse":
		marital[0] = 1
	elif entries[5] == "Divorced":
		marital[1] = 1
	elif entries[5] == "Never-married":
		marital[2] = 1
	elif entries[5] == "Separated":
		marital[3] = 1
	elif entries[5] == "Widowed":
		marital[4] = 1
	elif entries[5] == "Married-spouse-absent":
		marital[5] = 1
	else:
		marital[6] = 1
	x.extend(marital)

	occupation = [0 for i in range(14)]
	if entries[6] == "Tech-support":
		occupation[0] = 1
	elif entries[6] == "Craft-repair":
		occupation[1] = 1
	elif entries[6] == "Other-service":
		occupation[2] = 1
	elif entries[6] == "Sales":
		occupation[3] = 1
	elif entries[6] == "Exec-managerial":
		occupation[4] = 1
	elif entries[6] == "Prof-specialty":
		occupation[5] = 1
	elif entries[6] == "Handlers-cleaners":
		occupation[6] = 1
	elif entries[6] == "Machine-op-inspct":
		occupation[7] = 1
	elif entries[6] == "Adm-clerical":
		occupation[8] = 1
	elif entries[6] == "Farming-fishing":
		occupation[9] = 1
	elif entries[6] == "Transport-moving":
		occupation[10] = 1
	elif entries[6] == "Priv-house-serv":
		occupation[11] = 1
	elif entries[6] == "Protective-serv":
		occupation[12] = 1
	else:
		occupation[13] = 1
	x.extend(occupation)

	relationship = [0 for i in range(6)]
	if entries[7] == "Wife":
		relationship[0] = 1
	elif entries[7] == "Own-child":
		relationship[1] = 1
	elif entries[7] == "Husband":
		relationship[2] = 1
	elif entries[7] == "Not-in-family":
		relationship[3] = 1
	elif entries[7] == "Other-relative":
		relationship[4] = 1
	else:
		relationship[5] = 1
	x.extend(relationship)

	race = [0 for i in range(5)]
	if entries[8] == "White":
		race[0] = 1
	elif entries[8] == "Asian-Pac-Islander":
		race[1] = 1
	elif entries[8] == "Amer-Indian-Eskimo":
		race[2] = 1
	elif entries[8] == "Other":
		race[3] = 1
	else:
		race[4] = 1
	x.extend(race)

	if (entries[9] == "Male"):
		x.extend([1,0])
	else:
		x.extend([0,1])

	x.append(float(entries[10]))
	x.append(float(entries[11]))
	x.append(float(entries[12]))

	native = [0 for i in range(42)]
	if entries[14] == "United-States":
		native[1] = 1
	elif entries[14] == "Cambodia":
		native[2] = 1
	elif entries[14] == "England":
		native[3] = 1
	elif entries[14] == "Puerto-Rico":
		native[4] = 1
	elif entries[14] == "Canada":
		native[5] = 1
	elif entries[14] == "Germany":
		native[6] = 1
	elif entries[14] == "Outlying-US(Guam-USVI-etc)":
		native[7] = 1
	elif entries[14] == "India":
		native[8] = 1
	elif entries[14] == "Japan":
		native[9] = 1
	elif entries[14] == "Greece":
		native[10] = 1
	elif entries[14] == "South":
		native[11] = 1
	elif entries[14] == "China":
		native[12] = 1
	elif entries[14] == "Cuba":
		native[13] = 1
	elif entries[14] == "Iran":
		native[14] = 1
	elif entries[14] == "Honduras":
		native[15] = 1
	elif entries[14] == "Philippines":
		native[16] = 1
	elif entries[14] == "Italy":
		native[17] = 1
	elif entries[14] == "Poland":
		native[18] = 1
	elif entries[14] == "Jamaica":
		native[19] = 1
	elif entries[14] == "Vietnam":
		native[20] = 1
	elif entries[14] == "Mexico":
		native[21] = 1
	elif entries[14] == "Portugal":
		native[22] = 1
	elif entries[14] == "Ireland":
		native[23] = 1
	elif entries[14] == "France":
		native[24] = 1
	elif entries[14] == "Dominican-Republic":
		native[25] = 1
	elif entries[14] == "Laos":
		native[26] = 1
	elif entries[14] == "Ecuador":
		native[27] = 1
	elif entries[14] == "Taiwan":
		native[28] = 1
	elif entries[14] == "Haiti":
		native[29] = 1
	elif entries[14] == "Columbia":
		native[30] = 1
	elif entries[14] == "Hungary":
		native[31] = 1
	elif entries[14] == "Guatemala":
		native[32] = 1
	elif entries[14] == "Nicaragua":
		native[33] = 1
	elif entries[14] == "Scotland":
		native[34] = 1
	elif entries[14] == "Thailand":
		native[35] = 1
	elif entries[14] == "Yugoslavia":
		native[36] = 1
	elif entries[14] == "El-Salvador":
		native[37] = 1
	elif entries[14] == "Trinadad&Tobago":
		native[38] = 1
	elif entries[14] == "Peru":
		native[39] = 1
	elif entries[14] == "Hong":
		native[40] = 1
	else:
		native[41] = 1

	return x



def getFeatureVectorBank(entries):
	x = []
	x.append(float(entries[0])) # age = continous

	job = [0 for i in range(12)]
	if entries[1] == "admin.":
		job[0] = 1
	elif entries[1] == "blue-collar":
		job[1] = 1
	elif entries[1] == "entrepreneur":
		job[2] = 1
	elif entries[1] == "housemaid":
		job[3] = 1
	elif entries[1] == "management":
		job[4] = 1
	elif entries[1] == "retired":
		job[5] = 1
	elif entries[1] == "self-employed":
		job[6] = 1
	elif entries[1] == "services":
		job[7] = 1
	elif entries[1] == "student":
		job[8] = 1
	elif entries[1] == "technician":
		job[9] = 1
	elif entries[1] == "unemployed":
		job[10] = 1
	else:
		job[11] = 1

	x.extend(job)

	martial = [0 for i in range(4)]
	if entries[2] == "divorced":
		martial[0] = 1
	elif entries[2] == "married":
		martial[1] = 1
	elif entries[2] == "single":
		martial[2] = 1
	else:
		martial[3] = 1

	education = [0 for i in range(8)]
	if entries[3] == "basic.4y":
		education[0] = 1
	elif entries[3] == "basic.6y":
		education[1] = 1
	elif entries[3] == "basic.9y":
		education[2] = 1
	elif entries[3] == "high.school":
		education[3] = 1
	elif entries[3] == "illiterate":
		education[4] = 1
	elif entries[3] == "professional.course":
		education[5] = 1
	elif entries[3] == "university.degree":
		education[6] = 1
	else:
		education[7] = 1
	x.extend(education)


	if entries[4] == "no":
		x.extend([1,0,0])
	elif entries[4] == "yes":
		x.extend([0,1,0])
	else:
		x.extend([0,0,1])

	if entries[5] == "no":
		x.extend([1,0,0])
	elif entries[5] == "yes":
		x.extend([0,1,0])
	else:
		x.extend([0,0,1])

	if entries[6] == "no":
		x.extend([1,0,0])
	elif entries[6] == "yes":
		x.extend([0,1,0])
	else:
		x.extend([0,0,1])

	if entries[7] == "telephone":
		x.append(0)
	else:
		x.append(1)

	month = [0 for i in range(12)]
	if entries[8] == "jan":
		month[0] = 1
	elif entries[8] == "feb":
		month[1] = 1
	elif entries[8] == "mar":
		month[2] = 1
	elif entries[8] == "apr":
		month[3] = 1
	elif entries[8] == "may":
		month[4] = 1
	elif entries[8] == "jun":
		month[5] = 1
	elif entries[8] == "jul":
		month[6] = 1
	elif entries[8] == "aug":
		month[7] = 1
	elif entries[8] == "sep":
		month[8] = 1
	elif entries[8] == "oct":
		month[9] = 1
	elif entries[8] == "nov":
		month[10] = 1
	else:
		month[11] = 1
	x.extend(month)

	day = [0 for i in range(5)]
	if entries[9] == "mon":
		day[0] = 1
	elif entries[9] == "tue":
		day[1] = 1
	elif entries[9] == "wed":
		day[2] = 1
	elif entries[9] == "thu":
		day[3] = 1
	else:
		day[4] = 1
	x.extend(day)

	#x.append(float(entries[10]))
	x.append(int(float(entries[11])*1000))
	x.append(int(float(entries[12])*1000))
	x.append(int(float(entries[13])*1000))

	if entries[14] == "failure":
		x.extend([1,0,0])
	elif entries[14] == "nonexistent":
		x.extend([0,1,0])
	else:
		x.extend([0,0,1])

	x.append(int(float(entries[15])*1000))
	x.append(int(float(entries[16])*1000))
	x.append(int(float(entries[17])*1000))
	x.append(int(float(entries[18])*1000))
	x.append(int(float(entries[19])*1000))

	return x
