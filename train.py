import csv
import sys
from collections import defaultdict
import pandas as pd  
import numpy as np
import sklearn
from sklearn import preprocessing  

# trainSampleNum = 21048 
# testSampleNum = 4500 
# train_dir = "data/train.txt"
# test_dir = "data/test.txt"

trainSampleNum = 50 
testSampleNum = 20 
train_dir = "simpleData/train.txt"
test_dir = "simpleData/test.txt"

train_label_dir = "data/train.csv"
featureNum = 1024
classNum = 8
labelDict = {}

def loadTrainLabelDict(dataDir = train_label_dir):
    print("start loading data from %s..." %(dataDir))
    
    with open(dataDir,'r') as file:
        csvReader=csv.reader(file)
        count = 0
        for line in csvReader:
            if(count == 0):
                count += 1
                continue
            else:
                labelDict[str(line[0])] = int(line[1])
                
    print("finish load label from %s..." %(dataDir))
    return labelDict

def loadTrainData(dataDir = train_dir):
    print("start loading data from %s..." %(dataDir))

    data = np.zeros((trainSampleNum, featureNum))
    label = np.zeros(trainSampleNum)

    f = open(dataDir)  
    lines = f.readlines() 
    sampleCount = 0
    while sampleCount < trainSampleNum:  
        line = lines[sampleCount].split(' ')
        for index in range(len(line) - 1):
            if index == 0:
                label[sampleCount] = labelDict[line[index]]
            else:
                data[sampleCount, index-1] = float(line[index])
        sys.stdout.write('\r>> %d ' % (sampleCount))
        sampleCount += 1


    f.close()  

    print("finish load data from %s..." %(dataDir))
    return data, label

def loadTestData(dataDir = test_dir):
    print("start loading data from %s..." %(dataDir))

    data = np.zeros((trainSampleNum, featureNum))
    testNames = []

    f = open(dataDir)  
    lines = f.readlines() 
    sampleCount = 0
    while sampleCount < testSampleNum:  
        line = lines[sampleCount].split(' ')
        for index in range(len(line) - 1):
            if index == 0:
                testNames.append(str(line[index]))
            else:
                data[sampleCount, index-1] = float(line[index])
        sampleCount += 1
        sys.stdout.write('\r>> %d ' % (sampleCount))

    f.close()  

    print("finish load data from %s..." %(dataDir))
    return data, testNames
    
    

if __name__ == '__main__':
    loadTrainLabelDict()
    trainFeature, trainLabel = loadTrainData()
    testFeature, testNames = loadTestData()