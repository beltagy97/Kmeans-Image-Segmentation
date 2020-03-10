import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import pandas as pd
import math
import utils

def Kmeans(arr, K):

    centroids = generateCentroids(arr,K)
    while(True):
        labels = np.zeros((len(arr),len(arr[0])))
        labels, newMeans = getLabels(arr,labels,len(arr),len(arr[0]),K,centroids)
        newMeans = np.array(newMeans)
        
        if utils.isEqual(newMeans,centroids):
            break
        else:
            centroids = newMeans
    
    return labels , newMeans
    


def assignLabels(pixel,K,centroids):
    min = 1000000
    index = -1
    for i in range(0,K,1):
        temp = utils.eucl_dist(pixel,centroids[i])
        if temp < min:
            min = temp
            index = i
            
    return index


def getLabels(arr,labels,rows,columns,K,centroids):
    
    accumlatorClustering = utils.intializeArray(len(centroids))
    accumlatorClustering = np.array(accumlatorClustering)
    
    counters  = []
    newMeans  = []
    for k in range(0,len(centroids),1):
        counters.append(0)
        newMeans.append(0)
    
    counters = np.array(counters)
    
    for i in range(0,rows,1):
        for j in range(0,columns,1):
            labels[i][j] = assignLabels(arr[i][j],K,centroids)
            accumlatorClustering[int(labels[i][j])] = accumlatorClustering[int(labels[i][j])] + arr[i][j]
            counters[int(labels[i][j])] = counters[int(labels[i][j])] + 1
            
    for n in range(0,len(centroids),1):
        if(counters[n] == 0 ):
            newMeans[n] = accumlatorClustering[n]
        else:
            newMeans[n] = accumlatorClustering[n] // counters[n]
        
        
    
            
    return labels,newMeans

def generateCentroids(arr,K):
    centroids = []
    for i in range(0,K,1):
        centroids.append(randomizePoints(arr))
    centroids = np.array(centroids)
    return centroids


def randomizePoints(arr):
    return arr[np.random.randint(0,len(arr))][np.random.randint(0,len(arr[0]))]   


def getConditionalEntropy(matrix):
    entropy = 0
    for i in range(0, len(matrix), 1):
        rowSum = sum(matrix[i])
        classEntropy = 0
        for j in range(0, len(matrix[0]), 1):
            if(matrix[i][j] == 0):
                classEntropy = classEntropy + 0
            else:
                element = (matrix[i][j]/rowSum) * \
                    math.log2(matrix[i][j]/rowSum)
                classEntropy = classEntropy - element
        temp = (rowSum / 154401)*classEntropy
        entropy = entropy + temp

    return entropy

def getFScore(matrix):
    fScore = 0
    for i in range(0,len(matrix),1):
        rowSum = sum(matrix[i])
        idx = matrix[i].argsort()[::-1]
        #max element
        maxElement = matrix[i][idx[0]]
        sumArray = np.sum(matrix,axis=0)
        colSum = sumArray[idx[0]]
        score = ((2*maxElement) /(rowSum+colSum))
        fScore = fScore +score
    return (1/len(matrix))*fScore
        
    