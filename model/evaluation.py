from PIL import Image
import matplotlib.image as mpimg
import glob
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import Kmeans
import numpy as np
import os 



def SegmentImages(trainDataPath,trainGroundTruth):
    
    
    for filename in glob.glob(trainDataPath+"\\"+"*.jpg"): 
        #reading files from training data
        
        
        img = mpimg.imread(filename,format="jpg")
        
        rows = len(img)
        cols = len(img[0])
    
        labels , clusters = Kmeans.Kmeans(img,3)
        print("Image After Clustering ")
        plt.imshow(labels)
        plt.show()
        
        labelsAs1D = np.reshape(labels,154401)
        
        #print(f" {labelsAs1D}")

        
        
        
        #reading files from ground truth
        filename_w_ext = os.path.basename(filename)
        imageName, file_extension = os.path.splitext(filename_w_ext) 
        mat = scipy.io.loadmat(trainGroundTruth+"\\"+imageName+".mat")
        
        
        
        numberOfImages = len(mat['groundTruth'][0])
        fig , ax = plt.subplots(1,numberOfImages+1)
        ax[0].imshow(img)
        
        for k in range(0,numberOfImages,1):
            groundImage = mat['groundTruth'][0][k][0][0][0]
            ax[k+1].imshow(groundImage)
            
        plt.show()
        
        for i in range(0,numberOfImages,1):
            groundImage = mat['groundTruth'][0][i][0][0][0]
            groundTruthAs1D = np.reshape(groundImage,154401)
            matrix = pd.crosstab(labelsAs1D,groundTruthAs1D, rownames=['labels'], colnames=['img'])
            #print(matrix)
            #converting DataFrame to Numpy Array
            matrix = matrix.values
            fScore = Kmeans.getFScore(matrix)
            conditionalEntropy = Kmeans.getConditionalEntropy(matrix)
            print(f"Scores against groundTruth image {i}:")
            print("fScore is ",fScore)
            print("conditionalEntropy ",conditionalEntropy)
            print("\n\n")
            
        
if __name__ == '__main__':
    trainDataPath = os.getcwd() + "\\data\\images\\train"
    trainGroundTruth = os.getcwd() + "\\data\\groundTruth\\train"
    SegmentImages(trainDataPath,trainGroundTruth)