import numpy as np
def isEqual(arr1,arr2):
    flag = 1
    for i in range(0, len(arr1), 1):
        for j in range(0, 3, 1):
            if arr1[i][j] != arr2[i][j]:
                flag = 0
                break
    return flag

def intializeArray(size):
    
    arr = []
    for i in range (0,size,1):
        new = []
        for j in range (0, 3):
            new.append(0)
        arr.append(new)
    return arr
            
    
            
def eucl_dist(a, b):
    a = a.astype(float)
    b = b.astype(float)
    return np.linalg.norm(a - b)
