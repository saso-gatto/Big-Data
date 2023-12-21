import time
import sys
import math
import numpy as np

def readVectorsSeq(filename):
    with open(filename,'r') as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    return result

def readVectorsSeqNumpy(filename):
    with open(filename,'r') as f:
        result = [tuple(map(float, i.split(','))) for i in f]
    out = np.empty(len(result), dtype=object)
    out[:] = result
    return result
    
    
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)
