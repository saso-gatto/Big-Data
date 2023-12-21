import InputCode 
import sys
import random as rand
from operator import add
import time
import numpy as np


def define_bz(x,radius,Z,inputPoints,dist_matrix):
    points = []
    index_x = inputPoints.index(x)
    for y in Z:
        index_y = inputPoints.index(y)
        distance = dist_matrix[index_x,index_y]
        if (distance<=radius):
            points.append(y)
    return points
    
def SeqWeightedOutliers(inputPoints,weights,k,z, alpha =0):
    # First, we declare a list of solution where we'll add the set of k center which we'll compute
    #dist_matrix =[[0 for x in range(len(inputPoints))] for y in range(len(inputPoints))]
    dist_matrix = np.zeros((len(inputPoints),len(inputPoints)))
    first_distances =[]
    count = 0
    for i in range(len(inputPoints)):
        #count+=1
        for j in range(len(inputPoints)):
            count+=1
            dist_matrix[i,j] =InputCode.euclidean(inputPoints[i],inputPoints[j])
            if(count <= k+z+1 and i!=j):
                first_distances.append(dist_matrix[i,j])

    print(dist_matrix.shape) 
    initial_guess = min(first_distances)/2 # radius
    r = initial_guess
    num_iterations = 1
   
    while (True):
        Z = inputPoints.copy()
        S = []
        Wz = sum(weights)
        while (len(S)<k and Wz>0):
            val_max = 0
            #ball_weight =0
            for x in inputPoints:
                ball_weight = 0
                #val_max=0
                bz = define_bz(x,(1+2*alpha)*r,Z,inputPoints,dist_matrix)
                for y in bz:
                    ball_weight += weights[inputPoints.index(y)]
                if (ball_weight > val_max):
                    print(f"{ball_weight} > {val_max}")
                    val_max=ball_weight
                    new_center = x
            S.append(new_center)
            bz = define_bz(new_center,(3+4*alpha)*r,Z,inputPoints,dist_matrix)
            for y in bz:
                Z.remove(y)
                Wz-=weights[inputPoints.index(y)]
        if(Wz <=z):
            return S,num_iterations,r,initial_guess   
        else:
            print("Num it",num_iterations)
            num_iterations +=1
            r=2*r
                    

def ComputeObjective(inputPoints,S,z,r):
    
    distances = []
    print(len(S))
    i = 0
    #print("R is: ",r)
    distances=[]

    for x in inputPoints:
        min_dis = 1e6
        for s in S:
            dis = InputCode.euclidean(x,s)
            if (x!=s and dis<=min_dis):
                #print(f"{dis} <= {min_dis}")
                min_dis = dis
        if (min_dis not in distances and min_dis != 1e6):
            distances.append(min_dis)

    #for s in S: # for each center found
    #    print(f"s is {s}")
    #    for x in inputPoints: #go through the points and take the ones inside the cluster with center s
    #        print(f"x is {x}")
    #        dis = InputCode.euclidean(x,s)
    #        print(f"Distance {dis}")
    #        if (x!=s and dis<=r):
    #            # if inside the cluster
    #            distances.append(dis)
  #  print(f"z {z}")        
    distances.sort()
  #  print("Distances is",distances)

    for i in range(z):
        distances.pop()
    #print("*******************dopo********************\n",(distances))
    last = distances.pop()
    return last


def main():
        
    # INPUT READING
    # Path to dataset
    data_path = sys.argv[1]
    # Number of center
    k = int(sys.argv[2])
    #assert k.isdigit(), "H must be an integer"
    # Nunmber of allowed outliers
    z = int(sys.argv[3])

    # Reading inputs from data_apth
    inputPoints = InputCode.readVectorsSeq(data_path)
    weights = [1]*len(inputPoints)
    start= time.time()
    solution,num_iterations,r,initial_guess = SeqWeightedOutliers(inputPoints,weights,k,z,0)
    end=time.time()
    running_time = end-start
  #  print("Ho finito")
    objective = ComputeObjective(inputPoints,solution,z,r)
   # print(f"Solution : {solution}")
    print(f"Input size n = {len(inputPoints)}")
    print(f"Number of centers k = {k}")
    print(f"Number of outliers z = {z}")
    print(f"Initial guess = {initial_guess}")
    print(f"Final guess = {r}")
    print(f"Number of guesses = {num_iterations}")
    print(f"Objective function = {objective}")
    print(f"Time of SeqWeightedOutliers = {running_time}")


if __name__ == "__main__":
	main()