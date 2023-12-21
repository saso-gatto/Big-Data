# Import Packages
from pyspark import SparkConf, SparkContext
import numpy as np
import time
import random
import sys
import math

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# MAIN PROGRAM
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def main():
    # Checking number of cmd line parameters
    assert len(sys.argv) == 5, "Usage: python Homework3.py filepath k z L"

    # Initialize variables
    filename = sys.argv[1]
    k = int(sys.argv[2])
    z = int(sys.argv[3])
    L = int(sys.argv[4])
    start = 0
    end = 0

    # Set Spark Configuration
    conf = SparkConf().setAppName('MR k-center with outliers')
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")

    # Read points from file
    start = time.time()
    inputPoints = sc.textFile(filename, L).map(lambda x : strToVector(x)).repartition(L).cache()
    print("--------------------------------------------------------------------")
    print(inputPoints)
    N = inputPoints.count()
    end = time.time()

    # Pring input parameters
    print("File : " + filename)
    print("Number of points N = ", N)
    print("Number of centers k = ", k)
    print("Number of outliers z = ", z)
    print("Number of partitions L = ", L)
    print("Time to read from file: ", str((end-start)*1000), " ms")

    # Solve the problem
    solution = MR_kCenterOutliers(inputPoints, k, z, L)

    # Compute the value of the objective function
    start = time.time()
    objective = computeObjective(inputPoints, solution, z)
    end = time.time()

    print("Objective function = ", objective)
    print("Time to compute objective function: ", str((end-start)*1000), " ms")
     



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# AUXILIARY METHODS
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method strToVector: input reading
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def strToVector(str):
    out = tuple(map(float, str.split(',')))
    return out



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method squaredEuclidean: squared euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def squaredEuclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return res



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method euclidean:  euclidean distance
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def euclidean(point1,point2):
    res = 0
    for i in range(len(point1)):
        diff = (point1[i]-point2[i])
        res +=  diff*diff
    return math.sqrt(res)

def farthest_first_traversal():
    print("something")

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method MR_kCenterOutliers: MR algorithm for k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def MR_kCenterOutliers(points, k, z, L):

    
    #------------- ROUND 1 ---------------------------
    # We extract in parallel the coreset using mapPartition.
    start_r1= time.time()
    coreset = points.cache().mapPartitions(lambda iterator: extractCoreset(iterator, k+z+1))
    elems =  coreset.collect()
    end_r1 = time.time()
    time_r1 = (end_r1-start_r1)*1000 # ms conversion
    # END OF ROUND 1
    
    #------------- ROUND 2 ---------------------------
    # We compute the weights on each coreset 
    start_r2 = time.time()
    coresetPoints = list()
    coresetWeights = list()
    # We merge each coreset in a list
    for i in elems:
        coresetPoints.append(i[0])
        coresetWeights.append(i[1])
    
    # Compute the final solution (run SeqWeightedOutliers with alpha=2)
    # Measure and print times taken by Round 1 and Round 2, separately
    # Return the final solution 
    # We use the method SeqWeightedOutilers from the HW2
    solution = SeqWeightedOutliers(coresetPoints,coresetWeights,k,z, alpha =2)
    end_r2 = time.time()
    time_r2 = (end_r2-start_r2)*1000 # ms conversion
    print("Time Round 1: ",time_r1," ms")
    print("Time Round 2: ",time_r2," ms")
    return solution



# Function to initialize our main data structures: 
# - dist_matrix: contains the distances of all inputPoints
# - index_map: dictionary to easily access index of a point (key= point: value = index)
# - computes min_distance: the minimum distance among the first k+z+1 points, which will be our first guess for radius.
def initialize_data_structures(inputPoints,k,z):
    dist_matrix =[[0 for x in range(len(inputPoints))] for y in range(len(inputPoints))]  # initialize distance matrix
    min_distance=1e6 # we first set this variable to a high value
    index_map = {} #dictionary

    # In the following loops we initialize the dictionary index_map, the distance matrix and find the minimum distance 
    # of the first k+z+1 points
    for i in range(len(inputPoints)):
        index_map[inputPoints[i]] = i
        for j in range(i,len(inputPoints)): # matrix is symmetric
            distance = euclidean(inputPoints[i],inputPoints[j])
            dist_matrix[i][j] =distance
            dist_matrix[j][i] =distance
            if(i< k+z+1 and j < k+z+1 and i!=j):
                if(distance < min_distance):
                    min_distance = distance
    return dist_matrix,index_map,min_distance

# This function returns the list of indices of the points belonging to a cluster (ball Bz) with center x
# and radius given. 
def define_bz(x,radius,Z,dist_matrix,index_map):
    bz_indices = []
    for i in range(len(Z)): # loops through the uncovered points
        distance = dist_matrix[index_map.get(x)][index_map.get(Z[i])]
        if (distance<=radius): #if point belongs to the cluster
            bz_indices.append(i) # we take the index
    return bz_indices

# This function returns the weight of a ball of points of center x and radius given.
def compute_weight (x,radius,Z,dist_matrix,weights,index_map):
    ball_weight=0
    for y in Z: #loops through uncovered points
        distance = dist_matrix[index_map.get(x)][index_map.get(y)]
        if (distance<=radius): #if point belongs to the cluster
            ball_weight+= weights[index_map.get(y)] #adds weight of the point 
    return ball_weight


# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method extractCoreset: extract a coreset from a given iterator
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def extractCoreset(iter, points):
    partition = list(iter)
    centers = kCenterFFT(partition, points)
    weights = computeWeights(partition, centers)
    c_w = list()
    for i in range(0, len(centers)):
        entry = (centers[i], weights[i])
        c_w.append(entry)
    # return weighted coreset
    return c_w
    
    
    
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method kCenterFFT: Farthest-First Traversal
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def kCenterFFT(points, k):
    random.seed(len(points))
    idx_rnd = random.randint(0, len(points)-1)
    centers = [points[idx_rnd]]
    related_center_idx = [idx_rnd for i in range(len(points))]
    dist_near_center = [squaredEuclidean(points[i], centers[0]) for i in range(len(points))]

    for i in range(k-1):
        new_center_idx = max(enumerate(dist_near_center), key=lambda x: x[1])[0] # argmax operation
        centers.append(points[new_center_idx])
        for j in range(len(points)):
            if j != new_center_idx:
                dist = squaredEuclidean(points[j], centers[-1])
                if dist < dist_near_center[j]:
                    dist_near_center[j] = dist
                    related_center_idx[j] = new_center_idx
            else:
                dist_near_center[j] = 0
                related_center_idx[j] = new_center_idx
    return centers



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeWeights: compute weights of coreset points
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeWeights(points, centers):
    weights = np.zeros(len(centers))
    for point in points:
        mycenter = 0
        mindist = squaredEuclidean(point,centers[0])
        for i in range(1, len(centers)):
            dist = squaredEuclidean(point,centers[i])
            if dist < mindist:
                mindist = dist
                mycenter = i
        weights[mycenter] = weights[mycenter] + 1
    return weights



# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method SeqWeightedOutliers: sequential k-center with outliers
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
#def SeqWeightedOutliers (points, weights, k, z, alpha):
#
def SeqWeightedOutliers(inputPoints,weights,k,z, alpha):

    dist_matrix,index_map,min_distance = initialize_data_structures(inputPoints,k,z)
    
    initial_guess= min_distance/2 # radius
    r = initial_guess
    num_iterations = 1
    while (True):
        Z = inputPoints.copy() #contains uncovered points
        solution = [] # contains cluster centers
        Wz = sum(weights)
        while (len(solution)<k and Wz>0):
            val_max = 0
            for x in Z:
                ball_weight=0
                radius = (1+2*alpha)*r
                ball_weight = compute_weight(x,radius,Z,dist_matrix,weights,index_map)
                if (ball_weight > val_max):
                    val_max=ball_weight
                    new_center = x
            solution.append(new_center)

            radius = (3+4*alpha)*r
            # We delete from Z the points belonging to the cluster with center new_center and radius (3+4*alpha)*r
            # and remove their weights from Wz. The function define_bz returns the indices of points in the ball,
            # and we go through them in reversed order so to safely remove them from Z (from right to left) 
            for index_z in reversed(define_bz(new_center,radius,Z,dist_matrix,index_map)):
                Wz-=weights[index_map.get(Z[index_z])]
                del Z[index_z]


        if(Wz <=z):
            print("Input size n = ",len(inputPoints))
            print("Number of centers k = ",k)
            print("Number of outliers z = ",z)
            print("Initial guess = ",initial_guess)
            print("Final guess = ",r)
            print("Number of guesses = ",num_iterations)
            return solution
        else:
            num_iterations +=1
            r=2*r #updates radius


def distances_objective(iter, centers):
    
    distances = []
    points = list(iter)

    # For each input point, we compute its distance from its cluster center (by taking the minimum distance among
    # all cluster centers).  
    for x in points: 
        min_d=1e6
        for center in centers: 
            dist = euclidean(x,center)
            if (dist<min_d):
                min_d = dist
        distances.append(min_d) #contains the distances of all points from their relative cluster center
    return distances


def printRDD(rdd):
    dataColl=rdd.collect()
    for row in dataColl:
        print(row)

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Method computeObjective: computes objective function
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
def computeObjective(points,centers,z):
    # We compute the objective function in parallel
    distances = (points.mapPartitions(lambda iterator: distances_objective(iterator, centers))        
                       .sortBy(lambda x:-x) # Groups partitions by popularity and sorts them in descending order
                       .zipWithIndex().map(lambda x: (x[1], x[0])).lookup(z)
                )

    return (distances[0]) # we return the largest distance 




# Just start the main program
if __name__ == "__main__":
    main()

